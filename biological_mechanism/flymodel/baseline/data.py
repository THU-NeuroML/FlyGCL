"""Uniform prototype dataset, frozen power regions, and FlyWire sampling."""
from __future__ import annotations
import csv, json, os, re, shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator
import numpy as np
from .audit import array_record, atomic_json, canonical_sha256, file_sha256
from .config import Config

ORN_TYPE = re.compile(r"^ORN_([^,]+)$")
PN_TYPE = re.compile(r"^([^_+]+)_(?:adPN|lPN|vPN|lvPN|ilPN|il2PN|l2PN)$")

def _rng(cfg: Config, seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([*cfg.seed_namespace, seed, stream]))

def nearest_labels(x: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    x2 = np.sum(x*x, axis=1, dtype=np.float32)[:, None]; p2 = np.sum(prototypes*prototypes, axis=1, dtype=np.float32)[None]
    return np.argmin(x2+p2-np.float32(2)*(x@prototypes.T), axis=1).astype(np.uint8)

def power_labels(x: np.ndarray, centers: np.ndarray, prices: np.ndarray) -> np.ndarray:
    x2=np.sum(x*x,axis=1,dtype=np.float32)[:,None]; c2=np.sum(centers*centers,axis=1,dtype=np.float32)[None]
    return np.argmin(x2+c2-np.float32(2)*(x@centers.T)+prices[None],axis=1).astype(np.uint8)

def blocks(array: np.ndarray, size: int) -> Iterator[tuple[int,np.ndarray]]:
    for start in range(0,len(array),size): yield start,np.asarray(array[start:start+size],dtype=np.float32)

def _region_pass(train: np.ndarray, centers: np.ndarray, prices: np.ndarray, cfg: Config, sums: bool=False):
    counts=np.zeros(cfg.n_regions,np.int64); totals=np.zeros((cfg.n_regions,cfg.odor_dim),np.float64) if sums else None
    for _,x in blocks(train,cfg.region_chunk_size):
        labels=power_labels(x,centers,prices); counts+=np.bincount(labels,minlength=cfg.n_regions)
        if totals is not None:
            for r in range(cfg.n_regions):
                selected=x[labels==r]
                if len(selected): totals[r]+=selected.sum(0,dtype=np.float64)
    return counts,totals

def fit_regions(train: np.ndarray, seed: int, cfg: Config):
    rng=_rng(cfg,seed,3); candidates=np.asarray(train[rng.choice(len(train),min(50_000,len(train)),False)],dtype=np.float32)
    chosen=[int(np.argmax(np.sum((candidates-candidates.mean(0))**2,axis=1)))]; nearest=np.full(len(candidates),np.inf,np.float32)
    while len(chosen)<cfg.n_regions:
        nearest=np.minimum(nearest,np.sum((candidates-candidates[chosen[-1]])**2,axis=1,dtype=np.float32)); nearest[chosen]=-1; chosen.append(int(np.argmax(nearest)))
    centers=candidates[chosen].copy(); prices=np.zeros(cfg.n_regions,np.float64); target=len(train)/cfg.n_regions; history=[]
    for iteration in range(cfg.region_iterations):
        for _ in range(cfg.region_price_steps):
            counts,_=_region_pass(train,centers,prices,cfg); scale=max(float(np.mean(np.sum((centers-centers.mean(0))**2,axis=1))),1e-6); prices+=cfg.region_price_rate*scale*(counts-target)/target; prices-=prices.mean()
        counts,totals=_region_pass(train,centers,prices,cfg,True); centers=(totals/counts[:,None]).astype(np.float32); history.append({"iteration":iteration+1,"counts":counts.tolist(),"fractions":(counts/counts.sum()).tolist(),"prices":prices.tolist()})
    counts,_=_region_pass(train,centers,prices,cfg); fractions=counts/counts.sum()
    if np.any(fractions<cfg.min_region_fraction)|np.any(fractions>cfg.max_region_fraction): raise RuntimeError(f"region balance failed: {fractions}")
    return centers,prices.astype(np.float32),history

def extract_flywire_weights(raw_root: Path) -> tuple[np.ndarray,dict[str,Any]]:
    paths=[raw_root/"classification.csv",raw_root/"consolidated_cell_types.csv",raw_root/"connections_princeton.csv"]
    for path in paths:
        if not path.is_file(): raise FileNotFoundError(path)
    classes={"olfactory":set(),"ALPN":set()}
    with paths[0].open(newline="",encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["class"] in classes: classes[row["class"]].add(row["root_id"])
    orns={}; pns={}
    with paths[1].open(newline="",encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rid=row["root_id"]; primary=row["primary_type"].strip(); match=ORN_TYPE.fullmatch(primary) if rid in classes["olfactory"] else PN_TYPE.fullmatch(primary) if rid in classes["ALPN"] else None
            if match: (orns if rid in classes["olfactory"] else pns)[rid]=match.group(1)
    shared=sorted(set(orns.values())&set(pns.values()))
    if len(shared)!=50: raise RuntimeError(f"expected 50 shared glomeruli, found {len(shared)}")
    pairs=defaultdict(int)
    with paths[2].open(newline="",encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            og=orns.get(row["pre_root_id"]); pg=pns.get(row["post_root_id"])
            if og is not None and og==pg and og in shared: pairs[(row["pre_root_id"],row["post_root_id"])]+=int(row["syn_count"])
    values=np.asarray(sorted(pairs.values()),dtype=np.float32)
    if len(values)!=9_754 or float(values.min())!=5: raise RuntimeError(f"expected 9754 FlyWire edges with minimum 5, found n={len(values)} min={values.min() if len(values) else None}")
    return values,{"shared_glomeruli":shared,"edge_count":len(values),"minimum":float(values.min()),"sources":{p.name:{"sha256":file_sha256(p),"bytes":p.stat().st_size} for p in paths}}

def generate_pn_kc(seed:int,cfg:Config)->tuple[np.ndarray,dict[str,Any]]:
    path=cfg.pn_kc_stats_path
    if not path.is_file(): raise FileNotFoundError(path)
    stats=json.loads(path.read_text(encoding="utf-8")); source=np.asarray(stats["syn_count_nonzero"],dtype=np.float64); density=float(stats["density"])
    if source.ndim!=1 or not len(source) or np.any(source<=0) or not 0<density<=1: raise ValueError("invalid empirical PN-KC statistics")
    rng=np.random.default_rng(seed); shape=(cfg.n_pn,cfg.n_kc); mask=rng.random(shape)<density; empty=~mask.any(axis=0)
    if empty.any():
        columns=np.flatnonzero(empty); mask[rng.integers(0,cfg.n_pn,size=len(columns)),columns]=True
    source_rms=float(np.sqrt(np.mean(source*source))); sampled=np.random.default_rng(seed+20_000).choice(source,size=shape,replace=True)/source_rms; matrix=np.where(mask,sampled,0).astype(np.float32); fan_in=np.count_nonzero(matrix,axis=0)
    metadata={"rule":"Bernoulli mask at observed density; one edge added to empty KCs; empirical nonzero syn_count resampled with replacement; divide by complete source nonzero RMS","source":{"path":str(path),"sha256":file_sha256(path),"bytes":path.stat().st_size},"source_shape":stats.get("shape"),"source_nonzero_count":len(source),"source_density":density,"source_nonzero_rms":source_rms,"target_nonzero_count":int(mask.sum()),"target_density":float(mask.mean()),"fan_in":{"minimum":int(fan_in.min()),"mean":float(fan_in.mean()),"maximum":int(fan_in.max()),"empty_kcs":int(np.count_nonzero(fan_in==0))}}
    return matrix,metadata

def prepare_seed(seed: int, cfg: Config) -> Path:
    destination=cfg.data_root/f"seed_{seed}"; temporary=cfg.data_root/f".seed_{seed}.{os.getpid()}.tmp"
    if temporary.exists(): shutil.rmtree(temporary)
    temporary.mkdir(parents=True); prototypes=_rng(cfg,seed,0).random((cfg.n_classes,cfg.odor_dim),dtype=np.float32)
    arrays={"prototypes.npy":prototypes}
    for split,n,stream in (("train",cfg.n_train,1),("test",cfg.n_test,2)):
        samples=np.lib.format.open_memmap(temporary/f"{split}_samples.npy",mode="w+",dtype=np.float32,shape=(n,cfg.odor_dim)); labels=np.lib.format.open_memmap(temporary/f"{split}_labels.npy",mode="w+",dtype=np.uint8,shape=(n,))
        rng=_rng(cfg,seed,stream)
        for start in range(0,n,cfg.region_chunk_size):
            end=min(start+cfg.region_chunk_size,n); samples[start:end]=rng.random((end-start,cfg.odor_dim),dtype=np.float32); labels[start:end]=nearest_labels(samples[start:end],prototypes)
        samples.flush(); labels.flush(); del samples,labels
    train=np.load(temporary/"train_samples.npy",mmap_mode="r"); test=np.load(temporary/"test_samples.npy",mmap_mode="r"); centers,prices,history=fit_regions(train,seed,cfg)
    arrays.update({"centers.npy":centers,"prices.npy":prices,"stage_order.npy":_rng(cfg,seed,4).permutation(cfg.n_regions).astype(np.uint8)})
    for name,array in arrays.items(): np.save(temporary/name,array,allow_pickle=False)
    for split,array in (("train",train),("test",test)):
        out=np.lib.format.open_memmap(temporary/f"{split}_regions.npy",mode="w+",dtype=np.uint8,shape=(len(array),)); counts=np.zeros(cfg.n_regions,np.int64)
        for start,x in blocks(array,cfg.region_chunk_size): lab=power_labels(x,centers,prices); out[start:start+len(lab)]=lab; counts+=np.bincount(lab,minlength=cfg.n_regions)
        out.flush(); del out; arrays[f"{split}_counts"]=counts
    values,flywire=extract_flywire_weights(cfg.raw_flywire_root); weights=_rng(cfg,seed,5).choice(values,size=(cfg.n_pn,cfg.orn_per_channel),replace=True); weights=(weights/weights.sum(1,keepdims=True)).astype(np.float32); np.save(temporary/"orn_pn.npy",weights,allow_pickle=False)
    pn_kc,pn_kc_metadata=generate_pn_kc(seed,cfg); np.save(temporary/"pn_kc.npy",pn_kc,allow_pickle=False)
    files={p.name:array_record(p,np.load(p,mmap_mode="r",allow_pickle=False)) for p in temporary.glob("*.npy")}
    metadata={"schema_version":2,"experiment":"olfactory_fixed_encoder_assets","status":"complete","seed":seed,"config":cfg.to_dict(),"sampling":{"prototypes":"100 iid Uniform[0,1) float32 vectors","samples":"global iid Uniform[0,1) float32","labels":"explicit nearest prototype by squared Euclidean distance","regions":"training-only fitted power diagram; test assignment frozen"},"region_history":history,"counts":{"train":arrays["train_counts"].tolist(),"test":arrays["test_counts"].tolist()},"flywire":flywire,"orn_pn":{"rule":"1300 empirical edge samples arranged 50x26; each PN row sum normalized","shape":[cfg.n_pn,cfg.orn_per_channel]},"pn_kc":pn_kc_metadata,"files":files}; metadata["fingerprint"]=canonical_sha256(metadata); atomic_json(temporary/"metadata.json",metadata)
    if destination.exists(): shutil.rmtree(destination)
    destination.parent.mkdir(parents=True,exist_ok=True); os.replace(temporary,destination); return destination

def load_seed(seed:int,cfg:Config)->dict[str,Any]:
    root=cfg.data_root/f"seed_{seed}"; metadata=json.loads((root/"metadata.json").read_text()); fingerprint=metadata.pop("fingerprint",None)
    if metadata.get("schema_version")!=2 or metadata.get("experiment")!="olfactory_fixed_encoder_assets" or metadata.get("status")!="complete" or metadata.get("seed")!=seed: raise RuntimeError("data metadata identity mismatch")
    if metadata.get("config")!=cfg.to_dict() or fingerprint!=canonical_sha256(metadata): raise RuntimeError("data configuration/fingerprint mismatch")
    metadata["fingerprint"]=fingerprint; expected={"prototypes.npy":((cfg.n_classes,cfg.odor_dim),np.dtype("float32")),"train_samples.npy":((cfg.n_train,cfg.odor_dim),np.dtype("float32")),"test_samples.npy":((cfg.n_test,cfg.odor_dim),np.dtype("float32")),"train_labels.npy":((cfg.n_train,),np.dtype("uint8")),"test_labels.npy":((cfg.n_test,),np.dtype("uint8")),"centers.npy":((cfg.n_regions,cfg.odor_dim),np.dtype("float32")),"prices.npy":((cfg.n_regions,),np.dtype("float32")),"stage_order.npy":((cfg.n_regions,),np.dtype("uint8")),"train_regions.npy":((cfg.n_train,),np.dtype("uint8")),"test_regions.npy":((cfg.n_test,),np.dtype("uint8")),"orn_pn.npy":((cfg.n_pn,cfg.orn_per_channel),np.dtype("float32")),"pn_kc.npy":((cfg.n_pn,cfg.n_kc),np.dtype("float32"))}
    if set(metadata.get("files",{}))!=set(expected): raise RuntimeError("data file manifest mismatch")
    arrays={}
    for name,(shape,dtype) in expected.items():
        path=root/name; record=metadata["files"][name]
        if not path.is_file() or file_sha256(path)!=record["sha256"]: raise RuntimeError(f"artifact mismatch: {name}")
        array=np.load(path,mmap_mode="r",allow_pickle=False)
        if array.shape!=shape or array.dtype!=dtype or record["shape"]!=list(shape) or record["dtype"]!=str(dtype): raise RuntimeError(f"shape/dtype mismatch: {name}")
        arrays[name[:-4]]=array
    if sorted(arrays["stage_order"].tolist())!=list(range(cfg.n_regions)): raise RuntimeError("stage order mismatch")
    if not np.allclose(arrays["orn_pn"].sum(1),1,atol=1e-6) or np.any(np.count_nonzero(arrays["pn_kc"],axis=0)==0): raise RuntimeError("connection asset validation failed")
    if metadata.get("pn_kc",{}).get("source",{}).get("sha256")!=file_sha256(cfg.pn_kc_stats_path): raise RuntimeError("PN-KC source provenance mismatch")
    for split in ("train","test"):
        samples=arrays[f"{split}_samples"]; labels=arrays[f"{split}_labels"]
        for start,x in blocks(samples,cfg.region_chunk_size):
            if not np.array_equal(nearest_labels(x,arrays["prototypes"]),labels[start:start+len(x)]): raise RuntimeError(f"{split} nearest-prototype labels mismatch")
    recomputed=np.empty(cfg.n_test,np.uint8)
    for start,x in blocks(arrays["test_samples"],cfg.region_chunk_size): recomputed[start:start+len(x)]=power_labels(x,arrays["centers"],arrays["prices"])
    if not np.array_equal(recomputed,arrays["test_regions"]): raise RuntimeError("frozen test region assignment mismatch")
    for split in ("train","test"):
        counts=np.bincount(arrays[f"{split}_regions"],minlength=cfg.n_regions)
        if counts.tolist()!=metadata["counts"][split] or counts.sum()!=getattr(cfg,f"n_{split}"): raise RuntimeError(f"{split} region counts mismatch")
    return {"root":root,"metadata":metadata,**arrays}
