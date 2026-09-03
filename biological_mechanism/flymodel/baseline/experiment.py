"""Five-condition continual routing training and 26-point evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import hashlib, random
import numpy as np
import torch
from torch.nn import functional as F
from .config import Config, CONDITIONS
from .model import FlyModelV4

@dataclass(frozen=True)
class Training:
    learning_rate:float
    device:str="cuda"

def tensor_hash(x:torch.Tensor)->str: return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def set_seed(seed:int): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
def region_to_expert(order:np.ndarray,n_regions:int)->np.ndarray:
    if order.shape!=(n_regions,) or sorted(order.tolist())!=list(range(n_regions)): raise ValueError("invalid stage order")
    result=np.empty(n_regions,np.int64)
    for expert,region in enumerate(order): result[region]=expert
    return result
def online_route(kc,sums,counts,active): return (F.normalize(kc,dim=1)@F.normalize((sums[:active]/counts[:active,None]).to(kc.dtype),dim=1).T).argmax(1)
def stream_order(labels:np.ndarray,order:np.ndarray,seed:int):
    rng=np.random.default_rng(seed); parts=[]; boundaries=[]; total=0
    for region in order:
        indices=np.flatnonzero(labels==region); rng.shuffle(indices); parts.append(indices); total+=len(indices); boundaries.append(total)
    result=np.concatenate(parts)
    if len(result)!=len(labels) or np.unique(result).size!=len(labels): raise RuntimeError("stream conservation failed")
    return result,boundaries

@torch.no_grad()
def evaluate(model,test,test_labels,test_regions,condition,artifact,device,position,stage,active,cfg):
    model.eval(); predictions=[]; routes=[]; mapping=artifact["mapping"]
    before=(tensor_hash(artifact["sums"]),tensor_hash(artifact["counts"]))
    for start in range(0,len(test),cfg.evaluation_batch_size):
        x=torch.from_numpy(np.array(test[start:start+cfg.evaluation_batch_size],dtype=np.float32,copy=True)).to(device); kc=model.encode(x); logits=model.logits(kc); region=torch.as_tensor(np.asarray(test_regions[start:start+len(x)],dtype=np.int64),device=device)
        if condition=="shared": route=torch.zeros(len(x),dtype=torch.long,device=device); chosen=logits[:,0]
        elif condition=="unrouted_5": route=torch.full((len(x),),-1,dtype=torch.long,device=device); chosen=logits.mean(1)
        elif condition=="random_routed_5": route=(kc@artifact["projection"]).argmax(1); chosen=logits[torch.arange(len(x),device=device),route]
        elif condition=="online_routed_5" and active: route=online_route(kc,artifact["sums"],artifact["counts"],active); chosen=logits[torch.arange(len(x),device=device),route]
        elif condition=="online_routed_5": route=torch.full((len(x),),-1,dtype=torch.long,device=device); chosen=logits.mean(1)
        else: route=mapping[region]; chosen=logits[torch.arange(len(x),device=device),route]
        predictions.append(chosen.argmax(1).cpu()); routes.append(route.cpu())
    pred=torch.cat(predictions); route=torch.cat(routes); y=torch.as_tensor(np.asarray(test_labels,dtype=np.int64)); regions=torch.as_tensor(np.asarray(test_regions,dtype=np.int64)); true=mapping.cpu()[regions]
    if before!=(tensor_hash(artifact["sums"]),tensor_hash(artifact["counts"])): raise RuntimeError("evaluation mutated online state")
    matrix=torch.zeros(cfg.n_regions,cfg.n_regions,dtype=torch.long)
    for a,b in zip(route,true,strict=True):
        if a>=0: matrix[a,b]+=1
    routed=route>=0; route_accuracy=float((route[routed]==true[routed]).float().mean()) if condition in ("random_routed_5","oracle_routed_5") or condition=="online_routed_5" and active>0 else None
    return {"position":position,"stage":stage,"overall_accuracy":float((pred==y).float().mean()),"region_accuracy":[float((pred[regions==r]==y[regions==r]).float().mean()) for r in range(cfg.n_regions)],"routing":{"route_accuracy":route_accuracy,"route_accuracy_denominator":int(routed.sum()) if route_accuracy is not None else 0,"route_counts":torch.bincount(route[routed],minlength=cfg.n_regions).tolist() if routed.any() else None,"route_matrix":matrix.tolist(),"active_prototypes":active if condition=="online_routed_5" else None},"audit":{"sample_count":len(y),"label_hash":tensor_hash(y),"region_hash":tensor_hash(regions),"online_state_before":before,"read_only":True}}

def continual_metrics(records,stream_length,order,test_counts,points_per_region:int):
    positions=np.asarray([r["position"] for r in records],float); acc=np.asarray([r["region_accuracy"] for r in records]); counts=np.asarray(test_counts,float); order=np.asarray(order); auc=[]; ends=[]; current=[]; old=[]
    for stage in range(len(order)):
        ix=np.arange(stage*points_per_region,(stage+1)*points_per_region+1); seen=order[:stage+1]; values=np.average(acc[ix][:,seen],axis=1,weights=counts[seen]); auc.append(float(np.trapezoid(values,positions[ix])/(positions[ix[-1]]-positions[ix[0]]))); ends.append(float(values[-1])); current.append(float(acc[ix[-1],order[stage]]));
        if stage: old.append(float(np.average(acc[ix[-1],order[:stage]],weights=counts[order[:stage]])))
    forgetting=[float(acc[(i+1)*points_per_region:,r].max()-acc[-1,r]) for i,r in enumerate(order[:-1])]; lengths=np.diff(np.r_[0,positions[points_per_region::points_per_region]]); overall=np.asarray([r["overall_accuracy"] for r in records])
    return {"seen_anytime_auc":float(np.average(auc,weights=lengths)),"stage_seen_auc":auc,"stage_seen_accuracy":ends,"all_regions_auc":float(np.trapezoid(overall,positions)/stream_length),"final_accuracy":float(np.average(acc[-1],weights=counts)),"worst_region_accuracy":float(acc[-1].min()),"current_adaptation":float(np.mean(current)),"old_retention":float(np.mean(old)),"average_forgetting":float(np.mean(forgetting)),"region_forgetting":forgetting}

def train(dataset:dict[str,Any],condition:str,training:Training,seed:int,cfg:Config,realization:int=0):
    if condition not in CONDITIONS: raise ValueError(condition)
    set_seed(seed); device=torch.device(training.device); model=FlyModelV4(seed,1 if condition=="shared" else cfg.n_regions,dataset["orn_pn"],dataset["pn_kc"],cfg,training.device); initial=[tensor_hash(h.weight) for h in model.heads]; optimizer=torch.optim.Adam(model.parameters(),lr=training.learning_rate)
    order=np.asarray(dataset["stage_order"]); stream,boundaries=stream_order(dataset["train_regions"],order,seed+1000*realization); mapping=torch.as_tensor(region_to_expert(order,cfg.n_regions),device=device); projection=np.random.default_rng(seed+140000).standard_normal((cfg.n_kc,cfg.n_regions)).astype(np.float32); artifact={"mapping":mapping,"projection":torch.as_tensor(projection,device=device),"sums":torch.zeros((cfg.n_regions,cfg.n_kc),dtype=torch.float64,device=device),"counts":torch.zeros(cfg.n_regions,dtype=torch.long,device=device)}
    targets={s+round((e-s)*p/cfg.evaluation_points_per_region) for s,e in zip((0,*boundaries[:-1]),boundaries,strict=True) for p in range(1,cfg.evaluation_points_per_region+1)}; records=[evaluate(model,dataset["test_samples"],dataset["test_labels"],dataset["test_regions"],condition,artifact,device,0,None,0,cfg)]
    position=0; noise=torch.Generator(device=device).manual_seed(seed+150000); losses=[]; running=[]; updates=torch.zeros(cfg.n_regions,dtype=torch.long); route_matrix=torch.zeros(cfg.n_regions,cfg.n_regions,dtype=torch.long); flat=dataset["train_samples"]
    while position<len(stream):
        event=min((x for x in (*targets,*boundaries) if x>position),default=len(stream)); end=min(position+cfg.batch_size,event); indices=stream[position:end]; stage=next(i for i,b in enumerate(boundaries) if position<b); x=torch.from_numpy(np.array(flat[indices],dtype=np.float32,copy=True)).to(device); y=torch.as_tensor(np.asarray(dataset["train_labels"][indices],dtype=np.int64),device=device); true=torch.full((len(x),),stage,dtype=torch.long,device=device)
        model.train(); optimizer.zero_grad(set_to_none=True); kc=model.encode(x,cfg.train_noise_sigma,noise); logits=model.logits(kc)
        if condition=="shared": routes=torch.zeros_like(y); loss=F.cross_entropy(logits[:,0],y)
        elif condition=="unrouted_5": routes=torch.full_like(y,-1); loss=torch.stack([F.cross_entropy(logits[:,i],y) for i in range(cfg.n_regions)]).mean()
        elif condition=="random_routed_5": routes=(kc@artifact["projection"]).argmax(1); loss=F.cross_entropy(logits[torch.arange(len(y),device=device),routes],y)
        else: routes=true; loss=F.cross_entropy(logits[torch.arange(len(y),device=device),routes],y)
        loss.backward(); optimizer.step(); running.append(float(loss.detach())); valid=routes>=0; updates+=torch.bincount(routes[valid].cpu(),minlength=cfg.n_regions)
        for a,b in zip(routes[valid].cpu(),true[valid].cpu(),strict=True): route_matrix[a,b]+=1
        if condition in ("online_routed_5","oracle_routed_5"): artifact["sums"][stage]+=kc.detach().double().sum(0); artifact["counts"][stage]+=len(kc)
        position=end
        if position in targets:
            current=next(i for i,b in enumerate(boundaries) if position<=b); records.append(evaluate(model,dataset["test_samples"],dataset["test_labels"],dataset["test_regions"],condition,artifact,device,position,current,current+1 if condition=="online_routed_5" else 0,cfg)); losses.append({"position":position,"stage":current,"mean_loss":float(np.mean(running))}); running=[]
    expected_evaluations=1+cfg.n_regions*cfg.evaluation_points_per_region
    if len(records)!=expected_evaluations or [r["position"] for r in records]!=[0,*sorted(targets)]: raise RuntimeError(f"expected {expected_evaluations} registered evaluations")
    return {"schema_version":2,"experiment":"olfactory_baseline_regions","status":"complete","condition":condition,"seed":seed,"realization":realization,"training":{"learning_rate":training.learning_rate,"device":training.device},"stream":{"length":len(stream),"stage_order":order.tolist(),"stage_boundaries":boundaries,"stage_lengths":np.diff((0,*boundaries)).tolist()},"evaluation":{"expected_count":expected_evaluations,"records":records},"loss_trajectory":losses,"summary":continual_metrics(records,len(stream),order,dataset["metadata"]["counts"]["test"],cfg.evaluation_points_per_region),"routing_diagnostics":{"training_counts":updates.tolist(),"route_matrix":route_matrix.tolist(),"prototype_counts":artifact["counts"].tolist()},"initial_expert_hashes":initial,"initial_expert0_hash":initial[0],"audits":{"stream_conservation":True,"evaluation_read_only":True,"online_causal":True,"paired_initialization":True}}
