"""GPU calibration and formal experiment CLI."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from .analysis import finalize_formal, finalize_lr
from .audit import atomic_json, canonical_sha256, file_sha256, runtime_identity, source_identity
from .config import CALIBRATION_SEED, CONDITIONS, FORMAL_SEEDS, LR_CANDIDATES, LR_CONDITIONS, Config, PROJECT_ROOT
from .data import load_seed, prepare_seed
from .experiment import Training, train

def run_name(seed:int,condition:str)->str: return f"seed{seed}_{condition}.json"
def lr_name(realization:int,condition:str,lr:float)->str: return f"seed{CALIBRATION_SEED}_realization{realization}_{condition}_lr{lr:.8g}.json"
def execute(cfg:Config,suite:str,seed:int,realization:int,condition:str,lr:float,device:str)->Path:
    dataset=load_seed(seed,cfg); metadata_path=dataset["root"]/"metadata.json"; identity={"suite":suite,"seed":seed,"realization":realization,"condition":condition,"learning_rate":lr}; configuration={"identity":identity,"protocol":cfg.to_dict(),"data":{"metadata":str(metadata_path),"sha256":file_sha256(metadata_path),"fingerprint":dataset["metadata"]["fingerprint"],"test_region_counts":dataset["metadata"]["counts"]["test"],"pn_kc_source":dataset["metadata"]["pn_kc"]["source"],"assets":{name:record["sha256"] for name,record in dataset["metadata"]["files"].items()}},"source":source_identity(PROJECT_ROOT),"runtime":runtime_identity(device)}
    if suite=="formal":
        selection=cfg.result_root/"lr"/"selection.json"; configuration["lr_selection"]={"path":str(selection),"sha256":file_sha256(selection)}
    fingerprint=canonical_sha256(configuration); output=cfg.result_root/("lr/runs" if suite=="lr" else "formal/runs")/(lr_name(realization,condition,lr) if suite=="lr" else run_name(seed,condition))
    if output.is_file():
        old=json.loads(output.read_text());
        if old.get("fingerprint",{}).get("sha256")!=fingerprint or old.get("status")!="complete": raise RuntimeError(f"invalid existing run: {output}")
        return output
    started=time.time(); result=train(dataset,condition,Training(lr,device),seed,cfg,realization); result["identity"]=identity; result["fingerprint"]={"algorithm":"sha256-canonical-json","sha256":fingerprint,"configuration":configuration}; result["provenance"]={"independent_package":True,"legacy_runtime_imports":False,"data":configuration["data"],"source":configuration["source"]}; result["runtime"]={**configuration["runtime"],"seconds":time.time()-started}; atomic_json(output,result); print(output,flush=True); return output
def selected_lr(cfg:Config)->float:
    payload=json.loads((cfg.result_root/"lr"/"selection.json").read_text())
    if payload.get("status")!="frozen" or payload.get("schema_version")!=2 or payload.get("experiment")!="olfactory_baseline_regions": raise RuntimeError("LR selection identity/status invalid")
    return float(payload["selected_learning_rate"])
def parser():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--data-root",type=Path,default=None); p.add_argument("--result-root",type=Path,default=None); p.add_argument("--raw-flywire-root",type=Path,default=None); p.add_argument("--pn-kc-stats",type=Path,default=None); sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("prepare-data"); q.add_argument("--seed",type=int,choices=range(6))
    q=sub.add_parser("run-lr"); q.add_argument("--realization",type=int,choices=(0,1)); q.add_argument("--condition",choices=LR_CONDITIONS); q.add_argument("--learning-rate",type=float,choices=LR_CANDIDATES); q.add_argument("--device",default="cuda")
    sub.add_parser("finalize-lr"); q=sub.add_parser("run-formal"); q.add_argument("--seed",type=int,choices=FORMAL_SEEDS); q.add_argument("--condition",choices=CONDITIONS); q.add_argument("--device",default="cuda"); sub.add_parser("finalize-formal"); return p
def main():
    args=parser().parse_args(); base=Config(); cfg=Config(data_root=(args.data_root or base.data_root).resolve(),result_root=(args.result_root or base.result_root).resolve(),raw_flywire_root=(args.raw_flywire_root or base.raw_flywire_root).resolve(),pn_kc_stats_path=(args.pn_kc_stats or base.pn_kc_stats_path).resolve())
    if args.command=="prepare-data":
        for seed in range(6) if args.seed is None else (args.seed,): print(prepare_seed(seed,cfg),flush=True)
    elif args.command=="run-lr":
        for realization in (0,1) if args.realization is None else (args.realization,):
            for condition in LR_CONDITIONS if args.condition is None else (args.condition,):
                for lr in LR_CANDIDATES if args.learning_rate is None else (args.learning_rate,): execute(cfg,"lr",CALIBRATION_SEED,realization,condition,lr,args.device)
    elif args.command=="finalize-lr":
        runs=[json.loads(p.read_text()) for p in sorted((cfg.result_root/"lr/runs").glob("*.json"))]; payload=finalize_lr(runs); atomic_json(cfg.result_root/"lr/selection.json",payload); print(json.dumps(payload,indent=2))
    elif args.command=="run-formal":
        lr=selected_lr(cfg)
        for seed in FORMAL_SEEDS if args.seed is None else (args.seed,):
            for condition in CONDITIONS if args.condition is None else (args.condition,): execute(cfg,"formal",seed,0,condition,lr,args.device)
    else:
        runs=[json.loads(p.read_text()) for p in sorted((cfg.result_root/"formal/runs").glob("*.json"))]; payload=finalize_formal(runs); atomic_json(cfg.result_root/"formal/analysis.json",payload); print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
