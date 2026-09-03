"""Deep audit and aggregate the complete hierarchical-model matrix."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from .audit import atomic_json, canonical_sha256
from .config import GAMMAS, METRICS, READOUTS, SEEDS, Config
from .metrics import continual_metrics
from .run import TASKS, TASK_ITEMS, Task, expected_identity, output_path


def close(a:float,b:float)->bool: return bool(np.isclose(a,b,rtol=1e-8,atol=1e-9))

def stats(values:list[float])->dict[str,Any]:
    array=np.asarray(values,float); return {"mean":float(array.mean()),"std":float(array.std(ddof=1)),"values":array.tolist()}


def validate(payload:dict[str,Any],index:int,task:Task,eta:float)->None:
    if payload.get("status")!="complete" or payload.get("experiment")!="olfactory_hierarchical_model" or payload.get("identity")!=expected_identity(index,task,eta): raise RuntimeError(f"identity mismatch task {index}")
    training=payload.get("training",{}); identity=payload["identity"]
    if payload.get("condition")!=task.condition or payload.get("seed")!=task.seed or training.get("eta")!=identity["eta"] or training.get("gamma")!=task.gamma or training.get("rates")!=identity["rates"] or training.get("optimizer")!="Adam" or not training.get("optimizer_created_once") or training.get("recorded_integrations")!=list(READOUTS): raise RuntimeError(f"training mismatch task {index}")
    if payload.get("classes_per_stage")!=[20]*5 or len(payload.get("class_groups",[]))!=100: raise RuntimeError(f"class grouping mismatch task {index}")
    evaluation=payload.get("evaluation",{}); records=evaluation.get("records",[]); stream=payload.get("stream",{})
    if evaluation.get("expected_count")!=26 or len(records)!=26 or stream.get("length")!=50_000 or stream.get("stage_lengths")!=[10_000]*5 or [record["position"] for record in records]!=list(range(0,50_001,2_000)): raise RuntimeError(f"stream/evaluation mismatch task {index}")
    if payload.get("fingerprint",{}).get("sha256")!=canonical_sha256(payload["fingerprint"]["configuration"]): raise RuntimeError(f"fingerprint mismatch task {index}")
    for record in records:
        if set(record.get("readouts",{}))!=set(READOUTS) or not record.get("audit",{}).get("read_only"): raise RuntimeError(f"readout/audit mismatch task {index}")
    summaries=payload.get("readout_summaries",{})
    for readout in READOUTS:
        selected=[{**record,"overall_accuracy":record["readouts"][readout]["overall_accuracy"],"region_accuracy":record["readouts"][readout]["region_accuracy"]} for record in records]; recomputed=continual_metrics(selected,50_000,stream["stage_order"],[2_000]*5,5)
        for metric in METRICS:
            if not close(float(recomputed[metric]),float(summaries[readout][metric])): raise RuntimeError(f"summary mismatch task {index} {readout} {metric}")
    routed=task.condition.startswith("inherited_"); inheritance=payload.get("inheritance",{}); events=inheritance.get("events",[])
    if routed:
        if not inheritance.get("enabled") or inheritance.get("source")!="arithmetic mean of all previously trained experts" or not inheritance.get("parameter_values_only") or inheritance.get("optimizer_state_copied") or len(events)!=4: raise RuntimeError(f"inheritance mismatch task {index}")
        for target,event in enumerate(events,start=1):
            if event.get("target_expert")!=target: raise RuntimeError(f"inheritance target mismatch task {index}")
            for scale in event.get("scales",[]):
                if scale.get("source_experts")!=list(range(target)) or scale.get("maximum_mean_reconstruction_error")!=0 or scale.get("optimizer_state_before") or scale.get("optimizer_state_after"): raise RuntimeError(f"inheritance event invalid task {index}")
        snapshots=payload.get("stage_parameter_snapshots",[])
        if len(snapshots)!=5: raise RuntimeError(f"snapshot mismatch task {index}")
        for stage,snapshot in enumerate(snapshots):
            expected=[expert<=stage for expert in range(5)]
            if snapshot.get("optimizer_state_experts")!=expected: raise RuntimeError(f"optimizer state lifecycle mismatch task {index} stage {stage}")
            if stage:
                for old in range(stage):
                    if snapshot["hashes"][old]!=snapshots[stage-1]["hashes"][old]: raise RuntimeError(f"old expert mutation task {index}")
    elif inheritance.get("enabled") or events: raise RuntimeError(f"unexpected inheritance task {index}")


def aggregate(root:Path,eta:float=1e-3)->dict[str,Any]:
    runs=[]
    for index,task in TASK_ITEMS:
        path=output_path(root,task)
        if not path.is_file(): raise FileNotFoundError(path)
        payload=json.loads(path.read_text(encoding="utf-8")); validate(payload,index,task,eta); runs.append(payload)
    grouped=defaultdict(list)
    for run in runs: grouped[(float(run["training"]["gamma"]),run["condition"])].append(run)
    results={}
    for gamma in GAMMAS:
        gamma_results={}
        for condition in ("shared_el","inherited_moe_el","inherited_moe_mid","inherited_moe_fast","inherited_moe_slow"):
            source=grouped.get((gamma,condition),[])
            if not source and condition=="inherited_moe_mid": source=grouped[(1.0,condition)]
            if not source: continue
            ordered=sorted(source,key=lambda item:item["seed"]); entry={}
            for readout in READOUTS:
                entry[readout]={metric:stats([float(run["readout_summaries"][readout][metric]) for run in ordered]) for metric in METRICS}; entry[readout]["stage_seen_accuracy"]={"mean":np.asarray([run["readout_summaries"][readout]["stage_seen_accuracy"] for run in ordered]).mean(0).tolist(),"std":np.asarray([run["readout_summaries"][readout]["stage_seen_accuracy"] for run in ordered]).std(0,ddof=1).tolist()}
            routes=[float(run["evaluation"]["records"][-1]["routing"]["route_accuracy"]) for run in ordered if run["evaluation"]["records"][-1]["routing"]["route_accuracy"] is not None]
            if routes: entry["final_routing_accuracy"]=stats(routes)
            gamma_results[condition]=entry
        results[f"{gamma:g}"]=gamma_results
    comparisons={}
    for gamma in GAMMAS:
        key=f"{gamma:g}"; current=results[key]; inherited=current["inherited_moe_el"]["softmax_mean"]["seen_anytime_auc"]; fixed=[current[name]["softmax_mean"]["seen_anytime_auc"] for name in ("inherited_moe_fast","inherited_moe_mid","inherited_moe_slow") if name in current]; per_seed_best=np.max(np.asarray([item["values"] for item in fixed]),axis=0)
        comparisons[key]={"inherited_moe_el_minus_mid_auc":inherited["mean"]-current["inherited_moe_mid"]["softmax_mean"]["seen_anytime_auc"]["mean"],"inherited_moe_el_minus_per_seed_best_fixed_auc":{"mean":float(np.mean(np.asarray(inherited["values"])-per_seed_best)),"values":(np.asarray(inherited["values"])-per_seed_best).tolist(),"positive_seeds":int(np.sum(np.asarray(inherited["values"])>per_seed_best))},"shared_el_auc":current["shared_el"]["softmax_mean"]["seen_anytime_auc"]["mean"],"inherited_moe_el_auc":inherited["mean"]}
    return {"schema_version":1,"experiment":"olfactory_hierarchical_model","status":"complete","audit":{"expected_runs":len(TASKS),"valid_runs":len(runs),"gammas":list(GAMMAS),"seeds":list(SEEDS),"readouts":list(READOUTS),"eta":eta,"no_selection_or_early_stop":True},"results":results,"comparisons":comparisons}


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--eta",type=float,default=1e-3); parser.add_argument("--result-root",type=Path,default=Config().result_root); args=parser.parse_args(); root=args.result_root.resolve(); payload=aggregate(root,args.eta); atomic_json(root/"analysis.json",payload); print(json.dumps(payload,indent=2))

if __name__=="__main__": main()
