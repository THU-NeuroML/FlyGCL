"""LR selection and deep formal-run audits."""
from __future__ import annotations
from collections import defaultdict
from typing import Any
import numpy as np
from .audit import canonical_sha256
from .config import CONDITIONS, FORMAL_SEEDS, LR_CANDIDATES, LR_CONDITIONS
from .experiment import continual_metrics

EXPERIMENT="olfactory_baseline_regions"
def _same(a:Any,b:Any)->bool: return bool(np.allclose(a,b,atol=1e-10,rtol=1e-10,equal_nan=False))
def validate_run(run:dict[str,Any])->list[str]:
    errors=[]; records=run.get("evaluation",{}).get("records",[]); fp=run.get("fingerprint",{}); config=fp.get("configuration",{}); protocol=config.get("protocol",{}); stream=run.get("stream",{}); n_regions=int(protocol.get("n_regions",5)); points=int(protocol.get("evaluation_points_per_region",5)); n_test=int(protocol.get("n_test",200_000))
    if run.get("schema_version")!=2 or run.get("experiment")!=EXPERIMENT or run.get("status")!="complete": errors.append("schema/experiment/status")
    if fp.get("algorithm")!="sha256-canonical-json" or fp.get("sha256")!=canonical_sha256(config): errors.append("configuration fingerprint")
    if config.get("identity")!=run.get("identity") or run.get("identity",{}).get("condition")!=run.get("condition"): errors.append("identity")
    order=stream.get("stage_order",[]); boundaries=stream.get("stage_boundaries",[]); lengths=stream.get("stage_lengths",[])
    if len(order)!=n_regions or sorted(order)!=list(range(n_regions)) or len(boundaries)!=n_regions or len(lengths)!=n_regions or list(np.diff([0,*boundaries]))!=lengths or sum(lengths)!=stream.get("length"): errors.append("stage order/boundaries/lengths")
    expected={s+round((e-s)*p/points) for s,e in zip((0,*boundaries[:-1]),boundaries,strict=True) for p in range(1,points+1)} if len(boundaries)==n_regions else set(); expected_positions=[0,*sorted(expected)]
    if run.get("evaluation",{}).get("expected_count")!=1+n_regions*points or len(records)!=len(expected_positions) or [r.get("position") for r in records]!=expected_positions: errors.append("evaluation positions/count")
    if records:
        hashes={(r.get("audit",{}).get("label_hash"),r.get("audit",{}).get("region_hash")) for r in records}
        if len(hashes)!=1 or any(None in pair for pair in hashes) or {r.get("audit",{}).get("sample_count") for r in records}!={n_test} or not all(r.get("audit",{}).get("read_only") for r in records): errors.append("evaluation hashes/count/read-only")
        test_counts=config.get("data",{}).get("test_region_counts")
        if not isinstance(test_counts,list) or len(test_counts)!=n_regions or sum(test_counts)!=n_test: errors.append("test region counts provenance")
        else:
            summary=continual_metrics(records,stream["length"],np.asarray(order),test_counts,points)
            for key,value in summary.items():
                if key not in run.get("summary",{}) or not _same(value,run["summary"][key]): errors.append(f"summary {key}")
    diagnostics=run.get("routing_diagnostics",{}); condition=run.get("condition")
    if len(diagnostics.get("training_counts",[]))!=n_regions or np.asarray(diagnostics.get("route_matrix",[])).shape!=(n_regions,n_regions): errors.append("routing diagnostic shapes")
    if condition in ("online_routed_5","oracle_routed_5") and diagnostics.get("prototype_counts")!=lengths: errors.append("prototype counts")
    for record in records:
        routing=record.get("routing",{}); counts=routing.get("route_counts"); matrix=np.asarray(routing.get("route_matrix",[]))
        if matrix.shape!=(n_regions,n_regions) or counts is not None and (len(counts)!=n_regions or list(matrix.sum(1))!=counts): errors.append("route matrix/counts"); break
        should_have=condition in ("random_routed_5","oracle_routed_5") or condition=="online_routed_5" and routing.get("active_prototypes",0)>0
        if should_have != (routing.get("route_accuracy") is not None): errors.append("route accuracy semantics"); break
    return errors

def finalize_lr(runs:list[dict[str,Any]])->dict[str,Any]:
    expected={(r,c,lr) for r in (0,1) for c in LR_CONDITIONS for lr in LR_CANDIDATES}; actual={(x["realization"],x["condition"],x["training"]["learning_rate"]) for x in runs}; errors=[]
    if len(runs)!=len(expected) or actual!=expected: errors.append("calibration identities")
    rows=[]
    for lr in LR_CANDIDATES:
        selected=[r for r in runs if r["training"]["learning_rate"]==lr]; errors.extend(e for r in selected for e in validate_run(r)); declines=[]
        for run in selected:
            n_regions=len(run["stream"]["stage_order"])
            for stage in range(n_regions):
                values=[x["mean_loss"] for x in run["loss_trajectory"] if x["stage"]==stage]; declines.append(len(values)>=2 and np.mean(values[-2:])<np.mean(values[:2]))
        rows.append({"learning_rate":lr,"numerically_stable":bool(all(bool(np.isfinite([x["mean_loss"] for x in r["loss_trajectory"]]).all()) for r in selected)),"majority_stage_loss_decline":bool(sum(declines)>len(declines)/2),"mean_seen_anytime_auc":float(np.mean([r["summary"]["seen_anytime_auc"] for r in selected]))})
    valid=[r for r in rows if r["numerically_stable"] and r["majority_stage_loss_decline"]]
    if errors or not valid: raise RuntimeError(errors or "no valid LR")
    best=max(valid,key=lambda x:x["mean_seen_anytime_auc"]); small=min(valid,key=lambda x:x["learning_rate"]); chosen=small if best["mean_seen_anytime_auc"]-small["mean_seen_anytime_auc"]<.005 else best
    return {"schema_version":2,"experiment":EXPERIMENT,"status":"frozen","selected_learning_rate":chosen["learning_rate"],"rule":"stable, majority within-region loss decline, maximize seen-region anytime AUC, smaller LR within 0.005","candidates":rows,"audit":{"valid":True,"expected_runs":12}}

def finalize_formal(runs:list[dict[str,Any]])->dict[str,Any]:
    expected={(s,c) for s in FORMAL_SEEDS for c in CONDITIONS}; actual={(r["seed"],r["condition"]) for r in runs}; errors=[]
    if len(runs)!=25 or actual!=expected: errors.append("formal identities")
    errors.extend(e for r in runs for e in validate_run(r)); paired=defaultdict(dict)
    for r in runs: paired[r["seed"]][r["condition"]]=r
    for seed,group in paired.items():
        if len(group)!=len(CONDITIONS): continue
        if len({r["initial_expert0_hash"] for r in group.values()})!=1: errors.append(f"seed {seed} expert0 pairing")
        multi=[tuple(group[c]["initial_expert_hashes"]) for c in CONDITIONS if c!="shared"]
        if not multi or len(set(multi))!=1 or any(len(x)!=len(multi[0]) for x in multi) or len(multi[0])!=len(group["random_routed_5"]["stream"]["stage_order"]): errors.append(f"seed {seed} all-head pairing")
    metrics=("seen_anytime_auc","all_regions_auc","final_accuracy","worst_region_accuracy","current_adaptation","old_retention","average_forgetting"); groups=defaultdict(list)
    for r in runs: groups[r["condition"]].append(r["summary"])
    rows={c:{m:{"mean":float(np.mean([x[m] for x in values])),"std":float(np.std([x[m] for x in values],ddof=1))} for m in metrics} for c,values in groups.items()}
    routing={}
    for condition in ("random_routed_5","online_routed_5","oracle_routed_5"):
        selected=[r for r in runs if r["condition"]==condition]
        final=np.asarray([r["evaluation"]["records"][-1]["routing"]["route_accuracy"] for r in selected],dtype=np.float64)
        report={"final_accuracy":{"mean":float(final.mean()),"std":float(final.std(ddof=1)),"per_seed":final.tolist()}}
        if condition=="online_routed_5":
            stage_end=np.asarray([[run["evaluation"]["records"][(stage+1)*5]["routing"]["route_accuracy"] for stage in range(5)] for run in selected],dtype=np.float64)
            report["stage_end_accuracy"]={"mean":stage_end.mean(0).tolist(),"std":stage_end.std(0,ddof=1).tolist(),"active_prototypes":[1,2,3,4,5]}
        routing[condition]=report
    return {"schema_version":2,"experiment":EXPERIMENT,"primary_metric":"seen_anytime_auc","audit":{"valid":not errors,"errors":errors,"expected_runs":25,"present_runs":len(runs)},"conditions":rows,"routing":routing}
