"""Run the complete 195-task hierarchical-model matrix."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import NamedTuple
from .audit import atomic_json, canonical_sha256, runtime_identity, source_identity, training_source_identity
from .config import ETA, GAMMAS, PROJECT_ROOT, SEEDS, Config
from .data import formal_subset
from .experiment import train


class Task(NamedTuple):
    gamma: float
    seed: int
    condition: str


LEGACY_EL_START = {1.0: 0, 2.0: 20, 3.0: 30, 4.0: 40, 6.0: 50, 8.0: 60}
LEGACY_FIXED_START = {2.0: 85, 3.0: 95, 4.0: 105, 6.0: 115, 8.0: 125}
ADDED_START = {5.0: 135, 7.0: 155, 9.0: 175, 10.0: 195}


def build_task_items()->tuple[tuple[int,Task],...]:
    items=[]
    for gamma,start in LEGACY_EL_START.items():
        for seed in SEEDS:
            items.extend(((start+2*seed,Task(gamma,seed,"shared_el")),(start+2*seed+1,Task(gamma,seed,"inherited_moe_el"))))
    for seed in SEEDS: items.append((70+seed,Task(1.0,seed,"inherited_moe_mid")))
    for gamma,start in LEGACY_FIXED_START.items():
        for seed in SEEDS:
            items.extend(((start+2*seed,Task(gamma,seed,"inherited_moe_fast")),(start+2*seed+1,Task(gamma,seed,"inherited_moe_slow"))))
    for gamma,start in ADDED_START.items():
        for seed in SEEDS:
            offset=start+4*seed
            items.extend(((offset,Task(gamma,seed,"shared_el")),(offset+1,Task(gamma,seed,"inherited_moe_el")),(offset+2,Task(gamma,seed,"inherited_moe_fast")),(offset+3,Task(gamma,seed,"inherited_moe_slow"))))
    items=sorted(items)
    if tuple(sorted((*LEGACY_EL_START,*ADDED_START)))!=GAMMAS or len(items)!=195 or len({index for index,_ in items})!=195 or len({task for _,task in items})!=195: raise RuntimeError("task matrix construction failed")
    return tuple(items)

TASK_ITEMS=build_task_items()
TASKS=tuple(task for _,task in TASK_ITEMS)
TASK_BY_INDEX=dict(TASK_ITEMS)


def gamma_key(gamma:float)->str: return f"gamma{gamma:g}"

def output_path(root:Path,task:Task)->Path: return root/"runs"/gamma_key(task.gamma)/f"seed{task.seed}_{task.condition}.json"


def expected_identity(index:int,task:Task,eta:float=ETA)->dict:
    rates=[task.gamma*eta,eta,eta/task.gamma] if task.condition.endswith("_el") else [task.gamma*eta] if task.condition.endswith("_fast") else [eta/task.gamma] if task.condition.endswith("_slow") else [eta]
    return {"task_index":index,"seed":task.seed,"gamma":task.gamma,"eta":eta,"rates":rates,"condition":task.condition,"inheritance":"mean_all_previous_experts" if task.condition.startswith("inherited_") else "none","train_per_stage":10_000,"test_per_stage":2_000,"class_exclusive":True,"classes_per_stage":20}


def validate_existing(path:Path,index:int,task:Task,eta:float=ETA)->bool:
    if not path.is_file(): return False
    payload=json.loads(path.read_text(encoding="utf-8")); fingerprint=payload.get("fingerprint",{}); configuration=fingerprint.get("configuration",{})
    recorded_source=configuration.get("source",{}).get("files",{})
    current_training_source=training_source_identity(PROJECT_ROOT)["files"]
    training_source_matches=all(recorded_source.get(name)==digest for name,digest in current_training_source.items())
    return payload.get("status")=="complete" and payload.get("identity")==expected_identity(index,task,eta) and payload.get("evaluation",{}).get("expected_count")==26 and len(payload.get("evaluation",{}).get("records",[]))==26 and fingerprint.get("sha256")==canonical_sha256(configuration) and training_source_matches


def execute(index:int,device:str,root:Path,eta:float=ETA)->Path:
    task=TASK_BY_INDEX[index]; output=output_path(root,task)
    if validate_existing(output,index,task,eta): print(f"SKIP {index:03d} {output}",flush=True); return output
    if output.exists(): output.unlink()
    cfg=Config(result_root=root); dataset=formal_subset(task.seed,cfg); identity=expected_identity(index,task,eta); configuration={"identity":identity,"protocol":cfg.to_dict(),"class_groups":dataset["class_groups"].tolist(),"data_fingerprint":dataset["metadata"],"source":source_identity(PROJECT_ROOT),"training_source":training_source_identity(PROJECT_ROOT),"runtime":runtime_identity(device)}
    started=time.time(); payload=train(dataset,task.condition,eta,task.gamma,task.seed,cfg,device); payload.update({"identity":identity,"class_groups":dataset["class_groups"].tolist(),"classes_per_stage":[20]*5,"fingerprint":{"algorithm":"sha256-canonical-json","sha256":canonical_sha256(configuration),"configuration":configuration},"runtime":{**configuration["runtime"],"seconds":time.time()-started}}); atomic_json(output,payload); print(f"DONE {index:03d} {output}",flush=True); return output


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--task-index",type=int,choices=tuple(TASK_BY_INDEX)); parser.add_argument("--worker",type=int); parser.add_argument("--workers",type=int); parser.add_argument("--device",default="cuda"); parser.add_argument("--eta",type=float,default=ETA); parser.add_argument("--result-root",type=Path,default=Config().result_root); args=parser.parse_args()
    if args.eta<=0: parser.error("eta must be positive")
    if args.task_index is not None: indices=(args.task_index,)
    else:
        if args.worker is None or args.workers is None or not 0<=args.worker<args.workers: parser.error("provide --task-index or valid worker partition")
        indices=tuple(TASK_BY_INDEX)[args.worker::args.workers]
    for index in indices: execute(index,args.device,args.result_root.resolve(),args.eta)

if __name__=="__main__": main()
