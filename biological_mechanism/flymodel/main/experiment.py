"""Inherited-expert temporal continual training with three simultaneous readouts."""
from __future__ import annotations
import random
from typing import Any
import numpy as np
import torch
from torch.nn import functional as F
from .audit import tensor_sha256
from .config import Config, READOUTS
from .metrics import continual_metrics
from .model import ExpertBank, FixedEncoder


def set_seed(seed:int)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def online_route(kc:torch.Tensor,sums:torch.Tensor,counts:torch.Tensor,active:int)->torch.Tensor:
    prototypes=(sums[:active]/counts[:active,None]).to(kc.dtype)
    return (F.normalize(kc,dim=1)@F.normalize(prototypes,dim=1).T).argmax(1)


def stream_order(labels:np.ndarray,order:np.ndarray,seed:int)->tuple[np.ndarray,list[int]]:
    rng=np.random.default_rng(seed); parts=[]; boundaries=[]; total=0
    for region in order:
        indices=np.flatnonzero(labels==region); rng.shuffle(indices); parts.append(indices); total+=len(indices); boundaries.append(total)
    result=np.concatenate(parts)
    if len(result)!=len(labels) or np.unique(result).size!=len(labels): raise RuntimeError("stream conservation failed")
    return result,boundaries


def integrated_logits(logits:torch.Tensor)->dict[str,torch.Tensor]:
    probabilities=logits.softmax(-1)
    return {"logits_mean":logits.mean(1),"softmax_mean":probabilities.mean(1).clamp_min(1e-12).log(),"softmax_max":probabilities.max(1).values.clamp_min(1e-12).log()}


def inherit_expert(bank:ExpertBank,target:int,optimizer:torch.optim.Optimizer)->dict[str,Any]:
    if not 0<target<bank.n_experts: raise ValueError("invalid inheritance target")
    scales=[]
    with torch.no_grad():
        for scale in range(bank.n_timescales):
            parameter=bank.head(target,scale).weight
            source=torch.stack([bank.head(expert,scale).weight for expert in range(target)]).mean(0)
            source_hashes=[tensor_sha256(bank.head(expert,scale).weight) for expert in range(target)]
            state_before=bool(optimizer.state.get(parameter))
            parameter.copy_(source)
            error=float((parameter-source).abs().max())
            state_after=bool(optimizer.state.get(parameter))
            scales.append({"timescale":scale,"source_experts":list(range(target)),"source_hashes":source_hashes,"target_hash":tensor_sha256(parameter),"maximum_mean_reconstruction_error":error,"optimizer_state_before":state_before,"optimizer_state_after":state_after})
    if any(item["maximum_mean_reconstruction_error"]!=0 or item["optimizer_state_before"] or item["optimizer_state_after"] for item in scales): raise RuntimeError("inheritance audit failed")
    return {"target_expert":target,"rule":"per-timescale arithmetic mean of all previously trained experts; parameter values only","scales":scales}


@torch.no_grad()
def evaluate(encoder:FixedEncoder,bank:ExpertBank,dataset:dict[str,Any],routed:bool,artifact:dict[str,torch.Tensor],device:torch.device,position:int,stage:int|None,active:int,cfg:Config)->dict[str,Any]:
    encoder.eval(); bank.eval(); predictions={name:[] for name in READOUTS}; routes=[]; before=(tensor_sha256(artifact["sums"]),tensor_sha256(artifact["counts"]))
    for start in range(0,len(dataset["test_samples"]),cfg.evaluation_batch_size):
        x=torch.from_numpy(np.array(dataset["test_samples"][start:start+cfg.evaluation_batch_size],dtype=np.float32,copy=True)).to(device); kc=encoder(x)
        if routed and active: route=online_route(kc,artifact["sums"],artifact["counts"],active)
        elif routed: route=torch.full((len(x),),-1,dtype=torch.long,device=device)
        else: route=torch.zeros(len(x),dtype=torch.long,device=device)
        if routed and active:
            selected={name:[] for name in READOUTS}
            for expert in range(active):
                mask=route==expert
                if mask.any():
                    values=integrated_logits(bank.expert_logits(kc[mask],expert))
                    for name in READOUTS: selected[name].append((mask,values[name]))
            for name in READOUTS:
                merged=torch.empty((len(x),cfg.n_classes),device=device)
                for mask,values in selected[name]: merged[mask]=values
                predictions[name].append(merged.argmax(1).cpu())
        elif routed:
            all_values=[integrated_logits(bank.expert_logits(kc,expert)) for expert in range(bank.n_experts)]
            for name in READOUTS: predictions[name].append(torch.stack([value[name] for value in all_values],1).mean(1).argmax(1).cpu())
        else:
            values=integrated_logits(bank.expert_logits(kc,0))
            for name in READOUTS: predictions[name].append(values[name].argmax(1).cpu())
        routes.append(route.cpu())
    if before!=(tensor_sha256(artifact["sums"]),tensor_sha256(artifact["counts"])): raise RuntimeError("evaluation mutated router")
    predictions={name:torch.cat(parts) for name,parts in predictions.items()}; route=torch.cat(routes); labels=torch.as_tensor(np.asarray(dataset["test_labels"],dtype=np.int64)); regions=torch.as_tensor(np.asarray(dataset["test_regions"],dtype=np.int64)); routed_mask=route>=0; route_accuracy=float((route[routed_mask]==regions[routed_mask]).float().mean()) if routed and active else None
    readouts={name:{"overall_accuracy":float((values==labels).float().mean()),"region_accuracy":[float((values[regions==region]==labels[regions==region]).float().mean()) for region in range(cfg.n_regions)]} for name,values in predictions.items()}
    return {"position":position,"stage":stage,"overall_accuracy":readouts["softmax_mean"]["overall_accuracy"],"region_accuracy":readouts["softmax_mean"]["region_accuracy"],"readouts":readouts,"routing":{"route_accuracy":route_accuracy,"route_accuracy_denominator":int(routed_mask.sum()) if route_accuracy is not None else 0,"active_prototypes":active if routed else None},"audit":{"sample_count":len(labels),"label_hash":tensor_sha256(labels),"region_hash":tensor_sha256(regions),"online_state_before":before,"read_only":True}}


def train(dataset:dict[str,Any],condition:str,eta:float,gamma:float,seed:int,cfg:Config,device_name:str="cuda")->dict[str,Any]:
    if condition not in ("shared_el","inherited_moe_el","inherited_moe_mid","inherited_moe_fast","inherited_moe_slow"): raise ValueError(condition)
    if gamma<1: raise ValueError("gamma must be at least one")
    set_seed(seed); device=torch.device(device_name); routed=condition.startswith("inherited_"); temporal=condition.endswith("_el"); n_experts=cfg.n_regions if routed else 1; n_timescales=3 if temporal else 1
    if temporal: rates=[gamma*eta,eta,eta/gamma]
    elif condition.endswith("_fast"): rates=[gamma*eta]
    elif condition.endswith("_slow"): rates=[eta/gamma]
    else: rates=[eta]
    encoder=FixedEncoder(dataset["orn_pn"],dataset["pn_kc"],cfg,device_name); bank=ExpertBank(seed,n_experts,n_timescales,cfg,device_name); groups=[]
    for expert in range(n_experts):
        for scale,rate in enumerate(rates): groups.append({"params":bank.head(expert,scale).parameters(),"lr":rate})
    optimizer=torch.optim.Adam(groups)
    initial_hashes=[[tensor_sha256(bank.head(expert,scale).weight) for scale in range(n_timescales)] for expert in range(n_experts)]
    order=np.asarray(dataset["stage_order"]); stream,boundaries=stream_order(dataset["train_regions"],order,seed); artifact={"sums":torch.zeros((cfg.n_regions,cfg.n_kc),dtype=torch.float64,device=device),"counts":torch.zeros(cfg.n_regions,dtype=torch.long,device=device)}; targets={start+round((end-start)*point/cfg.evaluation_points_per_region) for start,end in zip((0,*boundaries[:-1]),boundaries,strict=True) for point in range(1,cfg.evaluation_points_per_region+1)}
    records=[evaluate(encoder,bank,dataset,routed,artifact,device,0,None,0,cfg)]; position=0; noise=torch.Generator(device=device).manual_seed(seed+150_000); running=[]; losses=[]; inheritance=[]; stage_snapshots=[]; initialized_stage=-1
    while position<len(stream):
        event=min((value for value in (*targets,*boundaries) if value>position),default=len(stream)); end=min(position+cfg.batch_size,event); stage=next(index for index,boundary in enumerate(boundaries) if position<boundary)
        if stage!=initialized_stage:
            if routed and stage>0: inheritance.append(inherit_expert(bank,stage,optimizer))
            initialized_stage=stage
        indices=stream[position:end]; x=torch.from_numpy(np.array(dataset["train_samples"][indices],dtype=np.float32,copy=True)).to(device); y=torch.as_tensor(np.asarray(dataset["train_labels"][indices],dtype=np.int64),device=device); expert=stage if routed else 0
        encoder.train(); bank.train(); optimizer.zero_grad(set_to_none=True); kc=encoder(x,cfg.train_noise_sigma,noise); logits=bank.expert_logits(kc,expert); loss=torch.stack([F.cross_entropy(logits[:,scale],y) for scale in range(n_timescales)]).mean(); loss.backward()
        for other in range(n_experts):
            if other!=expert and any(bank.head(other,scale).weight.grad is not None for scale in range(n_timescales)): raise RuntimeError("inactive expert received gradient")
        optimizer.step(); running.append(float(loss.detach()))
        if routed: artifact["sums"][stage]+=kc.detach().double().sum(0); artifact["counts"][stage]+=len(kc)
        position=end
        if position in targets:
            current=next(index for index,boundary in enumerate(boundaries) if position<=boundary); records.append(evaluate(encoder,bank,dataset,routed,artifact,device,position,current,current+1 if routed else 0,cfg)); losses.append({"position":position,"stage":current,"mean_loss":float(np.mean(running))}); running=[]
        if position in boundaries:
            snapshot=[[tensor_sha256(bank.head(expert_index,scale).weight) for scale in range(n_timescales)] for expert_index in range(n_experts)]
            if routed and stage_snapshots:
                for old_expert in range(stage):
                    if snapshot[old_expert]!=stage_snapshots[-1]["hashes"][old_expert]: raise RuntimeError("previous expert changed after its stage")
            stage_snapshots.append({"stage":stage,"hashes":snapshot,"optimizer_state_experts":[all(bool(optimizer.state.get(bank.head(expert_index,scale).weight)) for scale in range(n_timescales)) for expert_index in range(n_experts)]})
    expected=1+cfg.n_regions*cfg.evaluation_points_per_region
    if len(records)!=expected or [record["position"] for record in records]!=[0,*sorted(targets)]: raise RuntimeError("evaluation schedule mismatch")
    summaries={}
    for name in READOUTS:
        readout_records=[{**record,"overall_accuracy":record["readouts"][name]["overall_accuracy"],"region_accuracy":record["readouts"][name]["region_accuracy"]} for record in records]
        summaries[name]=continual_metrics(readout_records,len(stream),order,dataset["metadata"]["counts"]["test"],cfg.evaluation_points_per_region)
    return {"schema_version":1,"experiment":"olfactory_hierarchical_model","status":"complete","condition":condition,"seed":seed,"training":{"eta":eta,"gamma":gamma,"rates":rates,"device":device_name,"optimizer":"Adam","optimizer_created_once":True,"primary_integration":"softmax_mean","recorded_integrations":list(READOUTS)},"inheritance":{"enabled":routed,"source":"arithmetic mean of all previously trained experts","parameter_values_only":True,"optimizer_state_copied":False,"events":inheritance},"stream":{"length":len(stream),"stage_order":order.tolist(),"stage_boundaries":boundaries,"stage_lengths":np.diff((0,*boundaries)).tolist()},"evaluation":{"expected_count":expected,"records":records},"loss_trajectory":losses,"summary":summaries["softmax_mean"],"readout_summaries":summaries,"initial_head_hashes":initial_hashes,"stage_parameter_snapshots":stage_snapshots,"routing_diagnostics":{"prototype_counts":artifact["counts"].tolist()},"audits":{"stream_conservation":True,"evaluation_read_only":True,"online_causal":True,"inactive_gradient_none":True,"old_experts_immutable":routed,"inheritance_mean_exact":routed}}
