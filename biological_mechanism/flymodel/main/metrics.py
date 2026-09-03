"""Seen-stage continual-learning metrics."""
from __future__ import annotations
import numpy as np


def continual_metrics(records, stream_length, order, test_counts, points_per_region: int):
    positions=np.asarray([record["position"] for record in records],float); acc=np.asarray([record["region_accuracy"] for record in records]); counts=np.asarray(test_counts,float); order=np.asarray(order); auc=[]; ends=[]; current=[]; old=[]
    for stage in range(len(order)):
        indices=np.arange(stage*points_per_region,(stage+1)*points_per_region+1); seen=order[:stage+1]; values=np.average(acc[indices][:,seen],axis=1,weights=counts[seen]); auc.append(float(np.trapezoid(values,positions[indices])/(positions[indices[-1]]-positions[indices[0]]))); ends.append(float(values[-1])); current.append(float(acc[indices[-1],order[stage]]))
        if stage: old.append(float(np.average(acc[indices[-1],order[:stage]],weights=counts[order[:stage]])))
    forgetting=[float(acc[(index+1)*points_per_region:,region].max()-acc[-1,region]) for index,region in enumerate(order[:-1])]; lengths=np.diff(np.r_[0,positions[points_per_region::points_per_region]]); overall=np.asarray([record["overall_accuracy"] for record in records])
    return {"seen_anytime_auc":float(np.average(auc,weights=lengths)),"stage_seen_auc":auc,"stage_seen_accuracy":ends,"all_regions_auc":float(np.trapezoid(overall,positions)/stream_length),"final_accuracy":float(np.average(acc[-1],weights=counts)),"worst_region_accuracy":float(acc[-1].min()),"current_adaptation":float(np.mean(current)),"old_retention":float(np.mean(old)),"average_forgetting":float(np.mean(forgetting)),"region_forgetting":forgetting}
