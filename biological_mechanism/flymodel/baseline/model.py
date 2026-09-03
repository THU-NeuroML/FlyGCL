"""Independent 1300 ORN to 50 PN to 2000 KC encoder."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .config import Config

class FlyModelV4(nn.Module):
    def __init__(self, seed:int, n_experts:int, orn_pn:np.ndarray, pn_kc:np.ndarray, cfg:Config, device:str):
        super().__init__(); self.cfg=cfg
        if orn_pn.shape!=(cfg.n_pn,cfg.orn_per_channel) or orn_pn.dtype!=np.float32 or not np.allclose(orn_pn.sum(1),1,atol=1e-6): raise ValueError("invalid ORN-PN asset")
        if pn_kc.shape!=(cfg.n_pn,cfg.n_kc) or pn_kc.dtype!=np.float32 or np.any(pn_kc<0) or np.any(np.count_nonzero(pn_kc,axis=0)==0): raise ValueError("invalid PN-KC asset")
        self.register_buffer("orn_pn",torch.from_numpy(np.array(orn_pn,copy=True)))
        self.register_buffer("pn_kc",torch.from_numpy(np.array(pn_kc,copy=True)))
        generator=torch.Generator().manual_seed(seed); shared=torch.empty(cfg.n_classes,cfg.n_kc); nn.init.normal_(shared,std=cfg.n_kc**-.5,generator=generator)
        self.heads=nn.ModuleList([nn.Linear(cfg.n_kc,cfg.n_classes,bias=False) for _ in range(n_experts)])
        for head in self.heads:
            with torch.no_grad(): head.weight.copy_(shared)
        self.to(device)
    def encode(self,odors:torch.Tensor,sigma:float=0,generator:torch.Generator|None=None)->torch.Tensor:
        orn=odors.unsqueeze(-1).expand(-1,-1,self.cfg.orn_per_channel)
        if sigma: orn=orn+sigma*torch.randn(orn.shape,device=orn.device,dtype=orn.dtype,generator=generator)
        pn=F.relu(torch.sum(orn*self.orn_pn.unsqueeze(0),dim=-1)); kc=F.relu(pn@self.pn_kc); values,indices=kc.topk(self.cfg.kc_topk,dim=-1)
        return torch.zeros_like(kc).scatter(-1,indices,values)
    def logits(self,kc:torch.Tensor)->torch.Tensor: return torch.stack([head(kc) for head in self.heads],1)
