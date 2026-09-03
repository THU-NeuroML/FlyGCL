"""FlyGapLoss: FlyMethod + MindTheGap-style modality gap stability loss.

主体架构（prompt/adapter/lora 路径分配、RP 路由、EMA ensemble）完全继承自 FlyMethod，
不做任何改动。在此基础上叠加一个可选的 gap stability auxiliary loss：

  L_gap = (mu_neg_cur - mu_neg_ref)^2

其中 mu_neg_ref 是每个 session 训练开始前在训练数据上统计的负类余弦 logit 均值（冻结），
mu_neg_cur 是当前 batch 的负类余弦 logit 均值。梯度只流过 image features，
不流过 text features（text encoder 始终冻结）。

Config 参数（默认值均向后兼容，gap_loss_weight=0 时退化为标准 FlyMethod）：
  gap_loss_weight  (float, default 0.1)  — gap loss 权重 λ
  gap_ref_batches  (int,   default 5)    — 参考阶段使用的 batch 数
"""

from typing import Optional

import clip
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from .fly_method import FlyMethod


class FlyGapLossMethod(FlyMethod):
    """FlyMethod + MindTheGap modality gap stability loss（非侵入式叠加）。"""

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)

        # --- Gap loss 超参数 ---
        self.gap_loss_weight = float(getattr(cfg, "gap_loss_weight", 0.1))
        self.gap_ref_batches = int(getattr(cfg, "gap_ref_batches", 5))
        self._prompt_template = str(getattr(cfg, "prompt_template", "a photo of a {}."))

        # --- 运行时状态（每个 session 更新） ---
        self._gap_text_features: Optional[torch.Tensor] = None  # [K, D] 归一化 text features
        self._neg_ref_mean: Optional[float] = None              # 参考负类 logit 均值（标量）
        self._pending_gap_loss: Optional[torch.Tensor] = None   # 当前 batch 的 gap loss
        self._last_train_cls: Optional[torch.Tensor] = None     # expert 路径输出的 CLS 特征缓存

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _refresh_gap_text_features(self) -> None:
        """编码当前所有 seen class 的 text features（冻结），供 gap loss 使用。"""
        if self.gap_loss_weight <= 0.0 or not self.current_class_names:
            self._gap_text_features = None
            return
        tokens = clip.tokenize(
            [self._prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        feats = self.clip_model.encode_text(tokens).float()
        self._gap_text_features = F.normalize(feats, dim=-1).detach()

    def _project_visual(self, cls: torch.Tensor) -> torch.Tensor:
        """将 ViT CLS token 投影到 CLIP embedding 空间（如有 visual.proj）。"""
        if (
            hasattr(self.clip_model, "visual")
            and hasattr(self.clip_model.visual, "proj")
            and self.clip_model.visual.proj is not None
        ):
            return cls.float() @ self.clip_model.visual.proj.float()
        return cls.float()

    def _compute_neg_mean(self, sim: torch.Tensor) -> torch.Tensor:
        """给定 similarity 矩阵 [B, K]，用 pseudo-label 计算负类 logit 均值（标量 tensor）。"""
        K = sim.size(1)
        if K <= 1:
            return sim.mean() * 0.0  # 不足两类，返回 0
        pseudo = sim.detach().argmax(dim=-1)               # [B]
        pos_logits = sim.gather(1, pseudo.unsqueeze(1)).squeeze(1)  # [B]
        # neg_mean_per_sample = (sum_all - pos) / (K-1)
        neg_mean_per_sample = (sim.sum(dim=1) - pos_logits.detach()) / (K - 1)
        return neg_mean_per_sample.mean()

    # ------------------------------------------------------------------
    # 公开接口：参考阶段统计
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_neg_ref(self, data_loader, device, max_batches: int = 5) -> None:
        """在训练集前 max_batches 个 batch 上计算负类余弦 logit 参考均值。

        main_gcl.py 在每个 session 的 train_loader 构建完后、正式训练前调用。
        gap_loss_weight <= 0 时直接返回（no-op）。
        """
        if self.gap_loss_weight <= 0.0 or self._gap_text_features is None:
            return

        active_expert = max(0, min(self.current_task, self.task_num - 1))
        neg_means = []
        was_training = self.clip_model.training
        self.eval()

        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            images = batch[0].to(device)

            # 提取 expert 路径的 CLS 特征（与训练时一致）
            q = self._extract_query_features(images)
            expert_ids = torch.full(
                (images.size(0),), active_expert, device=device, dtype=torch.long
            )
            cls = self._extract_expert_features(images, q, expert_ids, train=False)
            img_f = F.normalize(self._project_visual(cls), dim=-1)

            scale = self.clip_model.logit_scale.exp().clamp(max=100.0)
            sim = scale * img_f @ self._gap_text_features.T   # [B, K]
            neg_means.append(self._compute_neg_mean(sim).item())

        if was_training:
            self.train()
        if neg_means:
            self._neg_ref_mean = float(sum(neg_means) / len(neg_means))

    # ------------------------------------------------------------------
    # 覆写 _forward_with_expert：训练时缓存 cls 特征
    # ------------------------------------------------------------------

    def _forward_with_expert(
        self,
        image: torch.Tensor,
        q_features: torch.Tensor,
        expert_ids: torch.Tensor,
        train: bool,
    ) -> torch.Tensor:
        cls = self._extract_expert_features(image, q_features, expert_ids, train=bool(train))
        if train:
            self._last_train_cls = cls   # 供 gap loss 使用
        return self._compute_clip_logits(cls)

    # ------------------------------------------------------------------
    # 覆写 adaptation：更新 text features 并重置参考值
    # ------------------------------------------------------------------

    def adaptation(self, task_id: int, reset: bool = False) -> None:
        super().adaptation(task_id, reset=reset)
        # 重置当前 session 的 gap 状态
        self._neg_ref_mean = None
        self._pending_gap_loss = None
        self._last_train_cls = None
        # 更新 text features（基于最新 seen class names）
        self._refresh_gap_text_features()

    # ------------------------------------------------------------------
    # 覆写 forward：训练分支中附加 gap loss 计算
    # ------------------------------------------------------------------

    def forward(self, image: torch.Tensor, test: bool = False, all_test: bool = False) -> torch.Tensor:
        # test 分支：直接走父类，无任何 gap loss 逻辑
        if test:
            return super().forward(image, test=True, all_test=all_test)

        # train 分支：先调父类（会触发覆写后的 _forward_with_expert，缓存 _last_train_cls）
        logits = super().forward(image, test=False, all_test=False)

        # 计算 gap stability loss
        self._pending_gap_loss = None
        if (
            self.gap_loss_weight > 0.0
            and self._neg_ref_mean is not None
            and self._gap_text_features is not None
            and self._last_train_cls is not None
        ):
            img_f = F.normalize(self._project_visual(self._last_train_cls), dim=-1)
            # logit_scale 不传梯度（与 MindTheGap 中 logit 计算一致）
            scale = self.clip_model.logit_scale.exp().clamp(max=100.0).detach()
            sim = scale * img_f @ self._gap_text_features.T   # [B, K]

            neg_mean_cur = self._compute_neg_mean(sim)
            ref = torch.tensor(
                self._neg_ref_mean, device=neg_mean_cur.device, dtype=neg_mean_cur.dtype
            )
            self._pending_gap_loss = (neg_mean_cur - ref).pow(2)

        return logits

    # ------------------------------------------------------------------
    # 覆写 auxiliary_loss / auxiliary_info
    # ------------------------------------------------------------------

    def auxiliary_loss(self):
        if self._pending_gap_loss is not None and self.gap_loss_weight > 0.0:
            return self.gap_loss_weight * self._pending_gap_loss
        return None

    def auxiliary_info(self):
        info = super().auxiliary_info()
        info["method"] = "fly_gaploss"
        info["gap_loss_weight"] = float(self.gap_loss_weight)
        info["neg_ref_mean"] = (
            round(float(self._neg_ref_mean), 6) if self._neg_ref_mean is not None else None
        )
        info["gap_loss"] = (
            round(float(self._pending_gap_loss.detach().item()), 6)
            if self._pending_gap_loss is not None
            else None
        )
        return info
