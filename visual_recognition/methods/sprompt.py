import gc
import logging
from typing import Dict, List

import numpy as np
import torch
from sklearn.cluster import KMeans

from methods._trainer import _Trainer

logger = logging.getLogger()


class SPrompt(_Trainer):
    def __init__(self, *args, **kwargs):
        super(SPrompt, self).__init__(*args, **kwargs)

        self.task_id = 0
        self.label_to_task: Dict[int, int] = {}

        # per-task prototypes: task_id -> torch.Tensor[K, D] (CPU)
        self.task_prototypes: Dict[int, torch.Tensor] = {}
        # current task features buffer (list of CPU tensors)
        self._cur_task_features: List[torch.Tensor] = []

        self.num_prototypes_per_task = 5
        self.max_features_per_task = 10000

    # ---------------------------- feature utils ----------------------------
    @torch.no_grad()
    def _extract_cls_feature(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLS features from pretrained backbone without prompts.

        Uses the same test_transform_tensor as FlyPrompt and only the backbone
        (no prompt routing). Features are moved to CPU for storage.
        """
        self.model_without_ddp.eval()
        # apply test transform (same as FlyPrompt collect)
        images = images.to(self.device, non_blocking=True)
        images = self.test_transform_tensor(images)

        # backbone forward (ViT): use forward_features and take CLS token
        if hasattr(self.model_without_ddp.backbone, "forward_features"):
            feats = self.model_without_ddp.backbone.forward_features(images)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            cls_feat = feats[:, 0]
        else:
            raise RuntimeError("Backbone must support forward_features for CLS extraction")

        return cls_feat.detach().cpu()

    def _append_cur_task_features(self, feats: torch.Tensor):
        """Append features to current task buffer with reservoir-like cap."""
        self._cur_task_features.append(feats)
        all_feats = torch.cat(self._cur_task_features, dim=0)
        if all_feats.size(0) > self.max_features_per_task:
            # randomly subsample to max_features_per_task on CPU
            idx = torch.randperm(all_feats.size(0))[: self.max_features_per_task]
            all_feats = all_feats[idx]
        self._cur_task_features = [all_feats]

    @torch.no_grad()
    def _collect_rp_features(self, images: torch.Tensor, labels: torch.Tensor):
        """Collect features for RPFC gating (if enabled).

        Follows the same pattern as FlyPrompt: apply test_transform_tensor,
        run backbone.forward_features, and let the model's RPFC head accumulate
        statistics with current task id.
        """
        use_rp_gate = getattr(self.model_without_ddp, "use_rp_gate", False)
        if not use_rp_gate or not hasattr(self.model_without_ddp, "collect"):
            return

        # Map labels to seen-class indices (consistent with training code)
        for j in range(len(labels)):
            labels[j] = self.exposed_classes.index(labels[j].item())

        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device)

        images = self.test_transform_tensor(images)

        self.model_without_ddp.eval()
        self.model_without_ddp.collect(images, labels)

    @torch.no_grad()
    def _build_prototypes_for_task(self, task_id: int):
        """Run KMeans on current task features to build prototypes."""
        if len(self._cur_task_features) == 0:
            return
        feats = torch.cat(self._cur_task_features, dim=0)
        # L2-normalize features for more stable clustering/routing
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-6)
        if feats.size(0) < self.num_prototypes_per_task:
            # too few points, just use unique features as prototypes
            self.task_prototypes[task_id] = feats.clone()
            return
        np_feats = feats.numpy()
        k = min(self.num_prototypes_per_task, np_feats.shape[0])
        kmeans = KMeans(n_clusters=k, random_state=0).fit(np_feats)
        centers = torch.from_numpy(kmeans.cluster_centers_).float()
        # also L2-normalize centers
        centers = centers / (centers.norm(dim=1, keepdim=True) + 1e-6)
        self.task_prototypes[task_id] = centers

    @torch.no_grad()
    def _route_batch_by_prototypes(self, images: torch.Tensor) -> torch.Tensor:
        """Given raw images, return per-sample task ids via nearest prototype."""
        assert len(self.task_prototypes) > 0, "No task prototypes available for routing"

        cls_feats = self._extract_cls_feature(images)  # [B, D] on CPU
        # L2-normalize query features for distance-based routing
        cls_feats = cls_feats / (cls_feats.norm(dim=1, keepdim=True) + 1e-6)
        B, D = cls_feats.shape
        cls_feats = cls_feats.unsqueeze(1)  # [B, 1, D]

        dists_per_task = []
        task_ids = sorted(self.task_prototypes.keys())
        for t in task_ids:
            centers = self.task_prototypes[t]  # [K, D] on CPU
            centers = centers.to(cls_feats.device)
            diff = cls_feats - centers.unsqueeze(0)  # [B, K, D]
            dist = diff.pow(2).sum(-1).sqrt().min(1)[0]  # [B]
            dists_per_task.append(dist.unsqueeze(1))

        dists = torch.cat(dists_per_task, dim=1)  # [B, T]
        min_idx = torch.argmin(dists, dim=1)  # [B]
        routed_tasks = torch.tensor([task_ids[i.item()] for i in min_idx], dtype=torch.long, device=self.device)
        return routed_tasks

    # ------------------------------ training -------------------------------
    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        _loss, _acc, _iter = 0.0, 0.0, 0

        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        # collect CLS features for current task (CPU, capped) and
        # optionally collect features for RPFC gating
        with torch.no_grad():
            feats = self._extract_cls_feature(images.clone())
            self._append_cur_task_features(feats)
            self._collect_rp_features(images.clone(), labels.clone())

        del images, labels
        gc.collect()
        return _loss / _iter, _acc / _iter

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        x, y = data

        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())

        logit_mask = torch.zeros_like(self.mask) - torch.inf
        cls_lst = torch.unique(y)
        for cc in cls_lst:
            logit_mask[cc] = 0

        x = x.to(self.device)
        y = y.to(self.device)

        x = self.train_transform(x)

        self.optimizer.zero_grad()
        if not self.no_batchmask:
            logit, loss = self.model_forward(x, y, mask=logit_mask)
        else:
            logit, loss = self.model_forward(x, y)

        _, preds = logit.topk(self.topk, 1, True, True)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        # EMA classifier head update (optional)
        if getattr(self.model_without_ddp, "use_ema_head", False):
            self.model_without_ddp.update_ema_fc(expert_id=self.task_id)

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct / total_num_data

    def model_forward(self, x, y, mask=None):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit = self.model(x)
            if mask is not None:
                logit += mask
            else:
                logit += self.mask

            loss = self.criterion(logit, y)

        return logit, loss


    # --------------------------- task boundaries ---------------------------
    def online_before_task(self, task_id):
        self.task_id = task_id
        self._cur_task_features = []

    def online_after_task(self, task_id):
        # advance model task counter and clear feature buffer for this task
        self.model_without_ddp.process_task_count()
        self._cur_task_features = []
        gc.collect()

    # ----------------------------- evaluation ------------------------------
    @torch.no_grad()
    def online_evaluate(self, test_loader, task_id=None, end=False):
        logger.info("Start evaluation...")

        use_rp_gate = getattr(self.model_without_ddp, "use_rp_gate", False)
        use_ema_head = getattr(self.model_without_ddp, "use_ema_head", False)

        # If RPFC gating is enabled, update RPFC weights before evaluation.
        if use_rp_gate and hasattr(self.model_without_ddp, "update"):
            self.model_without_ddp.update()

        # If we are currently training a task and not using RPFC gating,
        # rebuild its prototypes from the up-to-date feature buffer.
        if (not use_rp_gate) and len(self._cur_task_features) > 0:
            self._build_prototypes_for_task(self.task_id)

        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)

        self.model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                # map ground-truth labels to indices in exposed_classes
                for j in range(len(labels)):
                    labels[j] = self.exposed_classes.index(labels[j].item())

                images = images.to(self.device)
                labels = labels.to(self.device)

                if use_rp_gate:
                    # Use RPFC head to predict task ids from CLS features
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        logit_task = self.model_without_ddp.forward_with_rp(images)
                    # Only consider tasks that have been seen so far
                    E = self.task_id + 1
                    logit_task = logit_task[:, :E]
                    expert_ids = torch.argmax(logit_task, dim=-1)
                else:
                    # route each sample to a task via prototypes, then call model with prompts
                    expert_ids = self._route_batch_by_prototypes(images)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    if use_ema_head:
                        # Use EMA head bank (online + EMA heads) and ensemble
                        logit_ls = self.model_without_ddp.forward_with_ema(images, expert_ids=expert_ids)
                        logit_ls = [logit + self.mask for logit in logit_ls]
                        logit = self._ensemble_logits(logit_ls)
                    else:
                        logit = self.model(images, expert_ids=expert_ids)
                        logit = logit + self.mask

                    loss = self.criterion(logit, labels)

                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == labels.unsqueeze(1)).item()
                total_num_data += labels.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(labels, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()
                total_loss += loss.item()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()

        eval_dict = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}
        return eval_dict

    def _ensemble_logits(self, logit_ls):
        """Ensemble a list of logits from online and EMA heads.

        The behavior mirrors FlyPrompt's implementation and is controlled by
        self.ensemble_method (default: softmax_max_prob).
        """
        if not hasattr(self, "ensemble_method"):
            self.ensemble_method = "softmax_max_prob"

        if "softmax" in self.ensemble_method:
            logit_ls = [torch.softmax(logit, dim=-1) for logit in logit_ls]

        logit_stack = torch.stack(logit_ls, dim=-1)  # [batch_size, n_classes, n_heads]

        if "mean" in self.ensemble_method:
            return logit_stack.mean(dim=-1)
        elif "max_prob" in self.ensemble_method:
            return logit_stack.max(dim=-1)[0]
        elif "min_entropy" in self.ensemble_method:
            entropies = -torch.sum(logit_stack * torch.log(logit_stack + 1e-8), dim=1)
            min_entropy_indices = torch.argmin(entropies, dim=-1)
            batch_indices = torch.arange(logit_stack.size(0), device=logit_stack.device)
            return logit_stack[batch_indices, :, min_entropy_indices]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

