import gc
import logging
from typing import List

import torch

from methods._trainer import _Trainer
from models.hide_norga_prefix_vit import HiDePrefixModel, NoRGaPrefixModel
from utils.train_utils import select_optimizer, select_scheduler

logger = logging.getLogger()


class _BaseHiDeNoRGaTrainer(_Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_id = 0
        # will be initialized after dataset statistics are available
        self.class_to_first_task: List[int] = []
        # orthogonal loss & CA hyper-parameters
        self.lam_orth = getattr(self, "lam_orth", 1e-3)
        self.ca_num_per_class = getattr(self, "ca_num_per_class", 10)
        self.ca_steps = getattr(self, "ca_steps", 10)

    def setup_distributed_dataset(self):
        """Extend base dataset setup to also initialize class_to_first_task.

        _Trainer.setup_distributed_dataset defines self.n_classes, which we need
        to size class_to_first_task. We therefore initialize or resize the
        mapping here rather than in __init__.
        """
        super().setup_distributed_dataset()
        if not hasattr(self, "class_to_first_task") or len(self.class_to_first_task) != self.n_classes:
            self.class_to_first_task = [-1 for _ in range(self.n_classes)]

    def build_model(self, use_norga: bool = False):
        logger.info(f"Building {'NoRGa' if use_norga else 'HiDe'} Prefix model")
        if use_norga:
            model_cls = NoRGaPrefixModel
        else:
            model_cls = HiDePrefixModel

        extra_kwargs = {}
        if hasattr(self, "use_rp_gate"):
            extra_kwargs["use_rp_gate"] = self.use_rp_gate
        if hasattr(self, "rp_dim"):
            extra_kwargs["rp_dim"] = self.rp_dim
        if hasattr(self, "rp_ridge"):
            extra_kwargs["rp_ridge"] = self.rp_ridge
        if hasattr(self, "use_ema_head"):
            extra_kwargs["use_ema_head"] = self.use_ema_head
        if hasattr(self, "ema_ratio"):
            extra_kwargs["ema_ratio"] = self.ema_ratio

        self.model = model_cls(
            backbone_name=self.backbone,
            num_classes=self.n_classes,
            task_num=self.n_tasks,
            **extra_kwargs,
        ).to(self.device)
        self.model_without_ddp = self.model
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.lr_gamma = 0.99995 if "imagenet" in self.dataset else 0.9999
        self.scheduler = select_scheduler(self.sched_name, self.optimizer, self.lr_gamma)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.criterion = getattr(
            self.model_without_ddp,
            "loss_fn",
            torch.nn.CrossEntropyLoss(reduction="mean"),
        )


    def _get_old_class_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.n_classes, dtype=torch.bool, device=self.device)
        for c, t_first in enumerate(self.class_to_first_task):
            if t_first >= 0 and t_first < self.task_id:
                mask[c] = True
        return mask

    @torch.no_grad()
    def _collect_rp_features(self, images: torch.Tensor) -> None:
        """Collect features for RPFC gating (if enabled) using gate branch.

        We follow the FlyPrompt/SPrompt pattern: apply test_transform_tensor
        and let the model's RPFC head accumulate statistics with the current
        task id.
        """
        use_rp_gate = getattr(self.model_without_ddp, "use_rp_gate", False)
        if not use_rp_gate or not hasattr(self.model_without_ddp, "collect_rp"):
            return

        images = images.to(self.device, non_blocking=True)
        images = self.test_transform_tensor(images)

        task_labels = torch.full(
            (images.size(0),),
            self.task_id,
            device=self.device,
            dtype=torch.long,
        )

        self.model_without_ddp.eval()
        self.model_without_ddp.collect_rp(images, task_labels)

    def _update_class_first_task(self, labels: torch.Tensor):
        for y in labels.tolist():
            if self.class_to_first_task[y] == -1:
                self.class_to_first_task[y] = self.task_id

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        self._update_class_first_task(labels)
        _loss, _acc, _iter = 0.0, 0.0, 0
        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1

        # collect features for RPFC gating using the original batch
        with torch.no_grad():
            self._collect_rp_features(images.clone())

        del images, labels
        gc.collect()
        return _loss / _iter, _acc / _iter

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0
        x, y = data
        for j in range(len(y)):
            y[j] = self.exposed_classes.index(y[j].item())
        x = x.to(self.device)
        y = y.to(self.device)
        E = len(self.exposed_classes)

        logit_mask = torch.zeros_like(self.mask) - torch.inf
        cls_lst = torch.unique(y)
        for cc in cls_lst:
            logit_mask[cc] = 0

        x = self.train_transform(x)
        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit_p, feat_p = self.model_without_ddp.forward_prompt(
                x, task_id=self.task_id, labels=y, update_stats=True
            )
            logit_g, feat_g = self.model_without_ddp.forward_gate(
                x, detach_backbone=True, labels=y, update_stats=True
            )
            if not self.no_batchmask:
                logit_p = logit_p + logit_mask
            else:
                logit_p = logit_p + self.mask
            loss_p = self.criterion(logit_p, y)
            loss_g = self.criterion(logit_g[:, :E], y)
            old_class_mask = self._get_old_class_mask()
            ortho_p = self.model_without_ddp.compute_orth_loss(feat_p, old_class_mask, branch="prompt")
            ortho_g = self.model_without_ddp.compute_orth_loss(feat_g, old_class_mask, branch="gate")
            loss = loss_p + loss_g + self.lam_orth * (ortho_p + ortho_g)
        _, preds = logit_p.topk(self.topk, 1, True, True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        # EMA classifier head update for prompt branch (optional)
        if getattr(self.model_without_ddp, "use_ema_head", False) and hasattr(
            self.model_without_ddp, "update_ema_fc"
        ):
            self.model_without_ddp.update_ema_fc(expert_id=self.task_id)

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)
        return total_loss, total_correct / total_num_data

    def _predict_task_from_gate(self, x: torch.Tensor) -> torch.Tensor:
        """Predict task ids for a batch.

        If RPFC gating is enabled, we use the FlyPrompt-style RPFC head
        operating on the gate branch features. Otherwise, we fall back
        to the original HiDe/NoRGa class-to-task mapping based on the
        gate classifier's logits.
        """
        use_rp_gate = getattr(self.model_without_ddp, "use_rp_gate", False)
        if use_rp_gate and hasattr(self.model_without_ddp, "forward_with_rp"):
            with torch.no_grad():
                logit_task = self.model_without_ddp.forward_with_rp(x)
                # Only consider tasks that have been seen so far
                T_seen = self.task_id + 1
                logit_task = logit_task[:, :T_seen]
                task_hat = torch.argmax(logit_task, dim=-1)
            return task_hat

        # Fallback: original gate-branch class prediction + class→task mapping
        with torch.no_grad():
            logit_g, _ = self.model_without_ddp.forward_gate(x, detach_backbone=True)
            B, _ = logit_g.shape
            mask = torch.zeros_like(logit_g) - torch.inf
            E = len(self.exposed_classes)
            mask[:, :E] = 0.0
            logit_g = logit_g + mask
            pred_idx = torch.argmax(logit_g, dim=-1)
            task_hat = torch.empty_like(pred_idx)
            for i in range(B):
                idx = int(pred_idx[i].item())
                if idx >= E:
                    # should not happen due to masking, fall back to current task
                    task_hat[i] = self.task_id
                    continue
                orig_cls = self.exposed_classes[idx]
                t = self.class_to_first_task[orig_cls]
                if t < 0:
                    t = self.task_id
                task_hat[i] = t
        return task_hat

    def online_evaluate(self, test_loader, task_id=None, end=False):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        self.model.eval()

        # Finalize RPFC gating weights before evaluation
        use_rp_gate = getattr(self.model_without_ddp, "use_rp_gate", False)
        if use_rp_gate and hasattr(self.model_without_ddp, "update"):
            self.model_without_ddp.update()

        use_ema_head = getattr(self.model_without_ddp, "use_ema_head", False)

        with torch.no_grad():
            for x, y in test_loader:
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())
                x = x.to(self.device)
                y = y.to(self.device)
                task_hat = self._predict_task_from_gate(x)

                if use_ema_head and hasattr(self.model_without_ddp, "forward_prompt_with_ema"):
                    logit_ls = self.model_without_ddp.forward_prompt_with_ema(x, task_id=task_hat)
                    logit_ls = [logit + self.mask for logit in logit_ls]
                    logit_p = self._ensemble_logits(logit_ls)
                else:
                    logit_p, _ = self.model_without_ddp.forward_prompt(x, task_id=task_hat)
                    logit_p = logit_p + self.mask

                loss = self.criterion(logit_p, y)
                pred = torch.argmax(logit_p, dim=-1)
                _, preds = logit_p.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)
                total_loss += loss.item()
                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()
        avg_acc = total_correct / total_num_data if total_num_data > 0 else 0.0
        avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        return {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

    def _run_ca_for_branch(self, branch: str) -> None:
        feats, labels = self.model_without_ddp.sample_features_for_ca(
            branch, num_per_class=self.ca_num_per_class, device=self.device
        )
        if feats is None or labels is None:
            return
        self.model_without_ddp.eval()
        if branch == "prompt":
            head = self.model_without_ddp.prompt_head
        elif branch == "gate":
            head = self.model_without_ddp.g_head
        else:
            raise ValueError(f"Unknown branch {branch}")
        optimizer = torch.optim.SGD(head.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0)
        feats = feats.to(self.device)
        labels = labels.to(self.device)
        for _ in range(self.ca_steps):
            batch_size = min(self.batchsize, feats.size(0))
            idx = torch.randint(0, feats.size(0), (batch_size,), device=self.device)
            f_b = feats[idx]
            y_b = labels[idx]
            optimizer.zero_grad()
            logit = head(f_b)
            loss = self.criterion(logit, y_b)
            loss.backward()
            optimizer.step()

    def _ensemble_logits(self, logit_ls):
        """Ensemble a list of logits from online and EMA heads.

        Behavior mirrors FlyPrompt/SPrompt and is controlled by self.ensemble_method.
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

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, cur_iter):
        # run classifier alignment for both branches at task boundary
        self._run_ca_for_branch("prompt")
        self._run_ca_for_branch("gate")

        # Initialize EMA heads for the next task (if enabled)
        if getattr(self.model_without_ddp, "use_ema_head", False) and hasattr(self.model_without_ddp, "init_fc"):
            self.model_without_ddp.init_fc(expert_id=self.task_id + 1)

        self.task_id += 1


class HiDeGCLTrainer(_BaseHiDeNoRGaTrainer):
    pass


class NoRGaGCLTrainer(_BaseHiDeNoRGaTrainer):
    def online_after_task(self, cur_iter):
        if cur_iter == 0:
            self.model_without_ddp.freeze_act_scale()
        super().online_after_task(cur_iter)

