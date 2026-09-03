import torch
import torch.nn as nn
import torch.nn.functional as F


def _tensor_prompt(*shape):
    p = nn.Parameter(torch.empty(*shape), requires_grad=True)
    nn.init.uniform_(p)
    return p


def _task_window_indices(task_id: int, top_k: int, pool_size: int, device: torch.device):
    start = int(task_id) * int(top_k)
    end = (int(task_id) + 1) * int(top_k)
    if end > int(pool_size):
        return None
    return torch.arange(start, end, device=device, dtype=torch.long)


def _visible_prompt_indices(task_id: int, top_k: int, pool_size: int, device: torch.device, mode: str):
    mode = str(mode).lower()
    if task_id is None or int(task_id) < 0 or mode == "global":
        return None
    if mode == "hard_session":
        return _task_window_indices(task_id, top_k, pool_size, device)
    if mode == "cumulative":
        end = min((int(task_id) + 1) * int(top_k), int(pool_size))
        if end <= 0:
            return None
        return torch.arange(0, end, device=device, dtype=torch.long)
    return None


def _batchwise_major_prompt(idx: torch.Tensor, pool_size: int, top_k: int):
    # Match l2p-main batchwise_prompt behavior: use the most frequent prompt ids in a batch.
    counts = torch.bincount(idx.reshape(-1), minlength=int(pool_size))
    major_ids = torch.topk(counts, k=int(top_k), largest=True, sorted=True).indices
    return major_ids.unsqueeze(0).expand(idx.shape[0], -1)


class OfficialL2PPrompt(nn.Module):
    """PyTorch prompt-pool semantics aligned with l2p-main behavior."""

    def __init__(
        self,
        emb_d: int,
        n_tasks: int,
        pool_size: int = 10,
        prompt_length: int = 10,
        top_k: int = 4,
        batchwise_prompt: bool = True,
        deep_prompt: bool = False,
        use_prompt_mask: bool = False,
        e_prompt_layer_idx=None,
        key_dim: int = None,
        prompt_window_mode: str = "hard_session",
        prompt_eval_mode: str = "same_as_train",
    ):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = emb_d if key_dim is None else int(key_dim)
        self.n_tasks = n_tasks
        self.pool_size = int(pool_size)
        self.prompt_length = int(prompt_length)
        self.top_k = int(top_k)
        self.batchwise_prompt = bool(batchwise_prompt)
        self.deep_prompt = bool(deep_prompt)
        self.use_prompt_mask = bool(use_prompt_mask)
        self.prompt_window_mode = str(prompt_window_mode).lower()
        self.prompt_eval_mode = str(prompt_eval_mode).lower()
        self.e_layers = list(e_prompt_layer_idx) if e_prompt_layer_idx is not None else ([0, 1, 2, 3, 4] if self.deep_prompt else [0])

        self._reduce_sim = None
        self._last_idx = None
        self._last_visible = None
        self._last_mode = "global"
        self._train_usage = torch.zeros(self.pool_size, dtype=torch.long)
        self._eval_usage = torch.zeros(self.pool_size, dtype=torch.long)

        for e in self.e_layers:
            setattr(self, f"e_p_{e}", _tensor_prompt(self.pool_size, self.prompt_length, emb_d))
            setattr(self, f"e_k_{e}", _tensor_prompt(self.pool_size, self.key_d))

    def process_task_count(self):
        self.task_count += 1

    def reset_stats(self):
        self._reduce_sim = None

    def set_routing_modes(self, train_mode: str, eval_mode: str):
        self.prompt_window_mode = str(train_mode).lower()
        self.prompt_eval_mode = str(eval_mode).lower()

    def _active_mode(self, train: bool):
        if train:
            return self.prompt_window_mode
        if self.prompt_eval_mode == "same_as_train":
            return self.prompt_window_mode
        return self.prompt_eval_mode

    def _record_selection(self, idx: torch.Tensor, train: bool):
        idx_cpu = idx.detach().reshape(-1).cpu()
        binc = torch.bincount(idx_cpu, minlength=self.pool_size)
        if train:
            self._train_usage += binc
        else:
            self._eval_usage += binc

    def get_last_selected_indices(self):
        return self._last_idx

    def get_last_routing_info(self):
        return {
            "mode": self._last_mode,
            "visible_prompt_ids": list(self._last_visible) if self._last_visible is not None else None,
        }

    def get_usage_snapshot(self):
        train_nonzero = int((self._train_usage > 0).sum().item())
        eval_nonzero = int((self._eval_usage > 0).sum().item())
        return {
            "train_prompt_usage": self._train_usage.tolist(),
            "eval_prompt_usage": self._eval_usage.tolist(),
            "train_prompt_saturation": float(train_nonzero / max(self.pool_size, 1)),
            "eval_prompt_saturation": float(eval_nonzero / max(self.pool_size, 1)),
        }

    def get_last_reduce_sim(self):
        return self._reduce_sim

    def _select_prompt_indices(self, cos_sim: torch.Tensor, task_id: int, train: bool):
        bsz = cos_sim.shape[0]
        mode = self._active_mode(train)
        visible = None
        if self.use_prompt_mask:
            visible = _visible_prompt_indices(task_id, self.top_k, self.pool_size, cos_sim.device, mode)

        if visible is not None:
            masked = cos_sim.new_full(cos_sim.shape, float("-inf"))
            masked[:, visible] = cos_sim[:, visible]
            idx = torch.topk(masked, self.top_k, dim=1).indices
            self._last_visible = [int(v.item()) for v in visible.detach().cpu()]
        else:
            idx = torch.topk(cos_sim, self.top_k, dim=1).indices
            self._last_visible = None

        if self.batchwise_prompt:
            idx = _batchwise_major_prompt(idx, self.pool_size, self.top_k)
        self._last_mode = mode
        self._last_idx = idx.detach().cpu()
        self._record_selection(idx, train)
        return idx

    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        if l not in self.e_layers:
            return None, x_querry.new_zeros(1), x_block

        B, _ = x_querry.shape
        K = getattr(self, f"e_k_{l}")
        P = getattr(self, f"e_p_{l}")

        n_K = F.normalize(K, dim=1)
        q = F.normalize(x_querry, dim=1)
        cos_sim = torch.einsum("bd,kd->bk", q, n_K)

        idx = self._select_prompt_indices(cos_sim, int(task_id) if task_id is not None else -1, bool(train))
        P_sel = P[idx]  # (B, top_k, plen, d)

        k_sel = n_K[idx]  # (B, top_k, d)
        reduce_sim = (k_sel * q.unsqueeze(1)).sum() / B
        self._reduce_sim = reduce_sim if self._reduce_sim is None else (self._reduce_sim + reduce_sim)

        # L2P prompt-tuning (non-prefix): prompt tokens are shared for K/V in CLIP attention injection.
        Ek = P_sel.reshape(B, -1, self.emb_d)
        Ev = Ek
        return [Ek, Ev], x_querry.new_zeros(1), x_block


class OfficialDualPromptPrompt(nn.Module):
    """PyTorch DualPrompt semantics with configurable E/G prompt fields."""

    def __init__(
        self,
        emb_d: int,
        n_tasks: int,
        e_pool_size: int = 10,
        e_prompt_length: int = 5,
        g_prompt_length: int = 5,
        top_k: int = 1,
        batchwise_prompt: bool = True,
        use_prompt_mask: bool = True,
        use_g_prompt: bool = True,
        g_prompt_layer_idx=None,
        use_prefix_tune_for_g_prompt: bool = True,
        use_e_prompt: bool = True,
        e_prompt_layer_idx=None,
        use_prefix_tune_for_e_prompt: bool = True,
        key_dim: int = None,
        prompt_window_mode: str = "hard_session",
        prompt_eval_mode: str = "same_as_train",
    ):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = emb_d if key_dim is None else int(key_dim)
        self.n_tasks = n_tasks

        self.e_pool_size = int(e_pool_size)
        self.e_prompt_length = int(e_prompt_length)
        self.g_prompt_length = int(g_prompt_length)
        self.top_k = int(top_k)
        self.batchwise_prompt = bool(batchwise_prompt)
        self.use_prompt_mask = bool(use_prompt_mask)
        self.prompt_window_mode = str(prompt_window_mode).lower()
        self.prompt_eval_mode = str(prompt_eval_mode).lower()
        self.use_g_prompt = bool(use_g_prompt)
        self.use_prefix_tune_for_g_prompt = bool(use_prefix_tune_for_g_prompt)
        self.use_e_prompt = bool(use_e_prompt)
        self.use_prefix_tune_for_e_prompt = bool(use_prefix_tune_for_e_prompt)

        self.g_layers = list(g_prompt_layer_idx) if g_prompt_layer_idx is not None else [0, 1]
        self.e_layers = list(e_prompt_layer_idx) if e_prompt_layer_idx is not None else [2, 3, 4]

        self._reduce_sim = None
        self._last_idx = None
        self._last_visible = None
        self._last_mode = "global"
        self._train_usage = torch.zeros(self.e_pool_size, dtype=torch.long)
        self._eval_usage = torch.zeros(self.e_pool_size, dtype=torch.long)

        if self.use_g_prompt:
            for g in self.g_layers:
                if self.use_prefix_tune_for_g_prompt:
                    setattr(self, f"g_p_{g}", _tensor_prompt(2, self.g_prompt_length, emb_d))
                else:
                    setattr(self, f"g_p_{g}", _tensor_prompt(self.g_prompt_length, emb_d))

        if self.use_e_prompt:
            for e in self.e_layers:
                if self.use_prefix_tune_for_e_prompt:
                    setattr(self, f"e_p_{e}", _tensor_prompt(self.e_pool_size, 2, self.e_prompt_length, emb_d))
                else:
                    setattr(self, f"e_p_{e}", _tensor_prompt(self.e_pool_size, self.e_prompt_length, emb_d))
                setattr(self, f"e_k_{e}", _tensor_prompt(self.e_pool_size, self.key_d))

    def process_task_count(self):
        self.task_count += 1

    def reset_stats(self):
        self._reduce_sim = None

    def set_routing_modes(self, train_mode: str, eval_mode: str):
        self.prompt_window_mode = str(train_mode).lower()
        self.prompt_eval_mode = str(eval_mode).lower()

    def _active_mode(self, train: bool):
        if train:
            return self.prompt_window_mode
        if self.prompt_eval_mode == "same_as_train":
            return self.prompt_window_mode
        return self.prompt_eval_mode

    def _record_selection(self, idx: torch.Tensor, train: bool):
        idx_cpu = idx.detach().reshape(-1).cpu()
        binc = torch.bincount(idx_cpu, minlength=self.e_pool_size)
        if train:
            self._train_usage += binc
        else:
            self._eval_usage += binc

    def get_last_selected_indices(self):
        return self._last_idx

    def get_last_routing_info(self):
        return {
            "mode": self._last_mode,
            "visible_prompt_ids": list(self._last_visible) if self._last_visible is not None else None,
        }

    def get_usage_snapshot(self):
        train_nonzero = int((self._train_usage > 0).sum().item())
        eval_nonzero = int((self._eval_usage > 0).sum().item())
        return {
            "train_prompt_usage": self._train_usage.tolist(),
            "eval_prompt_usage": self._eval_usage.tolist(),
            "train_prompt_saturation": float(train_nonzero / max(self.e_pool_size, 1)),
            "eval_prompt_saturation": float(eval_nonzero / max(self.e_pool_size, 1)),
        }

    def get_last_reduce_sim(self):
        return self._reduce_sim

    def _select_prompt_indices(self, cos_sim: torch.Tensor, task_id: int, train: bool):
        bsz = cos_sim.shape[0]
        mode = self._active_mode(train)
        visible = None
        if self.use_prompt_mask:
            visible = _visible_prompt_indices(task_id, self.top_k, self.e_pool_size, cos_sim.device, mode)

        if visible is not None:
            masked = cos_sim.new_full(cos_sim.shape, float("-inf"))
            masked[:, visible] = cos_sim[:, visible]
            idx = torch.topk(masked, self.top_k, dim=1).indices
            self._last_visible = [int(v.item()) for v in visible.detach().cpu()]
        else:
            idx = torch.topk(cos_sim, self.top_k, dim=1).indices
            self._last_visible = None

        if self.batchwise_prompt:
            idx = _batchwise_major_prompt(idx, self.e_pool_size, self.top_k)
        self._last_mode = mode
        self._last_idx = idx.detach().cpu()
        self._record_selection(idx, train)
        return idx

    def forward(self, x_querry, l, x_block, train=False, task_id=None):
        B, _ = x_querry.shape

        e_valid = self.use_e_prompt and (l in self.e_layers)
        g_valid = self.use_g_prompt and (l in self.g_layers)
        loss = x_querry.new_zeros(1)

        if e_valid:
            K = getattr(self, f"e_k_{l}")
            P = getattr(self, f"e_p_{l}")
            n_K = F.normalize(K, dim=1)
            q = F.normalize(x_querry, dim=1)
            cos_sim = torch.einsum("bd,kd->bk", q, n_K)

            idx = self._select_prompt_indices(cos_sim, int(task_id) if task_id is not None else -1, bool(train))
            P_sel = P[idx]
            k_sel = n_K[idx]
            reduce_sim = (k_sel * q.unsqueeze(1)).sum() / B
            self._reduce_sim = reduce_sim if self._reduce_sim is None else (self._reduce_sim + reduce_sim)

            if self.use_prefix_tune_for_e_prompt:
                Ek = P_sel[:, :, 0, :, :].reshape(B, -1, self.emb_d)
                Ev = P_sel[:, :, 1, :, :].reshape(B, -1, self.emb_d)
            else:
                Ek = P_sel.reshape(B, -1, self.emb_d)
                Ev = Ek

        if g_valid:
            P_g = getattr(self, f"g_p_{l}")
            if self.use_prefix_tune_for_g_prompt:
                Gk = P_g[0].unsqueeze(0).expand(B, -1, -1)
                Gv = P_g[1].unsqueeze(0).expand(B, -1, -1)
            else:
                P_g = P_g.unsqueeze(0).expand(B, -1, -1)
                Gk = P_g
                Gv = P_g

        if e_valid and g_valid:
            return [torch.cat([Ek, Gk], dim=1), torch.cat([Ev, Gv], dim=1)], loss, x_block
        if e_valid:
            return [Ek, Ev], loss, x_block
        if g_valid:
            return [Gk, Gv], loss, x_block
        return None, loss, x_block
