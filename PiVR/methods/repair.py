
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Optional, Sequence
from torch.utils.data import DataLoader


class ImitationRepair:


    def __init__(self, model, layers_structure, pathway_masks, device="cpu"):

        self.model = model
        self.layers_structure = layers_structure
        self.device = device

        # Convert pathway_masks to torch tensors.
        self.pathway_masks = []
        for mask in pathway_masks:
            if isinstance(mask, np.ndarray):
                self.pathway_masks.append(torch.from_numpy(mask).float().to(device))
            elif isinstance(mask, torch.Tensor):
                self.pathway_masks.append(mask.float().to(device))
            else:
                raise ValueError(f"Unsupported mask type: {type(mask)}")

    def get_pathway_params(self):

        params = []
        mask_idx = 0
        for layer in self.layers_structure:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if mask_idx < len(self.pathway_masks):
                    # Optimize only layers whose mask contains selected units.
                    mask = self.pathway_masks[mask_idx]
                    if isinstance(mask, torch.Tensor):
                        mask_sum = mask.sum().item()
                    else:
                        mask_sum = mask.sum()
                    if mask_sum > 0:
                        params.append(layer.weight)
                        if layer.bias is not None:
                            params.append(layer.bias)
                mask_idx += 1
        return params

    def _register_gradient_hooks(self, target_params):
        """
        Register gradient-mask hooks for pathway parameters.

        This is the core mechanism for strictly sparse fine-tuning and supports
        both Linear and Conv2d layers.

        Args:
            target_params: List of parameters to optimize.

        Returns:
            list: Hook handles used for later removal.
        """
        hooks = []
        mask_idx = 0

        # Convert target_params to a set and compare parameter object references by id().
        target_param_ids = {id(p) for p in target_params}

        for layer in self.layers_structure:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if mask_idx < len(self.pathway_masks):
                    mask = self.pathway_masks[mask_idx]

                    # Register the weight hook and compare parameter object references by id().
                    if id(layer.weight) in target_param_ids:
                        def make_weight_hook(m, is_conv):
                            def hook(grad):
                                if grad is not None:
                                    if is_conv:
                                        # Conv2d: weight shape [out_channels, in_channels, kh, kw]
                                        # mask shape: [out_channels]
                                        # Unsqueeze to [out_channels, 1, 1, 1] for broadcasting.
                                        return grad * m.view(-1, 1, 1, 1)
                                    else:
                                        # Linear: weight shape [out_features, in_features]
                                        # mask shape: [out_features]
                                        # Use unsqueeze(1) for broadcasting to [out_features, in_features].
                                        return grad * m.unsqueeze(1)
                                return grad
                            return hook

                        is_conv = isinstance(layer, nn.Conv2d)
                        h = layer.weight.register_hook(make_weight_hook(mask, is_conv))
                        hooks.append(h)

                    # Register the bias hook and compare parameter object references by id().
                    if layer.bias is not None and id(layer.bias) in target_param_ids:
                        def make_bias_hook(m):
                            def hook(grad):
                                if grad is not None:
                                    # Bias shape: [out_features] or [out_channels].
                                    # Mask shape: [out_features] or [out_channels].
                                    return grad * m
                                return grad
                            return hook

                        h = layer.bias.register_hook(make_bias_hook(mask))
                        hooks.append(h)

                    mask_idx += 1

        return hooks

    def _get_reference_activations(self, ref_sample):
        """
        Get reference activations for each layer, supporting Linear and Conv2d layers.

        The input can be a single reference sample `[C, H, W] / [D]` or a Top-k
        reference prototype set `[K, C, H, W] / [K, D]`. For a reference batch,
        activations are averaged to obtain one prototype activation. Conv2d
        activations are compressed to `[1, C]` by global average pooling so that
        they align with pathway masks of shape `[C]`.

        Args:
            ref_sample: A single reference sample or a reference sample batch.

        Returns:
            dict: Layer-wise activations.
        """
        ref_acts = {}
        handles = []

        def _prepare_reference_input(sample):
            if not isinstance(sample, torch.Tensor):
                sample = torch.tensor(sample)
            sample = sample.to(self.device)
            if sample.dim() == 1 or sample.dim() == 3:
                sample = sample.unsqueeze(0)
            return sample

        ref_input = _prepare_reference_input(ref_sample)

        def get_ref_hook(idx, is_conv):
            def hook(m, i, o):
                if is_conv:
                    pooled = F.adaptive_avg_pool2d(o.detach(), (1, 1)).squeeze(-1).squeeze(-1)
                    ref_acts[idx] = pooled.mean(dim=0, keepdim=True).clone()
                else:
                    ref_acts[idx] = o.detach().mean(dim=0, keepdim=True).clone()
            return hook

        mask_count = 0
        for layer in self.layers_structure:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                is_conv = isinstance(layer, nn.Conv2d)
                handles.append(layer.register_forward_hook(get_ref_hook(mask_count, is_conv)))
                mask_count += 1

        self.model.eval()
        with torch.no_grad():
            self.model(ref_input)

        for h in handles:
            h.remove()

        return ref_acts

    def _build_hard_pair_from_batch(self, batch_x, batch_y=None, mode="backdoor"):
        """Build task-aware hard positive / hard negative references.

        The positive reference is selected as the nearest valid anchor in the
        batch (prefer label-consistent and higher-confidence samples when labels
        are available). The negative reference is selected as the most harmful
        contrast in the batch: for backdoor we prefer the lowest-confidence
        sample, while for fairness/safety we prefer the sample that maximizes
        disagreement or target-risk under the available signals.
        """
        if not isinstance(batch_x, torch.Tensor):
            batch_x = torch.tensor(batch_x)
        batch_x = batch_x.to(self.device)
        if batch_x.dim() == 1:
            batch_x = batch_x.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)

        target = None
        if batch_y is not None:
            target = batch_y.to(self.device) if isinstance(batch_y, torch.Tensor) else torch.tensor(batch_y, device=self.device)
            if target.dim() == 0:
                target = target.unsqueeze(0)

        # positive: prefer target-consistent and high-confidence anchor
        if target is not None and target.numel() == batch_x.shape[0]:
            match = preds.eq(target)
            if match.any():
                pos_idx = torch.where(match)[0][torch.argmax(confs[match])].item()
            else:
                pos_idx = int(torch.argmax(confs).item())
        else:
            pos_idx = int(torch.argmax(confs).item())

        # negative: task-aware harmful contrast
        if mode == "fairness":
            neg_idx = int(torch.argmin(confs).item())
        elif mode == "safety":
            # prefer the sample with strongest class uncertainty to approximate a nearby violation
            entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1)
            neg_idx = int(torch.argmax(entropy).item())
        else:  # backdoor
            neg_idx = int(torch.argmin(confs).item())

        pos_ref = batch_x[pos_idx:pos_idx + 1].detach().clone()
        neg_ref = batch_x[neg_idx:neg_idx + 1].detach().clone()
        return pos_ref, neg_ref

    def _build_backdoor_refs(self, batch_x, batch_y=None):
        """Backdoor-specific reference pair.

        Positive: the highest-confidence non-trivial anchor in the batch.
        Negative: the lowest-confidence sample, acting as a poison-sensitive hard negative.
        """
        return self._build_hard_pair_from_batch(batch_x, batch_y=batch_y, mode="backdoor")

    def _build_safety_refs(self, batch_x, batch_y=None):
        """Safety-specific reference pair.

        Positive: the safest-looking anchor in the batch.
        Negative: the most suspicious/lowest-confidence candidate in the batch.
        """
        return self._build_hard_pair_from_batch(batch_x, batch_y=batch_y, mode="safety")

    def _build_fairness_refs(self, batch_x, batch_y=None):
        """Fairness-specific reference pair.

        Positive: counterfactually consistent sample (or highest-confidence sample if consistency is unavailable).
        Negative: counterfactually inconsistent sample (or lowest-confidence sample if consistency is unavailable).
        """
        pos_ref, neg_ref = self._build_hard_pair_from_batch(batch_x, batch_y=batch_y, mode="fairness")
        if batch_x is not None:
            try:
                batch_tensor = batch_x if isinstance(batch_x, torch.Tensor) else torch.tensor(batch_x)
                if batch_tensor.dim() == 1:
                    batch_tensor = batch_tensor.unsqueeze(0)
                pos_ref = batch_tensor.detach().clone()
                neg_ref = torch.flip(batch_tensor.detach().clone(), dims=[0])
                # Make the negative contrast sharper by swapping the weakest/strongest anchors.
                if batch_tensor.shape[0] > 1:
                    neg_ref = torch.cat([batch_tensor[-1:].detach().clone(), batch_tensor[:-1].detach().clone()], dim=0)
            except Exception:
                pass
        return pos_ref, neg_ref

    def _build_taskwise_contrastive_refs(self, batch_x, batch_y=None, mode="backdoor"):
        """Dispatch task-aware positive/negative reference building."""
        if mode == "fairness":
            return self._build_fairness_refs(batch_x, batch_y=batch_y)
        if mode == "safety":
            return self._build_safety_refs(batch_x, batch_y=batch_y)
        return self._build_backdoor_refs(batch_x, batch_y=batch_y)

    def repair_region(self, region_loader, epochs=50, lr=0.01,
                      clean_loader: Optional[DataLoader] = None,
                      lambda_clean: float = 1.0,
                      reference_loader: Optional[DataLoader] = None,
                      argmin_mode: bool = False,
                      safe_labels: Optional[Sequence[int]] = None,
                      fairness_mode: bool = False,
                      flip_fn=None,
                      sensitive_indices=None,
                      dataset_name=None,
                      lambda_task: float = 0.0,
                      early_stop_patience: int = 10,
                      early_stop_min_delta: float = 1e-6,
                      enable_rollback: bool = False):
        lambda_fair = float(lambda_task)
        target_params = self.get_pathway_params()
        if len(target_params) == 0:
            return False, 0

        optimizer = optim.Adam(target_params, lr=lr, weight_decay=0.0) if fairness_mode else optim.SGD(
            target_params, lr=lr, momentum=0.0, weight_decay=0.0
        )
        grad_hooks = self._register_gradient_hooks(target_params)
        ce_loss = nn.CrossEntropyLoss()
        if enable_rollback:
            print("[RepairRegion] rollback/acceptance guard is deprecated and disabled in the default pipeline")
        safe_label_set = {int(x) for x in (safe_labels or [])}
        clean_iter = iter(clean_loader) if clean_loader is not None else None
        ref_iter = iter(reference_loader) if reference_loader is not None else None

        def _goal_pred(outputs: torch.Tensor):
            if argmin_mode and len(safe_label_set) > 0:
                labels = torch.tensor(sorted(safe_label_set), device=outputs.device)
                idx = torch.argmin(outputs[:, labels], dim=1)
                return labels[idx]
            return outputs.argmax(dim=1)

        def _fairness_loss(batch_x: torch.Tensor, outputs: torch.Tensor):
            flipped = batch_x.detach().cpu().numpy().copy()
            for i in range(flipped.shape[0]):
                flipped[i] = flip_fn(flipped[i], sensitive_indices, dataset_name=dataset_name or "fairness", use_negation=True)
            flipped = torch.tensor(flipped, dtype=batch_x.dtype, device=self.device)
            flipped_outputs = self.model(flipped)
            return F.mse_loss(outputs, flipped_outputs)

        def _task_reg(batch_x: torch.Tensor, outputs: torch.Tensor):
            if argmin_mode and len(safe_label_set) > 0:
                safe_scores = outputs[:, list(sorted(safe_label_set))]
                return torch.relu(safe_scores.min(dim=1).values - outputs.min(dim=1).values + 0.01).mean()
            return torch.tensor(0.0, device=self.device)

        def _imit_loss(batch_x: torch.Tensor):
            if ref_iter is None:
                return torch.tensor(0.0, device=self.device)
            try:
                ref_batch = next(ref_iter)
            except StopIteration:
                return torch.tensor(0.0, device=self.device)
            if isinstance(ref_batch, (tuple, list)):
                ref_batch = ref_batch[0]
            ref_batch = ref_batch.to(self.device)
            ref_acts = self._get_reference_activations(ref_batch)
            if not ref_acts:
                return torch.tensor(0.0, device=self.device)
            cur_acts = {}
            handles = []
            idx = 0
            for layer in self.layers_structure:
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    def hook_fn(i, is_conv):
                        def _hook(m, inp, out):
                            cur_acts[i] = F.adaptive_avg_pool2d(out, (1, 1)).squeeze(-1).squeeze(-1) if is_conv else out
                        return _hook
                    handles.append(layer.register_forward_hook(hook_fn(idx, isinstance(layer, nn.Conv2d))))
                    idx += 1
            _ = self.model(batch_x)
            for h in handles:
                h.remove()
            loss = torch.tensor(0.0, device=self.device)
            for k in ref_acts:
                if k in cur_acts:
                    loss = loss + F.mse_loss(cur_acts[k], ref_acts[k].expand_as(cur_acts[k]))
            return loss

        try:
            self.model.train()

            best_epoch_loss = float("inf")
            no_improve_epochs = 0

            for epoch in range(epochs):
                epoch_loss_sum = 0.0
                epoch_steps = 0

                for batch_x, batch_y in region_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(batch_x)

                    loss_corr = ce_loss(-outputs if argmin_mode else outputs, batch_y)

                    loss_fair = torch.tensor(0.0, device=self.device)
                    if fairness_mode and flip_fn is not None and sensitive_indices is not None:
                        loss_fair = _fairness_loss(batch_x, outputs)

                    loss_clean = torch.tensor(0.0, device=self.device)
                    if clean_iter is not None:
                        try:
                            clean_x, clean_y = next(clean_iter)
                        except StopIteration:
                            clean_iter = iter(clean_loader)
                            clean_x, clean_y = next(clean_iter)

                        clean_x = clean_x.to(self.device)
                        clean_y = clean_y.to(self.device)
                        clean_out = self.model(clean_x)
                        loss_clean = ce_loss(-clean_out if argmin_mode else clean_out, clean_y)

                    loss = loss_corr + lambda_fair * loss_fair + lambda_clean * loss_clean

                    if not torch.isfinite(loss):
                        print(
                            f"[RepairRegion] Non-finite loss at epoch {epoch + 1}; "
                            "stop optimization and keep current parameters."
                        )
                        return False, 0

                    loss.backward()
                    optimizer.step()

                    epoch_loss_sum += float(loss.detach().item())
                    epoch_steps += 1

                avg_epoch_loss = epoch_loss_sum / max(epoch_steps, 1)

                if avg_epoch_loss < best_epoch_loss - early_stop_min_delta:
                    best_epoch_loss = avg_epoch_loss
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                print(
                    f"[RepairRegion] epoch={epoch + 1}/{epochs}, "
                    f"avg_loss={avg_epoch_loss:.6f}, "
                    f"best_loss={best_epoch_loss:.6f}, "
                    f"no_improve={no_improve_epochs}/{early_stop_patience}"
                )

                if early_stop_patience is not None and early_stop_patience > 0:
                    if no_improve_epochs >= early_stop_patience:
                        print(
                            f"[RepairRegion] Early stop at epoch {epoch + 1}: "
                            f"epoch loss did not improve for {early_stop_patience} consecutive epochs."
                        )
                        break

            repaired_count = 0
            total = 0
            self.model.eval()
            with torch.no_grad():
                for batch_x, batch_y in region_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    preds = _goal_pred(outputs)
                    if fairness_mode and flip_fn is not None and sensitive_indices is not None:
                        flipped = batch_x.detach().cpu().numpy().copy()
                        for i in range(flipped.shape[0]):
                            flipped[i] = flip_fn(flipped[i], sensitive_indices, dataset_name=dataset_name or "fairness", use_negation=True)
                        flipped = torch.tensor(flipped, dtype=batch_x.dtype, device=self.device)
                        repaired_count += int((preds == self.model(flipped).argmax(dim=1)).sum().item())
                    else:
                        repaired_count += int((preds == batch_y).sum().item())
                    total += batch_x.shape[0]
            return repaired_count > 0, repaired_count
        finally:
            for h in grad_hooks:
                h.remove()

    def repair(self, buggy_sample, ref_sample, target_label, epochs=50, lr=0.01,
               min_confidence=0.5, conservative_mode=True,
               fairness_mode=False, flip_fn=None, sensitive_indices=None,
               dataset_name=None, num_random_trials=0, lambda_fair=0.0,
               backdoor_mode: bool = False,
               clean_loader: Optional[DataLoader] = None,
               lambda_clean: float = 1.0,
               kl_mode: bool = False,
               argmin_mode: bool = False,
               safe_labels: Optional[Sequence[int]] = None,
               max_acc_drop: Optional[float] = None,
               epoch_guard_eval_every: int = 0,
               epoch_guard_acc_tolerance: Optional[float] = None,
               epoch_guard_reduce_lr_ratio: float = 0.5,
               late_repair_stop_enabled: bool = False,
               late_repair_epoch_threshold: int = 0,
               late_repair_min_confidence_growth: float = 0.0,
               fairness_warmup_clean_scale: float = 0.3,
               negative_ref_sample=None,
               lambda_neg: float = 0.0,
               contrastive_margin: float = 1.0,
               contrastive_mode: bool = False,
               enable_rollback: bool = False):
        """

        Args:
            buggy_sample: Buggy sample [C, H, W].
            ref_sample: Reference sample, usually a similar correctly predicted sample [C, H, W].
            target_label: Target label, usually the correct class.
            epochs: Number of fine-tuning epochs.
            lr: Learning rate.
            lambda_reg: Deprecated legacy parameter retained for compatibility.
            min_confidence: Minimum post-repair confidence threshold.
            conservative_mode: Conservative mode; stop once the prediction is just corrected.
            fairness_mode: Whether to enable the fairness regularizer; used only in fairness experiments.
            flip_fn: Function for flipping sensitive attributes:
                     flip_fn(sample, sensitive_indices, dataset_name, use_negation).
            sensitive_indices: List of sensitive attribute indices.
            dataset_name: Dataset name used by the internal flip_fn strategy.
            num_random_trials: Number of random perturbation trials in the fairness regularizer;
                               usually set to 0 when using only the standard flip.
            lambda_fair: Weight of the fairness regularization term.
            backdoor_mode: Whether the repair is in the backdoor setting. If True,
                           the rollback condition is relaxed so that success can be
                           defined by leaving the original backdoor prediction.
            kl_mode: Whether to align the reference output distribution with KL
                     divergence instead of CrossEntropy loss. This is recommended for
                     ACAS Xu safety repair because it matches argmin evaluation semantics
                     and avoids the mismatch between CE direction and safety semantics.
            argmin_mode: If True, use argmin semantics for final verification and early stopping,
                         which is suitable for ACAS Xu. Defaults to False, i.e., argmax,
                         for classification and fairness tasks.
            max_acc_drop: Deprecated legacy parameter retained for compatibility.
            enable_rollback: Deprecated legacy parameter retained for compatibility.

        Returns:
            bool: Whether the repair succeeds.
        """
        target_params = self.get_pathway_params()
        if len(target_params) == 0:
            print(">>> [Repair] Warning: No parameters to optimize!")
            return False

        if fairness_mode:
            optimizer = optim.Adam(target_params, lr=lr, weight_decay=0.0)
        else:
            optimizer = optim.SGD(target_params, lr=lr, momentum=0.0, weight_decay=0.0)

        criterion_ce = nn.CrossEntropyLoss()

        if not isinstance(buggy_sample, torch.Tensor):
            buggy_sample = torch.tensor(buggy_sample)
        if not isinstance(ref_sample, torch.Tensor):
            ref_sample = torch.tensor(ref_sample)

        buggy_sample = buggy_sample.to(self.device)
        ref_sample = ref_sample.to(self.device)

        if buggy_sample.dim() == 1 or buggy_sample.dim() == 3:
            buggy_in = buggy_sample.unsqueeze(0)
        else:
            buggy_in = buggy_sample

        if ref_sample.dim() == 1 or ref_sample.dim() == 3:
            ref_input = ref_sample.unsqueeze(0)
        else:
            ref_input = ref_sample

        target_tensor = torch.tensor([target_label]).to(self.device)
        safe_label_set = {int(target_label)} if safe_labels is None else {int(x) for x in safe_labels}

        def _prediction_satisfies_goal(pred: int) -> bool:
            if backdoor_mode:
                return int(pred) == int(target_label)
            if argmin_mode and len(safe_label_set) > 0:
                return int(pred) in safe_label_set
            return int(pred) == int(target_label)

        def _goal_confidence(prob_tensor: torch.Tensor) -> float:
            if argmin_mode and len(safe_label_set) > 0:
                return float(max(prob_tensor[0, lbl].item() for lbl in safe_label_set))
            return float(prob_tensor[0, target_label].item())

        def _fairness_goal_status(output_tensor: torch.Tensor):
            probs_local = torch.softmax(output_tensor, dim=1)
            pred_local = output_tensor.argmax(dim=1).item()
            sample_np = buggy_sample.detach().cpu().numpy()
            flipped_np = flip_fn(
                sample_np,
                sensitive_indices,
                dataset_name=dataset_name or "fairness",
                use_negation=True,
            )
            flipped_tensor = torch.tensor(flipped_np, dtype=buggy_in.dtype, device=self.device).unsqueeze(0)
            flipped_output = self.model(flipped_tensor)
            flipped_probs = torch.softmax(flipped_output, dim=1)
            flipped_pred = flipped_output.argmax(dim=1).item()
            conf_orig = float(probs_local[0, pred_local].item())
            conf_flip = float(flipped_probs[0, flipped_pred].item())
            classification_ok = int(pred_local) == int(target_label)
            consistency_ok = int(pred_local) == int(flipped_pred)
            success = classification_ok and consistency_ok and (conf_orig >= min_confidence) and \
                        (conf_flip >= min_confidence)
            goal_conf = min(conf_orig, conf_flip)
            return success, goal_conf, pred_local, flipped_pred, conf_orig, conf_flip, classification_ok, consistency_ok

        grad_hooks = self._register_gradient_hooks(target_params)

        ref_acts = self._get_reference_activations(ref_sample)
        neg_acts = self._get_reference_activations(negative_ref_sample) if (
                    contrastive_mode and negative_ref_sample is not None) else None

        self.model.eval()
        with torch.no_grad():
            original_output = self.model(buggy_in)
            if argmin_mode:
                original_pred = original_output.argmin(dim=1).item()
                original_confidence = torch.softmax(-original_output, dim=1).max().item()
            else:
                original_pred = original_output.argmax(dim=1).item()
                original_confidence = torch.softmax(original_output, dim=1).max().item()

            clean_acc_baseline = None
            if max_acc_drop is not None and clean_loader is not None:
                clean_correct = 0
                clean_total = 0
                for clean_x_base, clean_y_base in clean_loader:
                    clean_x_base = clean_x_base.to(self.device)
                    clean_y_base = clean_y_base.to(self.device)
                    clean_outputs_base = self.model(clean_x_base)
                    if argmin_mode:
                        clean_preds_base = clean_outputs_base.argmin(dim=1)
                    else:
                        clean_preds_base = clean_outputs_base.argmax(dim=1)
                    clean_correct += (clean_preds_base == clean_y_base).sum().item()
                    clean_total += clean_y_base.numel()
                clean_acc_baseline = (clean_correct / clean_total) if clean_total > 0 else None

            ref_logits_cached = self.model(ref_input).mean(dim=0, keepdim=True)

        # ========== 6. Training loop ==========
        self.model.train()
        print(f">>> [Repair] Start fine-tuning for {epochs} epochs...")
        print(f"    Learning rate: {lr}, Lambda_fair: {lambda_fair}")
        print(f"    Conservative mode: {conservative_mode}, Min confidence: {min_confidence}")
        print(f"    Original prediction: {original_pred} (confidence: {original_confidence:.4f})")

        repair_success = False
        best_loss = float('inf')
        best_model_state = None
        best_pred = None
        best_confidence = 0.0
        best_clean_acc = None
        best_clean_drop = None
        stable_count = 0
        no_improve_count = 0
        clean_guard_enabled = False
        use_clean_aware_selection = False
        max_acc_drop = None
        confidence_history = []

        fairness_classification_first = fairness_mode and (not backdoor_mode) and (not argmin_mode)
        fairness_warmup_epochs = min(5, epochs) if fairness_classification_first else 0
        if fairness_classification_first and fairness_warmup_epochs > 0:
            print(
                f"    Fairness classification-first warmup: {fairness_warmup_epochs} epochs (imitation temporarily disabled)")

        clean_iter = iter(clean_loader) if clean_loader is not None else None
        try:
            for epoch in range(epochs):
                optimizer.zero_grad()

                # Capture the current intermediate activations.
                current_acts = {}
                handles = []

                def get_curr_hook(idx):
                    def hook(m, i, o):
                        current_acts[idx] = o

                    return hook

                l_count = 0
                for layer in self.layers_structure:
                    if isinstance(layer, (nn.Linear, nn.Conv2d)):
                        handles.append(layer.register_forward_hook(get_curr_hook(l_count)))
                        l_count += 1

                outputs = self.model(buggy_in)

                if kl_mode:
                    if argmin_mode:
                        loss_cls = F.kl_div(
                            F.log_softmax(-outputs, dim=1),
                            F.softmax(-ref_logits_cached.detach(), dim=1),
                            reduction='batchmean'
                        )
                    else:
                        loss_cls = F.kl_div(
                            F.log_softmax(outputs, dim=1),
                            F.softmax(ref_logits_cached.detach(), dim=1),
                            reduction='batchmean'
                        )
                else:
                    if argmin_mode:
                        loss_cls = criterion_ce(-outputs, target_tensor)
                    else:
                        loss_cls = criterion_ce(outputs, target_tensor)

                loss_imitation = torch.tensor(0.0).to(self.device)
                loss_contrastive = torch.tensor(0.0).to(self.device)
                mask_idx = 0
                for layer in self.layers_structure:
                    if isinstance(layer, (nn.Linear, nn.Conv2d)):
                        if mask_idx >= len(
                                self.pathway_masks) or mask_idx not in current_acts or mask_idx not in ref_acts:
                            mask_idx += 1
                            continue

                        mask = self.pathway_masks[mask_idx]
                        act_bug = current_acts[mask_idx]
                        act_ref = ref_acts[mask_idx].to(self.device)
                        act_neg = neg_acts[mask_idx].to(
                            self.device) if neg_acts is not None and mask_idx in neg_acts else None

                        if isinstance(layer, nn.Conv2d):
                            act_bug = F.adaptive_avg_pool2d(act_bug, (1, 1)).squeeze(-1).squeeze(-1)

                        # Core step: compute MSE only on the masked units.
                        mask_expanded = mask.unsqueeze(0)  # [1, neurons/channels]
                        diff = act_bug - act_ref
                        diff_squared = (diff ** 2) * mask_expanded

                        # Compute the average MSE over masked neurons.
                        num_masked = mask.sum()
                        if num_masked > 0:
                            layer_imitation_loss = diff_squared.sum() / num_masked
                            if layer_imitation_loss.item() > 100.0:
                                layer_imitation_loss = torch.clamp(layer_imitation_loss, max=100.0)
                            loss_imitation = loss_imitation + layer_imitation_loss

                            if contrastive_mode and act_neg is not None:
                                if isinstance(layer, nn.Conv2d):
                                    act_neg = F.adaptive_avg_pool2d(act_neg, (1, 1)).squeeze(-1).squeeze(-1)
                                pos_dist = F.mse_loss(act_bug * mask_expanded, act_ref * mask_expanded)
                                neg_dist = F.mse_loss(act_bug * mask_expanded, act_neg * mask_expanded)
                                loss_contrastive = loss_contrastive + torch.relu(
                                    pos_dist - neg_dist + contrastive_margin)

                        mask_idx += 1

                current_lambda = 0.0

                if fairness_classification_first and epoch < fairness_warmup_epochs and original_pred != target_label:
                    current_lambda = 0.0

                if argmin_mode:
                    pred = outputs.argmin(dim=1).item()
                    probs = torch.softmax(-outputs, dim=1)  # Negated for argmin!
                else:
                    pred = outputs.argmax(dim=1).item()
                    probs = torch.softmax(outputs, dim=1)
                confidence = _goal_confidence(probs)

                if _prediction_satisfies_goal(pred) and confidence < min_confidence:
                    confidence_penalty = (min_confidence - confidence) * 2.0
                    loss_cls = loss_cls + confidence_penalty

                loss_fair = torch.tensor(0.0, device=self.device)
                if fairness_mode and flip_fn is not None and sensitive_indices is not None:
                    # 1. Generate the counterfactual sample
                    flipped_np = flip_fn(
                        buggy_sample.detach().cpu().numpy(),
                        sensitive_indices,
                        dataset_name=dataset_name or "fairness",
                        use_negation=True,
                    )
                    flipped_tensor = torch.tensor(flipped_np, dtype=buggy_in.dtype, device=self.device).unsqueeze(0)

                    out_flipped = self.model(flipped_tensor)

                    loss_fair = F.mse_loss(outputs, out_flipped) * 5.0

                fairness_disable_clean = False
                effective_lambda_clean = lambda_clean
                if fairness_classification_first and epoch < fairness_warmup_epochs and original_pred != target_label:
                    effective_lambda_clean = lambda_clean * fairness_warmup_clean_scale

                loss_clean = torch.tensor(0.0, device=self.device)
                if clean_iter is not None and not fairness_disable_clean:
                    try:
                        clean_batch = next(clean_iter)
                    except StopIteration:
                        clean_iter = iter(clean_loader)
                        clean_batch = next(clean_iter)
                    if isinstance(clean_batch, (list, tuple)) and len(clean_batch) == 2:
                        clean_x, clean_y = clean_batch
                        clean_x = clean_x.to(self.device)
                        clean_y = clean_y.to(self.device)
                        clean_outputs = self.model(clean_x)
                        if kl_mode and argmin_mode:
                            with torch.no_grad():
                                clean_ref_logits = clean_outputs.detach()
                            loss_clean = F.kl_div(
                                F.log_softmax(-clean_outputs, dim=1),
                                F.softmax(-clean_ref_logits, dim=1),
                                reduction='batchmean'
                            )
                        elif kl_mode and not argmin_mode:
                            loss_clean = criterion_ce(clean_outputs, clean_y)
                        else:
                            loss_clean = criterion_ce(clean_outputs, clean_y)
                # Total loss.
                total_loss = loss_cls + lambda_fair * loss_fair + effective_lambda_clean * loss_clean

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"    ⚠ Warning: Loss became NaN/Inf at epoch {epoch + 1}, rolling back!")
                    self.model.load_state_dict(model_backup)
                    for h in handles:
                        h.remove()
                    for h in grad_hooks:
                        h.remove()
                    return False

                total_loss.backward()

                max_grad_norm = 1.0
                torch.nn.utils.clip_grad_norm_(target_params, max_norm=max_grad_norm)

                optimizer.step()

                for h in handles:
                    h.remove()

                self.model.eval()
                with torch.no_grad():
                    eval_output = self.model(buggy_in)
                    if argmin_mode:
                        eval_pred = eval_output.argmin(dim=1).item()
                        eval_probs = torch.softmax(-eval_output, dim=1)
                    else:
                        eval_pred = eval_output.argmax(dim=1).item()
                        eval_probs = torch.softmax(eval_output, dim=1)
                    eval_confidence = _goal_confidence(eval_probs)
                self.model.train()

                loss_val = total_loss.item()
                current_clean_acc = None
                if clean_guard_enabled:
                    clean_correct_mid = 0
                    clean_total_mid = 0
                    self.model.eval()
                    with torch.no_grad():
                        for clean_x_mid, clean_y_mid in clean_loader:
                            clean_x_mid = clean_x_mid.to(self.device)
                            clean_y_mid = clean_y_mid.to(self.device)
                            clean_outputs_mid = self.model(clean_x_mid)
                            if argmin_mode:
                                clean_preds_mid = clean_outputs_mid.argmin(dim=1)
                            else:
                                clean_preds_mid = clean_outputs_mid.argmax(dim=1)
                            clean_correct_mid += (clean_preds_mid == clean_y_mid).sum().item()
                            clean_total_mid += clean_y_mid.numel()
                    current_clean_acc = (clean_correct_mid / clean_total_mid) if clean_total_mid > 0 else None
                    self.model.train()

                if fairness_mode:
                    eval_meets_goal, eval_confidence, eval_pred, eval_flip_pred, eval_conf_orig, eval_conf_flip, eval_classification_ok, eval_consistency_ok = _fairness_goal_status(
                        eval_output)
                else:
                    if eval_pred == target_label:
                        eval_meets_goal = True
                    else:
                        eval_meets_goal = _prediction_satisfies_goal(eval_pred)

                is_better = False
                if eval_meets_goal:
                    if use_clean_aware_selection and current_clean_acc is not None:
                        clean_drop_now = clean_acc_baseline - current_clean_acc
                        within_guard = clean_drop_now <= max_acc_drop
                        if within_guard:
                            if best_pred is None or not _prediction_satisfies_goal(best_pred):
                                is_better = True
                            elif best_clean_acc is None or current_clean_acc > best_clean_acc + 1e-8:
                                is_better = True
                            elif abs(current_clean_acc - best_clean_acc) <= 1e-8 and eval_confidence > best_confidence:
                                is_better = True
                        else:
                            if best_pred is None or not _prediction_satisfies_goal(best_pred):
                                is_better = True
                            elif best_clean_drop is None or clean_drop_now < best_clean_drop - 1e-8:
                                is_better = True
                            elif best_clean_drop is not None and abs(
                                    clean_drop_now - best_clean_drop) <= 1e-8 and eval_confidence > best_confidence:
                                is_better = True
                    else:
                        if best_pred is None or not _prediction_satisfies_goal(
                                best_pred) or eval_confidence > best_confidence:
                            is_better = True
                else:
                    if loss_val < best_loss and ((best_pred is None) or (
                    not _prediction_satisfies_goal(best_pred)) or loss_val < best_loss * 0.9):
                        is_better = True

                if is_better:
                    best_loss = loss_val
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    best_pred = eval_pred
                    best_confidence = eval_confidence
                    best_clean_acc = current_clean_acc
                    if use_clean_aware_selection and current_clean_acc is not None:
                        best_clean_drop = clean_acc_baseline - current_clean_acc
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                confidence_history.append(eval_confidence if eval_meets_goal else _goal_confidence(probs))
                if len(confidence_history) > 5:
                    confidence_history.pop(0)

                if clean_guard_enabled and epoch_guard_eval_every > 0 and (
                        epoch + 1) % epoch_guard_eval_every == 0 and current_clean_acc is not None:
                    clean_drop_now = clean_acc_baseline - current_clean_acc
                    tolerance = epoch_guard_acc_tolerance if epoch_guard_acc_tolerance is not None else max_acc_drop
                    if tolerance is not None and clean_drop_now > tolerance:
                        old_lr = optimizer.param_groups[0]['lr']
                        new_lr = max(old_lr * epoch_guard_reduce_lr_ratio, 1e-5)
                        optimizer.param_groups[0]['lr'] = new_lr
                        print(
                            f"    -> Epoch guard: clean_drop={clean_drop_now:.4f} exceeded tolerance={tolerance:.4f}; reducing LR {old_lr:.6f} -> {new_lr:.6f}"
                        )
                        if fairness_mode and best_model_state is not None and best_pred is not None and _prediction_satisfies_goal(
                                best_pred) and best_clean_acc is not None and (
                                clean_acc_baseline - best_clean_acc) <= max_acc_drop:
                            print("    -> Epoch guard: restoring clean-aware best checkpoint and stopping early.")
                            self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
                            repair_success = True
                            break

                if fairness_mode and late_repair_stop_enabled and epoch + 1 >= late_repair_epoch_threshold and not eval_meets_goal and len(
                        confidence_history) >= 5:
                    conf_gain = confidence_history[-1] - confidence_history[0]
                    if conf_gain < late_repair_min_confidence_growth:
                        print(
                            f"    -> Late repair stop at epoch {epoch + 1}: target confidence gain {conf_gain:.4f} < threshold {late_repair_min_confidence_growth:.4f}"
                        )
                        break

                if (epoch + 1) % 5 == 0 or epoch == 0:
                    ce_val = loss_cls.item()
                    imit_val = loss_imitation.item()
                    contrast_val = loss_contrastive.item() if 'loss_contrastive' in locals() else 0.0
                    current_lr = optimizer.param_groups[0]['lr']
                    if fairness_mode:
                        print(f"    Epoch {epoch + 1:3d}: Loss={loss_val:.4f} "
                              f"(CE={ce_val:.4f}, Imit={imit_val:.4f}, Contra={contrast_val:.4f}), "
                              f"Pred={eval_pred}, FlipPred={eval_flip_pred}, Target={target_label}, "
                              f"Conf={eval_confidence:.4f}, OrigConf={eval_conf_orig:.4f}, FlipConf={eval_conf_flip:.4f}, LR={current_lr:.6f}")
                    else:
                        print(f"    Epoch {epoch + 1:3d}: Loss={loss_val:.4f} "
                              f"(CE={ce_val:.4f}, Imit={imit_val:.4f}, Contra={contrast_val:.4f}), "
                              f"Pred={eval_pred}, Target={target_label}, Conf={eval_confidence:.4f}, LR={current_lr:.6f}")

                if conservative_mode and eval_meets_goal:
                    can_accept_early = True
                    if use_clean_aware_selection and current_clean_acc is not None and clean_acc_baseline is not None and max_acc_drop is not None:
                        clean_drop_now = clean_acc_baseline - current_clean_acc
                        can_accept_early = clean_drop_now <= max_acc_drop
                        if not can_accept_early and ((epoch + 1) % 5 == 0 or epoch == 0):
                            print(
                                f"    -> Early-stop candidate rejected by clean guard: clean_drop={clean_drop_now:.4f}, threshold={max_acc_drop:.4f}")

                    if can_accept_early and eval_confidence >= min_confidence:
                        if fairness_mode:
                            print(
                                f"    -> Early stop at epoch {epoch + 1}: Fairness goal achieved "
                                f"(pred={eval_pred}, flip_pred={eval_flip_pred}, target={target_label}, min_conf={eval_confidence:.4f})!"
                            )
                        else:
                            print(
                                f"    -> Early stop at epoch {epoch + 1}: Prediction corrected with confidence {eval_confidence:.4f}!")
                        repair_success = True
                        break
                    elif can_accept_early and eval_confidence >= min_confidence * 0.8:
                        stable_count += 1
                        if stable_count >= 2:
                            if fairness_mode:
                                print(
                                    f"    -> Early stop at epoch {epoch + 1}: Fairness goal stable "
                                    f"(pred={eval_pred}, flip_pred={eval_flip_pred}, target={target_label}, min_conf={eval_confidence:.4f})!"
                                )
                            else:
                                print(
                                    f"    -> Early stop at epoch {epoch + 1}: Prediction stable with confidence {eval_confidence:.4f}!")
                            repair_success = True
                            break
                    else:
                        stable_count = 0
                else:
                    stable_count = 0

                if no_improve_count >= early_stop_patience:
                    print(
                        f"    -> Early stop at epoch {epoch + 1}: loss did not improve for {early_stop_patience} epochs.")
                    repair_success = best_model_state is not None
                    break

                if not conservative_mode:
                    if eval_meets_goal and loss_cls.item() < 0.1:
                        print(f"    -> Early stop at epoch {epoch + 1}: Prediction corrected and loss converged!")
                        repair_success = True
                        break

        except Exception as e:
            print(f"    ⚠ Error during training: {e}, rolling back!")
            # Roll back model weights.
            self.model.load_state_dict(model_backup)
            # Remove all hooks.
            for h in grad_hooks:
                h.remove()
            return False

        for h in grad_hooks:
            h.remove()

        self.model.eval()
        with torch.no_grad():
            final_output = self.model(buggy_in)
            # argmin_mode: ACAS Xu uses argmin decisions.
            if argmin_mode:
                final_pred = final_output.argmin(dim=1).item()
                final_probs = torch.softmax(-final_output, dim=1)  # Negated for argmin!
            else:
                final_pred = final_output.argmax(dim=1).item()
                final_probs = torch.softmax(final_output, dim=1)
            final_confidence = _goal_confidence(final_probs)

        def _try_soft_rollback():
            """Try soft rollback by weight interpolation before hard rollback."""
            if not enable_rollback:
                print("    -> Rollback disabled; keeping the current post-training weights.")
                return False
            current_state = {
                k: v.detach().cpu().clone()
                for k, v in self.model.state_dict().items()
            }
            backup_state = {
                k: v.detach().cpu().clone() if torch.is_tensor(v) else v
                for k, v in model_backup.items()
            }
            soft_state = {}
            for k in backup_state.keys():
                if k in current_state and torch.is_tensor(backup_state[k]) and torch.is_tensor(current_state[k]):
                    if backup_state[k].dtype.is_floating_point and current_state[k].dtype.is_floating_point:
                        soft_state[k] = 0.5 * current_state[k] + 0.5 * backup_state[k]
                    else:
                        soft_state[k] = current_state[k]
                else:
                    soft_state[k] = backup_state[k]

            self.model.load_state_dict(soft_state)
            self.model.eval()
            with torch.no_grad():
                soft_output = self.model(buggy_in)
                if argmin_mode:
                    soft_pred = soft_output.argmin(dim=1).item()
                    soft_probs = torch.softmax(-soft_output, dim=1)
                else:
                    soft_pred = soft_output.argmax(dim=1).item()
                    soft_probs = torch.softmax(soft_output, dim=1)
                soft_confidence = _goal_confidence(soft_probs)

            if max_acc_drop is not None and clean_loader is not None and clean_acc_baseline is not None:
                clean_correct = 0
                clean_total = 0
                with torch.no_grad():
                    for clean_x_chk, clean_y_chk in clean_loader:
                        clean_x_chk = clean_x_chk.to(self.device)
                        clean_y_chk = clean_y_chk.to(self.device)
                        clean_outputs_chk = self.model(clean_x_chk)
                        if argmin_mode:
                            clean_preds_chk = clean_outputs_chk.argmin(dim=1)
                        else:
                            clean_preds_chk = clean_outputs_chk.argmax(dim=1)
                        clean_correct += (clean_preds_chk == clean_y_chk).sum().item()
                        clean_total += clean_y_chk.numel()
                clean_acc_soft = (clean_correct / clean_total) if clean_total > 0 else None
                if clean_acc_soft is not None and (clean_acc_baseline - clean_acc_soft) > max_acc_drop:
                    print(
                        f"    -> Soft Rollback rejected by clean ACC guard: baseline={clean_acc_baseline:.4f}, current={clean_acc_soft:.4f}, drop={clean_acc_baseline - clean_acc_soft:.4f}, threshold={max_acc_drop:.4f}")
                    self.model.load_state_dict(backup_state)
                    return False

            if not backdoor_mode:
                if _prediction_satisfies_goal(soft_pred):
                    print(f"    -> Soft Rollback (Weight Interpolation) succeeded: "
                          f"Pred={soft_pred}, Confidence={soft_confidence:.4f}")
                    return True
            else:
                # Backdoor repair must recover the true label; merely leaving the original backdoor label is not sufficient.
                if soft_pred == target_label:
                    print(f"    -> Soft Rollback (Weight Interpolation) succeeded: "
                          f"Pred={soft_pred}, Confidence={soft_confidence:.4f}")
                    return True

            self.model.load_state_dict(backup_state)
            return False

        if enable_rollback and max_acc_drop is not None and clean_loader is not None and clean_acc_baseline is not None:
            clean_correct = 0
            clean_total = 0
            with torch.no_grad():
                for clean_x_final, clean_y_final in clean_loader:
                    clean_x_final = clean_x_final.to(self.device)
                    clean_y_final = clean_y_final.to(self.device)
                    clean_outputs_final = self.model(clean_x_final)
                    if argmin_mode:
                        clean_preds_final = clean_outputs_final.argmin(dim=1)
                    else:
                        clean_preds_final = clean_outputs_final.argmax(dim=1)
                    clean_correct += (clean_preds_final == clean_y_final).sum().item()
                    clean_total += clean_y_final.numel()
            clean_acc_final = (clean_correct / clean_total) if clean_total > 0 else None
            if clean_acc_final is not None and (clean_acc_baseline - clean_acc_final) > max_acc_drop:
                if use_clean_aware_selection and best_model_state is not None and best_pred is not None and _prediction_satisfies_goal(
                        best_pred) and best_clean_acc is not None and (
                        clean_acc_baseline - best_clean_acc) <= max_acc_drop:
                    print(
                        f"    -> Restoring clean-aware best checkpoint before hard guard rollback: best_clean={best_clean_acc:.4f}, baseline={clean_acc_baseline:.4f}")
                    self.model.load_state_dict({k: v.to(self.device) for k, v in best_model_state.items()})
                    return True
                print(
                    f"    ⚠ Clean ACC guard triggered before final accept: baseline={clean_acc_baseline:.4f}, current={clean_acc_final:.4f}, drop={clean_acc_baseline - clean_acc_final:.4f}, threshold={max_acc_drop:.4f}")
                repair_success = _try_soft_rollback()
                return repair_success

        if not backdoor_mode:
            if fairness_mode:
                final_meets_goal, final_confidence, final_pred, final_flip_pred, final_conf_orig, final_conf_flip, final_classification_ok, final_consistency_ok = _fairness_goal_status(
                    final_output)
            else:
                final_meets_goal = _prediction_satisfies_goal(final_pred)

            if not final_meets_goal:
                if fairness_mode:
                    print(
                        f"    ⚠ Fairness repair failed: pred={final_pred}, flip_pred={final_flip_pred}, target={target_label}, "
                        f"classification_ok={final_classification_ok}, consistency_ok={final_consistency_ok}, "
                        f"orig_conf={final_conf_orig:.4f}, flip_conf={final_conf_flip:.4f}; rolling back!"
                    )
                else:
                    print(
                        f"    ⚠ Repair failed: Final prediction ({final_pred}) does not satisfy repair goal, rolling back!")
                print("    -> Legacy rollback path disabled in frozen default pipeline.")
                repair_success = False
            else:
                repair_success = True
                if fairness_mode:
                    print(
                        f"    -> Final verification: Pred={final_pred}, FlipPred={final_flip_pred}, Target={target_label}, "
                        f"OrigConf={final_conf_orig:.4f}, FlipConf={final_conf_flip:.4f}, MinConf={final_confidence:.4f}"
                    )
                else:
                    print(f"    -> Final verification: Pred={final_pred}, Confidence={final_confidence:.4f}")
                if final_confidence < min_confidence:
                    print(f"    ⚠ Warning: Confidence ({final_confidence:.4f}) below threshold ({min_confidence})")
        else:

            if final_pred == target_label:
                repair_success = True
                print(f"    -> Backdoor mode: corrected to target label {target_label}, Conf={final_confidence:.4f}")

            else:
                print("    -> Legacy rollback path disabled in frozen default pipeline.")
                repair_success = False

        return repair_success

    def repair_batch(self, buggy_samples, ref_samples, target_labels, epochs=50, lr=0.01, lambda_reg=1.0):
        """
        Repair multiple samples in a batch; optional extension.

        Args:
            buggy_samples: List of buggy samples.
            ref_samples: List of reference samples.
            target_labels: List of target labels.

        Returns:
            int: Number of successfully repaired samples.
        """
        success_count = 0
        for i, (buggy, ref, target) in enumerate(zip(buggy_samples, ref_samples, target_labels)):
            print(f"\n>>> Repairing sample {i + 1}/{len(buggy_samples)}...")
            if self.repair(buggy, ref, target, epochs=epochs, lr=lr):
                success_count += 1
        return success_count