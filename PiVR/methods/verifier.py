import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CausalVerifier:
    def __init__(self, model, train_loader, layers_structure=None, device="cpu",
                 task_type=None, dataset_name=None, sensitive_indices=None,
                 flip_fn=None, lambda_ref_fair=0.5, reference_metric: str = "feature_cosine",
                 lambda_ref_usefulness: float = 0.75, ref_top_k: int = 5):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.task_type = task_type
        self.dataset_name = dataset_name
        self.sensitive_indices = sensitive_indices if sensitive_indices is not None else []
        self.flip_fn = flip_fn
        self.lambda_ref_fair = lambda_ref_fair
        self.lambda_ref_usefulness = lambda_ref_usefulness
        self.reference_metric = reference_metric
        self.ref_top_k = int(ref_top_k)
        self.layers_structure = getattr(model, 'layers_structure', layers_structure)
        if self.layers_structure is None:
            raise ValueError("layers_structure must be provided either as model attribute or constructor argument")
        self.cached_train_data = [(data.cpu(), target.cpu()) for data, target in train_loader]

    def select_reference_sample(self, buggy_sample, target_class, argmin_mode: bool = False, safe_labels=None):
        is_fairness_task = self.task_type == "fairness" or (self.dataset_name is not None and len(self.sensitive_indices) > 0)
        if is_fairness_task and len(self.sensitive_indices) > 0:
            ref_sample = self.find_fair_reference_sample(buggy_sample, target_class, self.sensitive_indices, self.dataset_name, self.flip_fn, self.lambda_ref_fair)
            if ref_sample is not None:
                return ref_sample
        if safe_labels is not None and len(safe_labels) > 0:
            return self.find_nearest_safe_set_sample(buggy_sample, safe_labels, argmin_mode)
        return self.find_nearest_positive_sample(buggy_sample, target_class, argmin_mode)

    def _fairness_discrepancy(self, sample_tensor, flipped_tensor):
        self.model.eval()
        with torch.no_grad():
            logits_orig = self.model(sample_tensor)
            logits_flip = self.model(flipped_tensor)
        return torch.norm(logits_orig - logits_flip, p=2).item()

    def _fairness_reference_usefulness(self, buggy_tensor, candidate_tensor, buggy_flipped_tensor):
        """Measure how much a candidate reduces counterfactual discrepancy.

        Positive values mean the candidate is more useful than the original buggy
        sample as a counterfactual anchor.
        """
        self.model.eval()
        with torch.no_grad():
            logits_buggy = self.model(buggy_tensor)
            logits_buggy_flip = self.model(buggy_flipped_tensor)
            discrepancy_orig = torch.norm(logits_buggy - logits_buggy_flip, p=2).item()

            ref_logits = self.model(candidate_tensor)
            # Compare candidate against the buggy counterfactual, not candidate against itself.
            discrepancy_ref = torch.norm(ref_logits - logits_buggy_flip, p=2).item()

        return discrepancy_orig - discrepancy_ref

    def find_fair_reference_sample(self, buggy_sample, target_class, sensitive_indices,
                                   dataset_name=None, flip_fn=None, lambda_ref_fair=0.5):
        top_k = self.ref_top_k
        candidate_pool = []

        if isinstance(buggy_sample, torch.Tensor):
            buggy_np = buggy_sample.detach().cpu().numpy().flatten()
        else:
            buggy_np = np.asarray(buggy_sample).flatten()
        buggy_tensor = torch.tensor(buggy_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        buggy_flip_tensor = None
        if flip_fn is not None:
            buggy_flip_np = flip_fn(buggy_np.copy(), sensitive_indices, dataset_name=dataset_name or "fairness", use_negation=True)
            buggy_flip_tensor = torch.tensor(buggy_flip_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        non_sensitive_mask = np.ones(len(buggy_np), dtype=bool)
        for idx in sensitive_indices:
            if idx < len(non_sensitive_mask):
                non_sensitive_mask[idx] = False

        self.model.eval()
        with torch.no_grad():
            for batch_data, batch_target in self.cached_train_data:
                mask = (batch_target == target_class)
                if mask.sum() == 0:
                    continue
                candidates = batch_data[mask]
                preds = torch.argmax(self.model(candidates.to(self.device)), dim=1).cpu()
                candidates = candidates[preds == target_class]
                for candidate in candidates:
                    candidate_np = candidate.detach().cpu().numpy().flatten()
                    vec_buggy = buggy_np[non_sensitive_mask]
                    vec_cand = candidate_np[non_sensitive_mask]
                    cos_sim = np.dot(vec_buggy, vec_cand) / (np.linalg.norm(vec_buggy) * np.linalg.norm(vec_cand) + 1e-8)
                    base_score = 1.0 - cos_sim
                    if flip_fn is not None:
                        cand_flip_np = flip_fn(candidate_np.copy(), sensitive_indices, dataset_name=dataset_name or "fairness", use_negation=True)
                        cand_flip = torch.tensor(cand_flip_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                        logits_cand = self.model(candidate.unsqueeze(0).to(self.device))
                        logits_flip = self.model(cand_flip)
                        penalty = torch.norm(logits_cand - logits_flip, p=2).item()
                        if torch.argmax(logits_cand, dim=1).item() != torch.argmax(logits_flip, dim=1).item():
                            penalty += 1.0
                        base_score += lambda_ref_fair * penalty
                    usefulness = 0.0
                    if buggy_flip_tensor is not None:
                        usefulness = self._fairness_reference_usefulness(buggy_tensor, candidate.unsqueeze(0).to(self.device), buggy_flip_tensor)
                    final_score = base_score - self.lambda_ref_usefulness * float(np.clip(usefulness, -2.0, 2.0))
                    candidate_pool.append((final_score, candidate.clone()))

        if not candidate_pool:
            return None
        candidate_pool.sort(key=lambda x: x[0])
        top_samples = [sample for _, sample in candidate_pool[:top_k]]
        return torch.stack(top_samples, dim=0).to(self.device)

        reranked_pool.sort(key=lambda x: x[0])
        top_samples = [sample for _, _, _, sample in reranked_pool[:top_k]]
        best_final_score, best_base_score, best_usefulness, _ = reranked_pool[0]
        print(
            f">>> [CausalVerifier] Fairness-aware selection: found reference batch from {candidate_count} candidates, "
            f"best_final={best_final_score:.4f}, base={best_base_score:.4f}, usefulness={best_usefulness:.4f}, top_k={len(top_samples)}"
        )
        return torch.stack(top_samples, dim=0).to(self.device)

    def _extract_reference_features(self, sample):
        """
        Extract feature representations for reference selection.

        By default, use the output of the last Conv2d/Linear layer:
        - Conv2d -> apply GAP to obtain [B, C].
        - Linear -> use the direct [B, D] output.
        If feature extraction fails, fall back to the flattened input.
        """
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample)
        sample = sample.to(self.device)
        if sample.dim() == 1 or sample.dim() == 3:
            sample_input = sample.unsqueeze(0)
        else:
            sample_input = sample

        features = {"value": None}
        target_layer = None
        for layer in self.layers_structure:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                target_layer = layer
        if target_layer is None:
            return sample_input.view(sample_input.shape[0], -1).detach().cpu()

        def hook(_m, _i, o):
            if isinstance(target_layer, nn.Conv2d):
                features["value"] = F.adaptive_avg_pool2d(o.detach(), (1, 1)).flatten(1)
            else:
                features["value"] = o.detach().flatten(1)

        handle = target_layer.register_forward_hook(hook)
        self.model.eval()
        with torch.no_grad():
            self.model(sample_input)
        handle.remove()

        if features["value"] is None:
            return sample_input.view(sample_input.shape[0], -1).detach().cpu()
        return features["value"].detach().cpu()

    def _pair_distance(self, source_vec: torch.Tensor, candidate_vecs: torch.Tensor) -> torch.Tensor:
        """Compute distances between the source vector and the candidate batch according to reference_metric."""
        if self.reference_metric == "input_l2" or self.reference_metric == "feature_l2":
            return torch.norm(candidate_vecs - source_vec.unsqueeze(0), dim=1)

        # cosine distance by default
        source_norm = F.normalize(source_vec.unsqueeze(0), p=2, dim=1)
        candidate_norm = F.normalize(candidate_vecs, p=2, dim=1)
        cosine_sim = torch.sum(candidate_norm * source_norm, dim=1)
        return 1.0 - cosine_sim

    def find_nearest_safe_set_sample(self, buggy_sample, safe_labels, argmin_mode: bool = False):
        top_k = 5
        candidate_pool = []
        safe_label_set = {int(lbl) for lbl in safe_labels}

        if self.reference_metric.startswith("feature_"):
            buggy_vec = self._extract_reference_features(buggy_sample).view(1, -1).squeeze(0)
        else:
            if not isinstance(buggy_sample, torch.Tensor):
                buggy_sample = torch.tensor(buggy_sample)
            buggy_vec = buggy_sample.view(-1).detach().cpu()

        self.model.eval()
        with torch.no_grad():
            for batch_data, batch_target in self.cached_train_data:
                batch_target_np = batch_target.view(-1).cpu().numpy()
                mask = torch.tensor([int(t) in safe_label_set for t in batch_target_np], dtype=torch.bool)
                if mask.sum() == 0:
                    continue

                candidates = batch_data[mask]
                candidate_logits = self.model(candidates.to(self.device))
                if argmin_mode:
                    candidate_preds = torch.argmin(candidate_logits, dim=1).cpu()
                else:
                    candidate_preds = torch.argmax(candidate_logits, dim=1).cpu()

                correct_mask = torch.tensor([int(pred) in safe_label_set for pred in candidate_preds], dtype=torch.bool)
                if correct_mask.sum() == 0:
                    continue

                candidates = candidates[correct_mask]
                if self.reference_metric.startswith("feature_"):
                    candidate_vecs = self._extract_reference_features(candidates)
                else:
                    candidate_vecs = candidates.view(candidates.shape[0], -1).detach().cpu()

                dists = self._pair_distance(buggy_vec, candidate_vecs)
                for i in range(candidates.shape[0]):
                    candidate_pool.append((dists[i].item(), candidates[i].clone()))

        if len(candidate_pool) == 0:
            return None

        candidate_pool.sort(key=lambda item: item[0])
        top_samples = [sample for _, sample in candidate_pool[:top_k]]
        return torch.stack(top_samples, dim=0).to(self.device)

    def find_nearest_positive_sample(self, buggy_sample, target_class,
                                       argmin_mode: bool = False):
        """
        Step A: Abduction - find counterfactual reference samples.

        Search the training set for the Top-5 samples that are most similar to
        buggy_sample and are correctly predicted as target_class. Return a batch
        tensor of shape [K, ...] as candidate latent prototypes.

        Args:
            argmin_mode: If True, use argmin for prediction check (ACAS Xu safety tasks).
                         If False, use argmax (classification/fairness tasks).
        """
        top_k = 5
        candidate_pool = []

        if self.reference_metric.startswith("feature_"):
            buggy_vec = self._extract_reference_features(buggy_sample).view(1, -1).squeeze(0)
        else:
            if not isinstance(buggy_sample, torch.Tensor):
                buggy_sample = torch.tensor(buggy_sample)
            buggy_vec = buggy_sample.view(-1).detach().cpu()

        self.model.eval()
        with torch.no_grad():
            for batch_data, batch_target in self.cached_train_data:
                mask = (batch_target == target_class)
                if mask.sum() == 0:
                    continue

                candidates = batch_data[mask]

                candidate_logits = self.model(candidates.to(self.device))
                if argmin_mode:
                    candidate_preds = torch.argmin(candidate_logits, dim=1).cpu()
                else:
                    candidate_preds = torch.argmax(candidate_logits, dim=1).cpu()

                correct_mask = (candidate_preds == target_class)
                if correct_mask.sum() == 0:
                    continue

                candidates = candidates[correct_mask]

                if self.reference_metric.startswith("feature_"):
                    candidate_vecs = self._extract_reference_features(candidates)
                else:
                    candidate_vecs = candidates.view(candidates.shape[0], -1).detach().cpu()

                dists = self._pair_distance(buggy_vec, candidate_vecs)

                for i in range(candidates.shape[0]):
                    candidate_pool.append((dists[i].item(), candidates[i].clone()))

        if len(candidate_pool) == 0:
            return None

        candidate_pool.sort(key=lambda item: item[0])
        top_samples = [sample for _, sample in candidate_pool[:top_k]]
        return torch.stack(top_samples, dim=0).to(self.device)

    def get_layer_activations(self, sample, layer_indices):
        """
        Helper function for obtaining activations at selected layers.

        Supports both single samples `[D] / [C, H, W]` and batches
        `[B, D] / [B, C, H, W]`.
        """
        activations = {}
        hooks = []

        def get_hook(layer_idx):
            def hook(module, input, output):
                activations[layer_idx] = output.detach().clone()
            return hook

        tracked_count = 0
        for layer in self.layers_structure:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if tracked_count in layer_indices:
                    h = layer.register_forward_hook(get_hook(tracked_count))
                    hooks.append(h)
                tracked_count += 1

        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample)
        sample = sample.to(self.device)
        if sample.dim() == 1 or sample.dim() == 3:
            sample_input = sample.unsqueeze(0)
        else:
            sample_input = sample

        # Forward pass.
        self.model.eval()
        with torch.no_grad():
            self.model(sample_input)

        for h in hooks:
            h.remove()
        return activations

    def calculate_pathway_sce(self, buggy_sample, ref_sample, pathway_masks,
                              target_class, argmin_mode: bool = False,
                              fairness_mode: bool = False,
                              fairness_flip_fn=None,
                              fairness_sensitive_indices=None,
                              fairness_dataset_name: str = None,
                              safe_labels=None):
        """
        Step B: Intervention and verification, following the CCBR-style idea.

        Force the values of buggy_sample at pathway_masks positions to be
        replaced by the latent prototype activations from ref_sample.

        Args:
            pathway_masks: List[Tensor], layer masks generated by the previous step.
            target_class: Target class used for target-aware SCE.
            argmin_mode: If True, evaluate the post-intervention prediction with
                         argmin, which is suitable for ACAS Xu safety tasks.
                         Otherwise use argmax for classification/fairness tasks.
            fairness_mode: If True, compute fairness-aware SCE by measuring whether
                           counterfactual discrepancy decreases after intervention,
                           instead of measuring target-class gain.
            fairness_flip_fn: Function for flipping sensitive attributes in fairness tasks.
            fairness_sensitive_indices: List of sensitive attribute indices.
            fairness_dataset_name: Fairness dataset name.
        """
        ref_activations_batch = self.get_layer_activations(ref_sample, range(len(pathway_masks)))
        ref_activations = {
            layer_idx: act.mean(dim=0, keepdim=True)
            for layer_idx, act in ref_activations_batch.items()
        }

        hooks = []
        normalized_masks = [
            torch.from_numpy(mask).float().to(self.device)
            if isinstance(mask, np.ndarray) else mask.float().to(self.device)
            for mask in pathway_masks
        ]

        def get_intervention_hook(layer_idx, mask, ref_val):
            def hook(module, input, output):
                # Linear: output [B, N], mask [N], ref_val [1, N]
                # Conv2d: output [B, C, H, W], mask [C], ref_val [1, C, H, W]
                if isinstance(module, nn.Conv2d):
                    mask_bc = mask.to(output.device).view(1, -1, 1, 1)
                else:
                    mask_bc = mask.to(output.device).unsqueeze(0)
                ref_val_bc = ref_val.to(output.device)
                modified_output = output * (1 - mask_bc) + ref_val_bc * mask_bc
                return modified_output
            return hook

        tracked_count = 0
        for layer in self.layers_structure:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                if tracked_count < len(normalized_masks):
                    mask = normalized_masks[tracked_count]
                    ref_val = ref_activations[tracked_count]
                    h = layer.register_forward_hook(
                        get_intervention_hook(tracked_count, mask, ref_val)
                    )
                    hooks.append(h)
                tracked_count += 1

        if not isinstance(buggy_sample, torch.Tensor):
            buggy_sample = torch.tensor(buggy_sample)
        buggy_sample = buggy_sample.to(self.device)
        if buggy_sample.dim() == 1 or buggy_sample.dim() == 3:
            buggy_input = buggy_sample.unsqueeze(0)
        else:
            buggy_input = buggy_sample

        self.model.eval()
        with torch.no_grad():
            logits_do = self.model(buggy_input)
            if argmin_mode:
                prob_do = torch.softmax(-logits_do, dim=1)
            else:
                prob_do = torch.softmax(logits_do, dim=1)

        for h in hooks:
            h.remove()

        # 4. Original prediction.
        with torch.no_grad():
            logits_orig = self.model(buggy_input)
            if argmin_mode:
                prob_orig = torch.softmax(-logits_orig, dim=1)
            else:
                prob_orig = torch.softmax(logits_orig, dim=1)

        safe_label_set = None
        if safe_labels is not None:
            safe_label_set = sorted({int(lbl) for lbl in safe_labels})

        if fairness_mode:
            if fairness_flip_fn is None or fairness_sensitive_indices is None:
                raise ValueError("fairness_mode=True requires fairness_flip_fn and fairness_sensitive_indices")

            buggy_np = buggy_input[0].detach().cpu().numpy().copy()
            buggy_flip_np = fairness_flip_fn(
                buggy_np,
                fairness_sensitive_indices,
                dataset_name=fairness_dataset_name or "fairness",
                use_negation=True,
            )
            buggy_flip = torch.tensor(buggy_flip_np, dtype=buggy_input.dtype, device=self.device).unsqueeze(0)

            with torch.no_grad():
                logits_flip_orig = self.model(buggy_flip)

            fairness_hooks = []
            tracked_count = 0
            for layer in self.layers_structure:
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    if tracked_count < len(normalized_masks):
                        mask = normalized_masks[tracked_count]
                        ref_val = ref_activations[tracked_count]
                        h = layer.register_forward_hook(
                            get_intervention_hook(tracked_count, mask, ref_val)
                        )
                        fairness_hooks.append(h)
                    tracked_count += 1

            with torch.no_grad():
                logits_flip_do = self.model(buggy_flip)

            for h in fairness_hooks:
                h.remove()

            discrepancy_orig = torch.norm(logits_orig - logits_flip_orig, p=2).item()
            discrepancy_do = torch.norm(logits_do - logits_flip_do, p=2).item()
            sce_score = discrepancy_orig - discrepancy_do
            intervened_pred = torch.argmax(logits_do).item()
            return sce_score, intervened_pred

        if safe_label_set is not None and len(safe_label_set) > 0:
            sce_score = max(prob_do[0, lbl].item() for lbl in safe_label_set) - max(prob_orig[0, lbl].item() for lbl in safe_label_set)
        else:
            sce_score = prob_do[0, target_class].item() - prob_orig[0, target_class].item()

        if argmin_mode:
            intervened_pred = torch.argmin(logits_do).item()
        else:
            intervened_pred = torch.argmax(logits_do).item()

        return sce_score, intervened_pred
