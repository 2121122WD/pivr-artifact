"""
Core principles:
1. LRP relevance propagation: compute each neuron's contribution to the output.
2. Critical neuron extraction: find the smallest neuron set whose accumulated
   LRP relevance is greater than or equal to alpha * g_f(x).
3. Hit Spectrum construction: count critical-neuron hits on Pass/Fail samples.
4. SBFL suspiciousness scoring: compute each neuron's suspiciousness based on
   the Hit Spectrum.
5. Top-k mask generation: select the Top-k suspicious neurons in each layer and
   generate binary masks.

Notes:
- This implementation does not explicitly track path tuples to avoid
  combinatorial explosion.
- I_P and I_F are computed using the mathematically correct definitions.
- Adaptive Connectivity Pruning post-processing is no longer included.
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from PiVR.utils import (
    myLRPModel,
    min_subarray_with_sum_gt_target,
    get_BARINEL_score,
)


class PathwayDeepCP:
    """
    Pathway localization module used by PiVR.

    Compared with the original NP-SBFL implementation, this version adds two
    task-oriented enhancements:
    1. Layer-Adaptive Critical Neuron Extraction:
       Instead of forcing all layers to share a global relevance budget, each
       layer can use its own positive-relevance coverage budget. This mitigates
       scale mismatch across layers.
    2. Task-Aware Hit Spectrum Construction:
       Pass/Fail labels are not limited to whether the prediction equals the
       ground-truth label. They can also be defined by an external violation_fn
       for safety, backdoor, fairness, or other task-specific semantics, making
       localization better aligned with the downstream repair objective.
    """

    def __init__(self, model_name, model, layers_structure, alpha=0.7, beta=0.0,
                 activation_threshold=0., input_size=(1, 28, 28), train_loader=None,
                 path_to_save_pickles="pickles", device="cpu",
                 argmin_mode=False, k=None, layer_k_ratio_cap=0.5,
                 critical_budget_mode="coverage",
                 layer_budget_min_ratio=0.05,
                 task_type="classification",
                 violation_fn=None,
                 failure_score_fn=None,
                 failure_score_power=1.0,
                 conv_feature_mode="channel",
                 conv_spatial_top_ratio=0.1,
                 conv_spatial_min_k=1,
                 min_layer_k=1,
                 max_layer_k=None,
                 small_layer_full_preserve=3):

        self.model_name = model_name
        self.input_size = input_size
        self.alpha = alpha
        self.beta = beta  # deprecated compatibility field; unused in default pipeline
        self.activation_threshold = 0.0
        self.k = k
        self.layer_k_ratio_cap = layer_k_ratio_cap
        self.argmin_mode = argmin_mode
        self.critical_budget_mode = "coverage"

        self.path_to_save_pickles = path_to_save_pickles
        self.train_loader = train_loader
        self.cuda = True if device == "cuda" else False
        self.device = device
        self.layer_budget_min_ratio = layer_budget_min_ratio
        self.task_type = task_type
        self.violation_fn = violation_fn
        self.failure_score_fn = failure_score_fn
        self.failure_score_power = failure_score_power
        self.conv_feature_mode = conv_feature_mode
        self.conv_spatial_top_ratio = conv_spatial_top_ratio
        self.conv_spatial_min_k = conv_spatial_min_k
        self.min_layer_k = min_layer_k
        self.max_layer_k = max_layer_k
        self.small_layer_full_preserve = small_layer_full_preserve

        self.model = model
        self.layers_structure = layers_structure
        self.lrp_model = myLRPModel(model, layers_structure)

        # Cache Hit Spectrums to avoid redundant computation.
        self._cached_hit_spectrums = None

        self.has_conv = any(isinstance(l, nn.Conv2d) for l in layers_structure)
        if self.has_conv:
            self._lrp_filter_types = ["RelevancePropagationConv2d", "RelevancePropagationLinear"]
        else:
            self._lrp_filter_types = ["RelevancePropagationLinear"]
        print(f"[PathwayDeepCP] has_conv={self.has_conv}, SBFL filter: {self._lrp_filter_types}")

        # Infer layer shapes.
        model_device = next(self.model.parameters()).device if any(True for _ in self.model.parameters()) else torch.device("cuda" if self.cuda else "cpu")
        _, activations, _, _ = self.get_relevancy_and_activations(
            torch.randn((1, *self.input_size), device=model_device)
        )
        self.layer_shapes = [a.shape[1] for a in activations]
        print(f"[PathwayDeepCP] Layer shapes: {self.layer_shapes}")

    def _process_conv_tensor(self, tensor, use_relu=True):
        if use_relu:
            tensor = F.relu(tensor)
        if tensor.dim() != 4:
            return tensor.flatten(1)

        bsz, channels, height, width = tensor.shape
        flat = tensor.reshape(bsz, channels, height * width)
        # Keep the default behavior closer to the original spatial-top-k semantics
        top_k = max(1, int(math.ceil(height * width * 0.1)))
        top_k = min(top_k, height * width)
        values, _ = torch.topk(flat, k=top_k, dim=2)
        return values.mean(dim=2)

    def _sanitize_layer_budget(self, layer_size, requested_k):
        layer_size = int(layer_size)
        if layer_size <= 0:
            return 0
        final_k = min(int(requested_k), layer_size)
        ratio_cap = max(0.0, min(1.0, float(self.layer_k_ratio_cap)))
        if ratio_cap > 0:
            final_k = min(final_k, max(1, int(math.ceil(layer_size * ratio_cap))))
        return max(1, final_k)

    def _fairness_prediction_info(self, x_in, model):
        model.eval()
        with torch.no_grad():
            logits = model(x_in)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            confidence = probs[0, pred].item()
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            margin = (sorted_probs[0, 0] - sorted_probs[0, 1]).item() if probs.shape[1] > 1 else confidence
        return pred, confidence, margin

    def get_relevancy_and_activations(self, data):
        """
        Get LRP relevancies and activations.

        Args:
            data: Input tensor with shape (1, *input_size). The batch size must
                be 1.

        Returns:
            relevancy: list[tensor], LRP relevance values for each layer.
            activations: list[tensor], activation values for each layer.
            g_fx: float, total relevance, computed as the output-layer
                relevance sum.
            predicted_class: int, predicted class.
        """
        try:
            model_device = next(self.model.parameters()).device
            if hasattr(data, "to"):
                data = data.to(model_device)
        except StopIteration:
            pass

        # Ensure that batch_size == 1.
        if data.shape[0] != 1:
            raise ValueError(f"batch_size must be 1, got {data.shape[0]}")

        relevancies, activations = self.lrp_model.forward(data)

        activations = activations[:-1]

        filter_types = self._lrp_filter_types

        def process_relevancy(item):
            name, rel = item
            if name == "RelevancePropagationConv2d":
                return self._process_conv_tensor(rel, use_relu=False)
            else:
                return rel.flatten(1)
        relevancy = list(filter(lambda item: item[0] in filter_types, relevancies))
        relevancy = list(map(process_relevancy, relevancy))

        activations = list(filter(lambda item: item[0] in filter_types, activations))
        # Conv2d activations keep spatial hot-spot statistics; Linear activations are flattened directly.
        def process_activation(item):
            name, act = item
            if name == "RelevancePropagationConv2d":
                return self._process_conv_tensor(act, use_relu=True)
            else:
                return F.relu(act).flatten(1) if name == "RelevancePropagationLinear" else act.flatten(1)
        activations = list(map(process_activation, activations))

        # Total relevance from the output layer.
        g_fx = torch.sum(relevancies[0][1]).item()

        # Predicted class, supporting both argmin and argmax modes.
        if self.argmin_mode:
            predicted_class = torch.argmin(relevancies[-1][1], dim=1).item()
        else:
            predicted_class = torch.argmax(relevancies[-1][1], dim=1).item()

        return relevancy, activations, g_fx, predicted_class

    def _get_layer_budget(self, layer_relevancy, global_relevance_sum):
        """Coverage-based criticality budget used in the default pipeline."""
        positive_sum = torch.clamp(layer_relevancy, min=0).sum().item()
        return float(self.alpha * positive_sum)

    def _is_fail_sample(self, data, target, predicted_class):
        """
        Determine whether a sample is a Fail/Violation sample.

        Default behavior:
        - classification/backdoor/safety: compare the prediction with target.
        - fairness: if violation_fn is provided, use external task semantics,
          typically consistency between the original sample and its
          protected-attribute-flipped counterpart.
        """
        if self.violation_fn is not None:
            return bool(self.violation_fn(
                data=data,
                target=target,
                predicted_class=predicted_class,
                model=self.model,
                argmin_mode=self.argmin_mode,
            ))
        return predicted_class != target.item()

    def _get_failure_weight(self, data, target, predicted_class, is_fail):
        """
        Get the sample-level failure weight.

        This weight is used to construct a task-aware or severity-aware Hit
        Spectrum.
        """
        if not is_fail:
            return 1.0
        if self.failure_score_fn is None:
            return 1.0
        raw_score = float(self.failure_score_fn(
            data=data,
            target=target,
            predicted_class=predicted_class,
            model=self.model,
            argmin_mode=self.argmin_mode,
        ))
        raw_score = max(0.0, raw_score)
        if self.failure_score_power != 1.0:
            raw_score = raw_score ** float(self.failure_score_power)
        return max(1e-8, raw_score)

    def generate_cdp_representation(self, data):
        """
        Generate the CDP representation, i.e., the critical-neuron set, for a
        single sample.

        Args:
            data: Input tensor with shape (1, *input_size).

        Returns:
            cdp_representation: 0/1 vector with shape (sum(layer_shapes),).
            critical_neurons_layers: list[tensor], critical-neuron indices for
                each layer.
            predicted_class: int, predicted class.
            activation_mask: 0/1 vector with shape (sum(layer_shapes),).
        """
        relevancy, activations, g_fx, predicted_class = self.get_relevancy_and_activations(data)

        # Step 2a: Critical neuron extraction.
        # This version supports a layer-adaptive relevance budget
        # instead of the global alpha * g_f(x) budget used by NP-SBFL.
        critical_neurons_layers = []
        for i in range(1, len(relevancy)):
            layer_budget = self._get_layer_budget(relevancy[i][0], g_fx)
            critical_neurons_layer = min_subarray_with_sum_gt_target(relevancy[i][0], layer_budget)
            if critical_neurons_layer.shape[0] == 0:
                return None, None, None, None
            critical_neurons_layers.append(critical_neurons_layer)

        # Generate the CDP representation: critical-neuron positions are 1 and all others are 0.
        cdp_representation = np.zeros(sum(self.layer_shapes))
        indices = np.cumsum([0] + self.layer_shapes)
        for i in range(len(critical_neurons_layers)):
            layer_size = self.layer_shapes[i]
            raw_indices = critical_neurons_layers[i].cpu().numpy()
            # Boundary protection: discard neuron indices outside the current layer size.
            valid_indices = raw_indices[raw_indices < layer_size]
            if len(valid_indices) == 0 and len(raw_indices) > 0:
                # If all indices are out of range, map them back to the valid range by modulo.
                valid_indices = raw_indices % layer_size
            cdp_representation[indices[i] + valid_indices] = 1

        # Generate the activation mask. By default, activation > 0 is treated as a hit.
        activations_vector = np.zeros(sum(self.layer_shapes))
        indices = np.cumsum([0] + self.layer_shapes)
        for i in range(indices.shape[0] - 1):
            activations_vector[indices[i] : indices[i + 1]] = activations[i].cpu().numpy().reshape(-1)

        activation_mask = np.zeros(sum(self.layer_shapes))
        indices = np.cumsum([0] + self.layer_shapes)
        for i in range(indices.shape[0] - 1):
            sl = slice(indices[i], indices[i + 1])
            activation_mask[sl] = np.where(activations_vector[sl] > 0.0, 1, 0)
            if sum(activation_mask[sl]) == 0:
                activation_mask[sl][np.argmax(activations_vector[sl])] = 1

        return cdp_representation, critical_neurons_layers, predicted_class, activation_mask

    def calculate_neuron_hit_spectrums(self):
        """
        Step 2b: Count the Hit Spectrum tuple (A_P, A_F, I_P, I_F) for critical
        neurons.

        Mathematically correct implementation:
        - A_P: number of times critical neurons are activated on Pass samples.
        - A_F: number of times critical neurons are activated on Fail samples.
        - I_P: number of times critical neurons are inactive on Pass samples.
        - I_F: number of times critical neurons are inactive on Fail samples.

        Returns:
            dict: {"A_P": np.array, "A_F": np.array, "I_P": np.array, "I_F": np.array}
        """
        # Return cached results if available.
        if self._cached_hit_spectrums is not None:
            print("[PathwayDeepCP] Using cached Hit Spectrums (avoiding recomputation)")
            return self._cached_hit_spectrums

        total_neurons = sum(self.layer_shapes)

        # Initialize counters.
        A_P = np.zeros(total_neurons)
        A_F = np.zeros(total_neurons)
        total_passed = 0.0
        total_failed = 0.0

        print("[PathwayDeepCP] Computing Hit Spectrums...")
        for data, target in tqdm.tqdm(self.train_loader, desc="Hit Spectrum"):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()

            data, target = Variable(data), Variable(target)

            # Generate the CDP representation.
            cdp_representation, _, predicted_class, activation_mask = self.generate_cdp_representation(data)
            if cdp_representation is None:
                continue

            # Hit mask: critical-neuron mask AND activation mask.
            hit_mask = activation_mask * cdp_representation

            # Task-aware pass/fail + severity-aware weighting
            is_fail = self._is_fail_sample(data, target, predicted_class)
            sample_weight = self._get_failure_weight(data, target, predicted_class, is_fail)

            if is_fail:
                A_F += sample_weight * hit_mask
                total_failed += sample_weight
            else:
                A_P += hit_mask
                total_passed += 1.0

        # Compute I_P and I_F using the mathematically correct definitions.
        I_P = total_passed - A_P
        I_F = total_failed - A_F

        # Cache the results.
        self._cached_hit_spectrums = {
            "A_P": A_P,
            "A_F": A_F,
            "I_P": I_P,
            "I_F": I_F,
        }

        return self._cached_hit_spectrums

    def get_scores_from_spectrums(self, neuron_hit_spectrums, sfl_strategy):
        """
        Step 2c: Compute SBFL suspiciousness based on the Hit Spectrum.

        Args:
            neuron_hit_spectrums: dict containing A_P, A_F, I_P, and I_F.
            sfl_strategy: legacy argument retained for compatibility; Barinel
                is used by default.

        Returns:
            np.array: suspiciousness score for each neuron.
        """
        return get_BARINEL_score(neuron_hit_spectrums)

    def get_layerwise_suspiciousness_scores(self, scores_vector):
        """
        Split the global suspiciousness vector into layer-wise suspiciousness
        lists.

        Args:
            scores_vector: suspiciousness vector with shape
                (sum(layer_shapes),).

        Returns:
            list[list[(neuron_idx, score)]]: one list for each layer.
        """
        layerwise_scores = []
        indices = np.cumsum([0] + self.layer_shapes)
        for i in range(indices.shape[0] - 1):
            layer_scores = scores_vector[indices[i] : indices[i + 1]]
            layer_scores = list(zip(range(layer_scores.shape[0]), layer_scores))
            layerwise_scores.append(layer_scores)
        return layerwise_scores

    def get_suspicious_neurons_per_layer(self, sfl_strategy, k):
        """
        Step 2d: Select the Top-k suspicious neurons from each layer.

        Args:
            sfl_strategy: legacy argument retained for compatibility; Barinel
                is used by default.
            k: number of Top-k neurons selected per layer.

        Returns:
            list[list[(neuron_idx, score)]]: Top-k neurons and their scores for
                each layer.
        """
        neuron_hit_spectrums = self.calculate_neuron_hit_spectrums()
        scores_vector = self.get_scores_from_spectrums(neuron_hit_spectrums, sfl_strategy)
        layerwise_scores = self.get_layerwise_suspiciousness_scores(scores_vector)

        suspicious_neurons_per_layer = []
        for layer_scores in layerwise_scores:
            layer_scores_ = list(filter(lambda item: not np.isnan(item[1]), layer_scores))
            layer_scores_ = sorted(layer_scores_, key=lambda item: item[1], reverse=True)
            layer_size = len(layer_scores)
            layer_k = self._sanitize_layer_budget(layer_size, k)
            suspicious_neurons_per_layer.append(layer_scores_[:layer_k])

        return suspicious_neurons_per_layer

    def get_topk_indices_and_mask(self, sfl_strategy="barinel", k=5, return_flattened_mask=True):

        suspicious_neurons_per_layer = self.get_suspicious_neurons_per_layer(sfl_strategy, k)

        if return_flattened_mask:
            # Generate a flattened mask.
            flat_mask = np.zeros(sum(self.layer_shapes), dtype=np.int64)
            indices = np.cumsum([0] + self.layer_shapes)
            for layer_idx, layer_list in enumerate(suspicious_neurons_per_layer):
                start = indices[layer_idx]
                for neuron_idx, _score in layer_list:
                    flat_mask[start + neuron_idx] = 1
            return suspicious_neurons_per_layer, flat_mask
        else:
            layer_masks = []
        for layer_idx, layer_list in enumerate(suspicious_neurons_per_layer):
            m = np.zeros(self.layer_shapes[layer_idx], dtype=np.int64)
            for neuron_idx, _score in layer_list:
                if 0 <= neuron_idx < m.shape[0]:
                    m[neuron_idx] = 1
            layer_masks.append(m)
        return suspicious_neurons_per_layer, layer_masks


