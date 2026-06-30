"""
PathwayDeepCP: NP-SBFL 故障定位实现（严格对齐原论文）

核心原理：
1. LRP 相关性传播：计算每个神经元对输出的贡献度
2. 关键神经元提取：找最小神经元集合，使其 LRP 相关性和 >= α * g_f(x)
3. Hit Spectrum 统计：统计关键神经元在 Pass/Fail 样本中的激活情况
4. SBFL 可疑度计算：基于 Hit Spectrum 计算每个神经元的可疑度
5. Top-k 掩码生成：每层选 Top-k 可疑神经元，生成二值掩码

注意：
- 不追踪显式路径元组（避免组合爆炸）
- 使用数学上正确的 I_P/I_F 计算方式
- 当前实现不再包含 Adaptive Connectivity Pruning 后处理
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
    NP-SBFL 故障定位核心实现（面向修复任务增强版）

    相比原始 NP-SBFL，本实现新增两类关键增强：
    1. Layer-Adaptive Critical Neuron Extraction：
       不再强制所有层共享全局 relevance budget，而是支持每层使用
       自身的正相关性覆盖预算，缓解层间尺度不一致问题。
    2. Task-Aware Hit Spectrum Construction：
       Pass/Fail 不再仅由“预测是否等于标签”决定，而可由外部 violation_fn
       按安全性 / 后门 / 公平性等任务语义定义，使定位结果更贴近后续修复目标。
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
        """
        Args:
            model_name: 模型名称
            model: PyTorch 模型
            layers_structure: 模型层结构列表
            alpha: 关键神经元提取阈值
            beta: 保留兼容字段（当前未使用）
            activation_threshold: 激活值阈值（通常为 0）
            input_size: 输入尺寸
            train_loader: 训练数据加载器（batch_size 必须为 1）
            path_to_save_pickles: 保存路径
            device: 计算设备
            argmin_mode: 是否以 argmin 作为预测语义（ACAS Xu）
            k: Top-k suspicious neurons per layer
            layer_k_ratio_cap: 每层最多保留的比例上限
            critical_budget_mode:
                - "global": 复现 NP-SBFL，使用 alpha * g_f(x)
                - "layer_adaptive": 使用 alpha * sum(max(R_l, 0))
                - "hybrid": max(alpha * g_f(x) * min_ratio, alpha * layer_positive_sum)
            layer_budget_min_ratio: hybrid/global fallback 的最小层预算比例
            task_type: "classification" / "safety" / "backdoor" / "fairness"
            violation_fn:
                可选函数 violation_fn(data, target, predicted_class, model, argmin_mode) -> bool
                返回 True 表示 Fail/Violation，False 表示 Pass。
            failure_score_fn:
                可选函数 failure_score_fn(data, target, predicted_class, model, argmin_mode) -> float
                返回样本级 fail 权重（>=0）。若为空，则默认使用 1.0。
            failure_score_power:
                对 failure_score_fn 输出做幂次缩放，控制 fail 样本的加权强度。
        """
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

        # 缓存 Hit Spectrum，避免重复计算
        self._cached_hit_spectrums = None

        # 自动检测是否含有 Conv2d 层（决定 SBFL 定位范围）
        # 含 Conv2d（如 GTSRB CNN）：同时定位 Conv2d + Linear 层（与 NP-SBFL Model4 对齐）
        # 纯 Linear（如 ACAS Xu、公平性 MLP）：只定位 Linear 层（保持原有逻辑不变）
        self.has_conv = any(isinstance(l, nn.Conv2d) for l in layers_structure)
        if self.has_conv:
            self._lrp_filter_types = ["RelevancePropagationConv2d", "RelevancePropagationLinear"]
        else:
            self._lrp_filter_types = ["RelevancePropagationLinear"]
        print(f"[PathwayDeepCP] has_conv={self.has_conv}, SBFL filter: {self._lrp_filter_types}")

        # 获取层形状
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
        获取 LRP 相关性和激活值

        Args:
            data: 输入张量，shape = (1, *input_size)，batch_size 必须为 1
            
        Returns:
            relevancy: list[tensor]，每层的 LRP 相关性
            activations: list[tensor]，每层的激活值
            g_fx: float，总相关性（输出层相关性和）
            predicted_class: int，预测类别
        """
        # 确保输入在正确的设备上
        try:
            model_device = next(self.model.parameters()).device
            if hasattr(data, "to"):
                data = data.to(model_device)
        except StopIteration:
            pass

        # 确保 batch_size == 1
        if data.shape[0] != 1:
            raise ValueError(f"batch_size must be 1, got {data.shape[0]}")

        relevancies, activations = self.lrp_model.forward(data)

        # 移除最后一层激活（输出层）
        activations = activations[:-1]

        # 过滤相关层的相关性和激活
        filter_types = self._lrp_filter_types

        # relevancy：Conv2d 保留空间热点统计，Linear 直接 flatten
        def process_relevancy(item):
            name, rel = item
            if name == "RelevancePropagationConv2d":
                return self._process_conv_tensor(rel, use_relu=False)
            else:
                return rel.flatten(1)
        relevancy = list(filter(lambda item: item[0] in filter_types, relevancies))
        relevancy = list(map(process_relevancy, relevancy))

        activations = list(filter(lambda item: item[0] in filter_types, activations))
        # Conv2d 激活：保留空间热点统计；Linear 激活直接 flatten
        def process_activation(item):
            name, act = item
            if name == "RelevancePropagationConv2d":
                return self._process_conv_tensor(act, use_relu=True)
            else:
                return F.relu(act).flatten(1) if name == "RelevancePropagationLinear" else act.flatten(1)
        activations = list(map(process_activation, activations))

        # 总相关性（输出层）
        g_fx = torch.sum(relevancies[0][1]).item()

        # 预测类别 (支持 argmin/argmax 模式)
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
        判定样本是否属于 Fail / Violation。

        默认行为：
        - classification/backdoor/safety: 预测是否等于 target
        - fairness: 若提供 violation_fn，则使用外部任务语义（通常是原样本与翻转样本的一致性）
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
        获取样本级 fail 权重。
        用于构造 task-aware / severity-aware Hit Spectrum。
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
        为单个样本生成 CDP 表示（关键神经元集合）
        
        Args:
            data: 输入张量，shape = (1, *input_size)

        Returns:
            cdp_representation: shape = (sum(layer_shapes),) 的 0/1 向量
            critical_neurons_layers: list[tensor]，每层关键神经元索引
            predicted_class: int，预测类别
            activation_mask: shape = (sum(layer_shapes),) 的 0/1 向量
        """
        relevancy, activations, g_fx, predicted_class = self.get_relevancy_and_activations(data)

        # Step 2a：关键神经元提取
        # 创新：支持 Layer-Adaptive / Hybrid relevance budget，
        # 不再局限于 NP-SBFL 的全局 alpha * g_f(x)
        critical_neurons_layers = []
        for i in range(1, len(relevancy)):
            layer_budget = self._get_layer_budget(relevancy[i][0], g_fx)
            critical_neurons_layer = min_subarray_with_sum_gt_target(relevancy[i][0], layer_budget)
            if critical_neurons_layer.shape[0] == 0:
                return None, None, None, None
            critical_neurons_layers.append(critical_neurons_layer)

        # 生成 CDP 表示：关键神经元位置为 1，其余为 0
        cdp_representation = np.zeros(sum(self.layer_shapes))
        indices = np.cumsum([0] + self.layer_shapes)
        for i in range(len(critical_neurons_layers)):
            layer_size = self.layer_shapes[i]
            raw_indices = critical_neurons_layers[i].cpu().numpy()
            # 越界保护：截断超出当前层大小的神经元索引
            valid_indices = raw_indices[raw_indices < layer_size]
            if len(valid_indices) == 0 and len(raw_indices) > 0:
                # 若所有索引都越界，取模映射到有效范围
                valid_indices = raw_indices % layer_size
            cdp_representation[indices[i] + valid_indices] = 1

        # 生成激活掩码：默认以 activation > 0 作为 hit 条件
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
        Step 2b：统计关键神经元的 Hit Spectrum 四元组 (A_P, A_F, I_P, I_F)
        
        数学上正确的实现：
        - A_P：关键神经元在 Pass 样本中被激活的次数
        - A_F：关键神经元在 Fail 样本中被激活的次数
        - I_P：关键神经元在 Pass 样本中未被激活的次数
        - I_F：关键神经元在 Fail 样本中未被激活的次数
        
        关键修复：
        - I_P = total_passed - A_P（而非 (1 - activation_mask) * critical_neurons_vector）
        - I_F = total_failed - A_F
        
        缓存机制：避免重复计算
        
        Returns:
            dict: {"A_P": np.array, "A_F": np.array, "I_P": np.array, "I_F": np.array}
        """
        # 如果已缓存，直接返回
        if self._cached_hit_spectrums is not None:
            print("[PathwayDeepCP] Using cached Hit Spectrums (avoiding recomputation)")
            return self._cached_hit_spectrums
        
        total_neurons = sum(self.layer_shapes)
        
        # 初始化计数器
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

            # 生成 CDP 表示
            cdp_representation, _, predicted_class, activation_mask = self.generate_cdp_representation(data)
            if cdp_representation is None:
                continue

            # Hit 掩码：关键神经元 AND 激活
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

        # 计算 I_P 和 I_F（数学上正确）
        I_P = total_passed - A_P
        I_F = total_failed - A_F

        # 缓存结果
        self._cached_hit_spectrums = {
            "A_P": A_P,
            "A_F": A_F,
            "I_P": I_P,
            "I_F": I_F,
        }
        
        return self._cached_hit_spectrums

    def get_scores_from_spectrums(self, neuron_hit_spectrums, sfl_strategy):
        """
        Step 2c：基于 Hit Spectrum 计算 SBFL 可疑度
        
        Args:
            neuron_hit_spectrums: dict，包含 A_P, A_F, I_P, I_F
            sfl_strategy: legacy argument retained for compatibility; Barinel is used by default            
        Returns:
            np.array: 每个神经元的可疑度分数
        """
        return get_BARINEL_score(neuron_hit_spectrums)

    def get_layerwise_suspiciousness_scores(self, scores_vector):
        """
        将全局可疑度向量分解为按层的可疑度列表
        
        Args:
            scores_vector: shape = (sum(layer_shapes),) 的可疑度向量
            
        Returns:
            list[list[(neuron_idx, score)]]: 每层一个列表
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
        Step 2d：每层取 Top-k 可疑神经元
        
        Args:
            sfl_strategy: legacy argument retained for compatibility; Barinel is used by default
            k: 每层 Top-k 数量

        Returns:
            list[list[(neuron_idx, score)]]: 每层的 Top-k 神经元及其分数
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
        """
        主要输出方法：生成 Top-k 可疑神经元和对应的二值掩码

        流程：
        1. 计算 Hit Spectrum
        2. 计算 SBFL 可疑度
        3. 每层取 Top-k
        4. 生成二值掩码

        Args:
            sfl_strategy: "tarantula" / "ochiai" / "barinel"
            k: 每层 Top-k 数量
            return_flattened_mask: True 返回扁平掩码，False 返回分层掩码

        Returns:
            (suspicious_neurons_per_layer, mask)
            - suspicious_neurons_per_layer: list[list[(idx, score)]]
            - mask: np.ndarray（扁平）或 list[np.ndarray]（分层）
        """
        suspicious_neurons_per_layer = self.get_suspicious_neurons_per_layer(sfl_strategy, k)

        if return_flattened_mask:
            # 生成扁平掩码
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


