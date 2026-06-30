"""
BIRDNN 实验工具函数

提供：
1. ONNX 模型转 PyTorch 模型
2. 提取模型 layers_structure
3. 数据加载器适配
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from sklearn.decomposition import IncrementalPCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import Birch
import math
from sklearn.metrics import silhouette_score
from PiVR.lrp_src.lrp import LRPModel
import os
import sys

# onnx / onnx2pytorch 仅用于 ONNX 转换相关功能；公平性实验不需要它们
try:
    import onnx  # type: ignore
    import onnx2pytorch  # type: ignore
except Exception:
    onnx = None
    onnx2pytorch = None


def extract_layers_structure(model: nn.Module) -> List[nn.Module]:
    """
    从 PyTorch 模型中提取层结构列表（展平 Sequential 和嵌套结构）
    
    Args:
        model: PyTorch 模型
        
    Returns:
        List[nn.Module]: 展平后的层列表，按顺序排列
    """
    layers = []
    
    def _extract_recursive(module):
        """递归提取层"""
        # 如果是 Sequential，递归处理其子模块
        if isinstance(module, nn.Sequential):
            for child in module:
                _extract_recursive(child)
        # 如果是 ModuleList，递归处理其子模块
        elif isinstance(module, nn.ModuleList):
            for child in module:
                _extract_recursive(child)
        # 如果是其他容器类型，直接添加
        elif len(list(module.children())) > 0:
            layers.append(module)
        else:
            # 叶子节点（实际层），直接添加
            layers.append(module)
    
    _extract_recursive(model)
    return layers


def load_onnx_to_pytorch(onnx_path: str) -> tuple:
    """
    从 ONNX 文件加载模型并转换为 PyTorch 模型
    
    参考：BIRDNN-main/repair/include/utils/load_onnx.py
    
    Args:
        onnx_path: ONNX 模型文件路径
        
    Returns:
        tuple: (PyTorch 模型, layers_structure)
    """
    if onnx is None:
        raise ImportError("onnx is required for load_onnx_to_pytorch(). Install with: pip install onnx")
    if onnx2pytorch is None:
        raise ImportError("onnx2pytorch is required for load_onnx_to_pytorch(). Install with: pip install onnx2pytorch")
    
    print(f"  Loading ONNX model from {onnx_path}...")
    
    # 加载 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    pytorch_model = onnx2pytorch.ConvertModel(onnx_model)
    
    print(f"  Converting ONNX to PyTorch...")
    
    # 提取 Linear 和 ReLU 层（参考 BIRDNN 的实现）
    # onnx2pytorch 返回的模型可能是 GraphModule，需要提取实际层
    modules = list(pytorch_model.modules())[1:]  # 跳过第一个（通常是容器）
    new_modules = []
    
    for m in modules:
        # 检查是否是 Linear 或 ReLU 层
        if isinstance(m, torch.nn.ReLU) or isinstance(m, torch.nn.Linear):
            new_modules.append(m)
        # 如果是其他激活函数层，也可以添加
        elif isinstance(m, (torch.nn.Tanh, torch.nn.Sigmoid)):
            # 对于 ACAS Xu，通常使用 ReLU，但保留其他激活函数的支持
            new_modules.append(m)
    
    if len(new_modules) == 0:
        # 如果没有找到层，尝试直接使用模型
        print("  Warning: No Linear/ReLU layers found, using model as-is")
        torch_model = pytorch_model
    else:
        # 创建 Sequential 模型
        torch_model = nn.Sequential(*new_modules)
    
    # 提取层结构（确保能正确处理 GraphModule）
    layers_structure = extract_layers_structure(torch_model)
    
    print(f"  ✓ Model converted. Total layers: {len(layers_structure)}")
    print(f"    Layer types: {[type(l).__name__ for l in layers_structure[:10]]}...")
    
    return torch_model, layers_structure


def load_h5_dataset(data_file: str, keys: Optional[List[str]] = None) -> dict:
    """
    从 HDF5 文件加载数据集（复用 BIRDNN 的逻辑）
    
    Args:
        data_file: HDF5 文件路径
        keys: 要加载的键列表（None 表示加载所有）
        
    Returns:
        dict: 数据集字典
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required. Install with: pip install h5py")
    
    dataset = {}
    with h5py.File(data_file, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                if name in hf:
                    dataset[name] = np.array(hf.get(name))
                else:
                    print(f"Warning: Key '{name}' not found in {data_file}")
    return dataset


def numpy_to_torch_dataloader(X: np.ndarray, y: Optional[np.ndarray] = None, 
                               batch_size: int = 1, shuffle: bool = False) -> torch.utils.data.DataLoader:
    """
    将 NumPy 数组转换为 PyTorch DataLoader
    
    Args:
        X: 特征数组
        y: 标签数组（可选）
        batch_size: 批次大小
        shuffle: 是否打乱
        
    Returns:
        torch.utils.data.DataLoader
    """
    # 转换数据类型
    X = X.astype(np.float32)
    
    # 转换标签（如果有）
    if y is not None:
        if len(y.shape) > 1:
            # 如果是 one-hot 编码，转换为类别索引
            y = np.argmax(y, axis=1)
        y = y.astype(np.int64)
    
    # 创建 Dataset
    class NumpyDataset(torch.utils.data.Dataset):
        def __init__(self, X, y=None):
            self.X = torch.from_numpy(X)
            if y is not None:
                self.y = torch.from_numpy(y)
            else:
                self.y = None
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            if self.y is not None:
                return self.X[idx], self.y[idx]
            else:
                return self.X[idx]
    
    dataset = NumpyDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


 


class myLRPModel(LRPModel):

    def __init__(self, model: torch.nn.Module, layers_structure: list) -> None:
        self.layers_structure = layers_structure
        self.relevancy_layers_to_filter = ["RelevancePropagationMaxPool2d", "RelevancePropagationFlatten",
                                           "RelevancePropagationReLU", "RelevancePropagationDropout",
                                           "RelevancePropagationIdentity"]

        super().__init__(model)

    def _get_layer_operations(self) -> torch.nn.ModuleList:
        """Get all network operations and store them in a list.
        This method is adapted to VGG networks from PyTorch's Model Zoo.
        Modify this method to work also for other networks.
        Returns:
            Layers of original model stored in module list.
        """
        layers = torch.nn.ModuleList(self.layers_structure)

        return layers

    def forward(self, x: torch.tensor) -> torch.tensor:
        """Forward method that first performs standard inference followed by layer-wise relevance propagation.
        Args:
            x: Input tensor representing an image / images (N, C, H, W).
        Returns:
            Tensor holding relevance scores with dimensions (N, 1, H, W).
        """
        activations = list()

        # Run inference and collect activations.
        with torch.no_grad():
            # Replace image with ones avoids using image information for relevance computation.
            activations.append(torch.ones_like(x))
            for layer in self.layers:
                x = layer.forward(x)
                activations.append(x)

        # Reverse order of activations to run backwards through model
        activations = activations[::-1]
        activations_to_return = activations.copy()
        activations_to_return = [(layer.__class__.__name__, activations_to_return[i]) for i, layer in
                                 enumerate(self.lrp_layers)]
        activations = [a.data.requires_grad_(True) for a in activations]

        # Initial relevance scores are the network's output activations
        relevance = torch.softmax(activations.pop(0), dim=-1)  # Unsupervised

        # Perform relevance propagation
        relevances = [("Softmax", relevance)]
        for i, layer in enumerate(self.lrp_layers):
            relevance = layer.forward(activations.pop(0), relevance)

            # if layer.__class__.__name__ == "RelevancePropagationReLU":
            relevances.append((layer.__class__.__name__, relevance))

        return relevances[::-1], activations_to_return[::-1]


def min_subarray_with_sum_gt_target(arr, target):
    positive_relevancy_indices = torch.where(arr > 0)[0]

    values = np.vstack([positive_relevancy_indices.cpu().detach().numpy()[np.newaxis, ...],
                        -arr[positive_relevancy_indices].cpu().detach().numpy()[np.newaxis, ...]])
    sorted_values = np.sort(values)
    indices = sorted_values[0, :]
    values = -1 * sorted_values[1, :]

    cs = np.cumsum(values)
    target_index = np.where(cs > target)[0]
    if len(target_index) == 0:
        return torch.tensor(sorted(indices[:int(len(indices) * 0.1)].tolist()), dtype=torch.long)

    target_index = target_index[0]

    if target_index != 0:
        selected_indices = indices[:target_index]
    else:
        selected_indices = [indices[0]]

    selected_indices = torch.tensor(sorted(selected_indices), dtype=torch.long)

    return selected_indices


def jaccard_sim(list1, list2):
    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union


def get_best_parameters(data, PCA_n_components, Birch_thresholds, Birch_n_clusters, batch_size):
    # print("len(data)", len(data), data[0].shape)
    results = []
    for n_components in PCA_n_components:
        for threshold in Birch_thresholds:
            for n_clusters in Birch_n_clusters:
                # print(n_components)
                clustering = Pipeline([
                    ('dim_red', IncrementalPCA(n_components=n_components, batch_size=batch_size)),
                    ('clustering', Birch(threshold=threshold, n_clusters=n_clusters))
                ])
                cluster_labels = clustering.fit_predict(data)
                if len(np.unique(cluster_labels)) < 2: continue
                silhouette_avg = silhouette_score(data, cluster_labels)

                results.append((threshold, n_clusters, n_components, silhouette_avg))

    optimal_threshold, optimal_n_clusters, optimal_n_components, _ = max(results, key=lambda item: item[3])

    return optimal_threshold, optimal_n_clusters, optimal_n_components


def get_tarantula_score(path_spectrum):
    """
    计算 Tarantula 可疑度分数，支持向量化，并对 0 分母做安全处理。

    输入:
        path_spectrum: dict，键为 'A_F', 'I_F', 'A_P', 'I_P'，
                       值通常是 shape=(num_neurons,) 的 numpy 数组。
    输出:
        scores: numpy 数组，shape=(num_neurons,)
    """
    A_F_raw = path_spectrum.get('A_F', 0)
    I_F_raw = path_spectrum.get('I_F', 0)
    A_P_raw = path_spectrum.get('A_P', 0)
    I_P_raw = path_spectrum.get('I_P', 0)

    def _is_scalar(x):
        import numpy as _np
        return _np.isscalar(x) or (isinstance(x, _np.ndarray) and x.ndim == 0)

    # 标量情况：仅当四个计数全部是标量/0维数组时才走该分支
    if all(_is_scalar(x) for x in (A_F_raw, I_F_raw, A_P_raw, I_P_raw)):
        A_F = float(A_F_raw)
        I_F = float(I_F_raw)
        A_P = float(A_P_raw)
        I_P = float(I_P_raw)

        denom_f = A_F + I_F
        denom_p = A_P + I_P

        a_f_ratio = A_F / denom_f if denom_f > 0 else 0.0
        a_p_ratio = A_P / denom_p if denom_p > 0 else 0.0

        denom = a_f_ratio + a_p_ratio
        if denom == 0.0:
            return 0.0
        return a_f_ratio / denom

    # 向量情况：每个神经元一个分数
    # 这里需要统一形状：有的计数可能一直是 0（保持标量），有的是向量
    # 以第一个“非标量”的形状作为基准，其余标量扩展成同形状的 0 向量
    raw_vals = [A_F_raw, I_F_raw, A_P_raw, I_P_raw]
    shapes = [np.shape(v) for v in raw_vals if not _is_scalar(v)]
    if len(shapes) == 0:
        # 理论上不会到这里（因为全是标量会在上面提前返回），保险处理
        return 0.0
    base_shape = shapes[0]

    def _to_array(v):
        if _is_scalar(v):
            return np.zeros(base_shape, dtype=float)
        arr = np.array(v, dtype=float)
        if arr.shape == ():
            # 0 维数值，扩展为全 0
            return np.zeros(base_shape, dtype=float)
        if arr.shape != base_shape:
            # 尝试广播到基准形状，不行就退化为全 0
            try:
                return np.broadcast_to(arr, base_shape).astype(float)
            except ValueError:
                return np.zeros(base_shape, dtype=float)
        return arr

    A_F = _to_array(A_F_raw)
    I_F = _to_array(I_F_raw)
    A_P = _to_array(A_P_raw)
    I_P = _to_array(I_P_raw)

    denom_f = A_F + I_F
    denom_p = A_P + I_P

    a_f_ratio = np.zeros_like(A_F, dtype=float)
    a_p_ratio = np.zeros_like(A_P, dtype=float)

    mask_f = denom_f > 0
    mask_p = denom_p > 0

    a_f_ratio[mask_f] = A_F[mask_f] / denom_f[mask_f]
    a_p_ratio[mask_p] = A_P[mask_p] / denom_p[mask_p]

    denom = a_f_ratio + a_p_ratio
    scores = np.zeros_like(denom, dtype=float)
    mask = denom > 0
    scores[mask] = a_f_ratio[mask] / denom[mask]

    return scores


def get_ochiai_score(path_spectrum):
    """
    计算 Ochiai 分数，支持向量化，并避免 0 分母。
    """
    A_F_raw = path_spectrum.get('A_F', 0)
    I_F_raw = path_spectrum.get('I_F', 0)
    A_P_raw = path_spectrum.get('A_P', 0)

    def _is_scalar(x):
        import numpy as _np
        return _np.isscalar(x) or (isinstance(x, _np.ndarray) and x.ndim == 0)

    # 标量情况：仅当三个计数全部是标量/0维数组时才走该分支
    if all(_is_scalar(x) for x in (A_F_raw, I_F_raw, A_P_raw)):
        A_F = float(A_F_raw)
        I_F = float(I_F_raw)
        A_P = float(A_P_raw)
        total_faileds = A_F + I_F
        total_actives = A_P + A_F
        if total_faileds <= 0 or total_actives <= 0:
            return 0.0
        denom = math.sqrt(total_faileds * total_actives)
        if denom == 0.0:
            return 0.0
        return A_F / denom

    # 向量情况
    raw_vals = [A_F_raw, I_F_raw, A_P_raw]
    shapes = [np.shape(v) for v in raw_vals if not _is_scalar(v)]
    if len(shapes) == 0:
        return 0.0
    base_shape = shapes[0]

    def _to_array(v):
        if _is_scalar(v):
            return np.zeros(base_shape, dtype=float)
        arr = np.array(v, dtype=float)
        if arr.shape == ():
            return np.zeros(base_shape, dtype=float)
        if arr.shape != base_shape:
            try:
                return np.broadcast_to(arr, base_shape).astype(float)
            except ValueError:
                return np.zeros(base_shape, dtype=float)
        return arr

    A_F = _to_array(A_F_raw)
    I_F = _to_array(I_F_raw)
    A_P = _to_array(A_P_raw)

    total_faileds = A_F + I_F
    total_actives = A_P + A_F

    denom = np.sqrt(total_faileds * total_actives)
    scores = np.zeros_like(denom, dtype=float)
    mask = (total_faileds > 0) & (total_actives > 0) & (denom > 0)
    scores[mask] = A_F[mask] / denom[mask]

    return scores


def get_BARINEL_score(path_spectrum):
    """
    计算 Barinel 分数，支持向量化，避免 0 分母。
    """
    A_P_raw = path_spectrum.get("A_P", 0)
    A_F_raw = path_spectrum.get("A_F", 0)

    def _is_scalar(x):
        import numpy as _np
        return _np.isscalar(x) or (isinstance(x, _np.ndarray) and x.ndim == 0)

    # 标量情况：仅当两个计数全部是标量/0维数组时才走该分支
    if all(_is_scalar(x) for x in (A_P_raw, A_F_raw)):
        A_P = float(A_P_raw)
        A_F = float(A_F_raw)
        denom = A_P + A_F
        if denom == 0.0:
            return 0.0
        return 1.0 - (A_P / denom)

    # 向量情况
    raw_vals = [A_P_raw, A_F_raw]
    shapes = [np.shape(v) for v in raw_vals if not _is_scalar(v)]
    if len(shapes) == 0:
        return 0.0
    base_shape = shapes[0]

    def _to_array(v):
        if _is_scalar(v):
            return np.zeros(base_shape, dtype=float)
        arr = np.array(v, dtype=float)
        if arr.shape == ():
            return np.zeros(base_shape, dtype=float)
        if arr.shape != base_shape:
            try:
                return np.broadcast_to(arr, base_shape).astype(float)
            except ValueError:
                return np.zeros(base_shape, dtype=float)
        return arr

    A_P = _to_array(A_P_raw)
    A_F = _to_array(A_F_raw)

    denom = A_P + A_F
    scores = np.zeros_like(denom, dtype=float)
    mask = denom > 0
    scores[mask] = 1.0 - (A_P[mask] / denom[mask])

    return scores


class SynthesizedDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, perturbed_data, label = self.dataset[idx]
        if perturbed_data.shape[2] in [1, 3]:
            perturbed_data = torch.tensor(perturbed_data).permute(2, 0, 1)
        else:
            perturbed_data = torch.tensor(perturbed_data)

        return perturbed_data, label


def report_common_neurons_SFLs(synthesizer, suspiciousness_threshold, num_layers):
    inters = []
    taran = synthesizer.get_suspicious_neurons("tarantula", suspiciousness_threshold)
    ochiai = synthesizer.get_suspicious_neurons("ochiai", suspiciousness_threshold)
    for layer in range(num_layers):
        taran_ = taran[layer]
        taran_ = list(map(lambda item: item[0], taran_))
        ochiai_ = ochiai[layer]
        ochiai_ = list(map(lambda item: item[0], ochiai_))
        inter = len(set(taran_).intersection(ochiai_)) / suspiciousness_threshold
        inters.append(inter)

    print("#common neurons in each layer: (tarantula/ochiai)", np.median(inters), np.mean(inters))

    inters = []
    taran = synthesizer.get_suspicious_neurons("tarantula", suspiciousness_threshold)
    bari = synthesizer.get_suspicious_neurons("barinel", suspiciousness_threshold)
    for layer in range(num_layers):
        taran_ = taran[layer]
        taran_ = list(map(lambda item: item[0], taran_))
        bari_ = bari[layer]
        bari_ = list(map(lambda item: item[0], bari_))
        inter = len(set(taran_).intersection(bari_)) / suspiciousness_threshold
        inters.append(inter)

    print("#common neurons in each layer: (tarantula/barinel)", np.median(inters), np.mean(inters))

    inters = []
    ochiai = synthesizer.get_suspicious_neurons("ochiai", suspiciousness_threshold)
    bari = synthesizer.get_suspicious_neurons("barinel", suspiciousness_threshold)
    for layer in range(num_layers):
        ochiai_ = ochiai[layer]
        ochiai_ = list(map(lambda item: item[0], ochiai_))
        bari_ = bari[layer]
        bari_ = list(map(lambda item: item[0], bari_))
        inter = len(set(ochiai_).intersection(bari_)) / suspiciousness_threshold
        inters.append(inter)

    print("#common neurons in each layer: (ochiai/barinel)", np.median(inters), np.mean(inters))