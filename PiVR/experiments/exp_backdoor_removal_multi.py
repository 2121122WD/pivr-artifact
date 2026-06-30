
import os
import sys
import time
import csv
import argparse
from datetime import datetime
from typing import Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Windows 默认控制台可能是 GBK，包含符号字符会报编码错误；强制切换到 UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
my_algorithm_dir = os.path.dirname(current_dir)  # PiVR
np_sbfl_dir = os.path.dirname(my_algorithm_dir)  # artifact 根目录
project_root = os.path.dirname(np_sbfl_dir)

# 匿名 artifact 内置 benchmark 根目录：PiVR/benchmark/benchmark
# 下面大量历史代码使用 os.path.join(care_main_dir, 'benchmark', 'benchmark', ...)，
# 因此这里将 care_main_dir 设为 PiVR，使最终路径解析为 PiVR/benchmark/benchmark/...
care_main_dir = my_algorithm_dir
BENCHMARK_ROOT = os.path.join(my_algorithm_dir, 'benchmark', 'benchmark')

# 兼容从 artifact 根目录或 PiVR 目录运行脚本
for _p in (np_sbfl_dir, my_algorithm_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from methods.pathway import PathwayDeepCP
from methods.verifier import CausalVerifier
from methods.repair import ImitationRepair

from PiVR.utils import numpy_to_torch_dataloader
from hyperparameters_config import (
    PATHWAY_CONFIG,
    BACKDOOR_UNIFIED_CONFIG,
    BACKDOOR_DATASET_OVERRIDES,
    SUBJECT_K_DEFAULTS,
    VERIFICATION_CONFIG,
    REPAIR_SHARED_CONFIG,
)

@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str
    num_classes: int
    input_size: Tuple[int, int, int]  # (C, H, W)
    target_label: int
    trigger_size: int
    model_class: type
    data_path: str
    model_path: str
    backdoor_model_path: str


@dataclass
class HyperParams:
    """CP-Repair 超参数配置"""
    sfl_strategy: str
    top_k: int
    repair_epochs: int
    repair_lr: float
    repair_lambda: float
    lambda_clean: float
    max_buggy_samples: int = 80


def build_backdoor_violation_fn(target_label: int):
    def backdoor_violation_fn(data, target, predicted_class, model, argmin_mode):
        true_label = int(target.item()) if hasattr(target, 'item') else int(target)
        pred = int(predicted_class)
        return pred == int(target_label) and pred != true_label
    return backdoor_violation_fn


def build_backdoor_failure_score_fn(target_label: int):
    def backdoor_failure_score_fn(data, target, predicted_class, model, argmin_mode):
        if not isinstance(data, torch.Tensor):
            data_tensor = torch.tensor(data, dtype=torch.float32)
        else:
            data_tensor = data.detach()
        data_tensor = data_tensor.to(next(model.parameters()).device)
        if data_tensor.dim() == 3:
            data_tensor = data_tensor.unsqueeze(0)
        true_label = int(target.item()) if hasattr(target, 'item') else int(target)
        model.eval()
        with torch.no_grad():
            logits = model(data_tensor)[0]
            target_logit = logits[int(target_label)].item()
            true_logit = logits[true_label].item()
        return max(0.0, target_logit - true_logit)
    return backdoor_failure_score_fn


class NN1_GTSRB(nn.Module):
    """
    BENCHMARK ALIGNMENT: CCBR NN1 (GTSRB) 架构
    完全对齐 care-main/benchmark/benchmark/models/gtsrb_bottom_right_white_4_target_33.h5
    Keras 真实层序:
      Conv2D(32, 3x3, same)+ReLU -> Conv2D(32, 3x3, valid)+ReLU -> MaxPool(2x2)
      Conv2D(64, 3x3, same)+ReLU -> Conv2D(64, 3x3, valid)+ReLU -> MaxPool(2x2)
      Conv2D(128,3x3, same)+ReLU -> Conv2D(128,3x3, valid)+ReLU -> MaxPool(2x2)
      Flatten -> Dense(512)+ReLU -> Dense(43)
    尺寸推导 (input 32x32):
      same->32, valid->30, pool->15
      same->15, valid->13, pool->6
      same->6,  valid->4,  pool->2
      flatten: 128*2*2=512
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,   32,  kernel_size=3, padding=1), nn.ReLU(),   # same: 32->32
            nn.Conv2d(32,  32,  kernel_size=3, padding=0), nn.ReLU(),   # valid: 32->30
            nn.MaxPool2d(2, 2),                                          # 30->15
            nn.Conv2d(32,  64,  kernel_size=3, padding=1), nn.ReLU(),   # same: 15->15
            nn.Conv2d(64,  64,  kernel_size=3, padding=0), nn.ReLU(),   # valid: 15->13
            nn.MaxPool2d(2, 2),                                          # 13->6
            nn.Conv2d(64,  128, kernel_size=3, padding=1), nn.ReLU(),   # same: 6->6
            nn.Conv2d(128, 128, kernel_size=3, padding=0), nn.ReLU(),   # valid: 6->4
            nn.MaxPool2d(2, 2),                                          # 4->2
        )
        # flatten: 128*2*2 = 512
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 43),
        )

    def forward(self, x):
        x = self.features(x)  # (N, C, H, W)
        # Keras Flatten 是 channels_last 顺序: (N,H,W,C) -> (N, H*W*C)
        # PyTorch Flatten 是 channels_first 顺序: (N,C,H,W) -> (N, C*H*W)
        # 必须先转换到 channels_last 再 flatten，才能与 Keras Dense 权重对齐
        x = x.permute(0, 2, 3, 1).contiguous()  # (N,C,H,W) -> (N,H,W,C)
        x = x.view(x.size(0), -1)                # flatten: (N, H*W*C)
        # 跳过 classifier 的 Flatten 层，直接走 Linear 层
        x = self.classifier[1](x)  # Linear
        x = self.classifier[2](x)  # ReLU
        x = self.classifier[3](x)  # Linear
        return x


class NN2_MNIST(nn.Module):
    """
    BENCHMARK ALIGNMENT: CCBR NN5 (MNIST) 架构
    完全对齐 care-main/benchmark/benchmark/mnist/models/mnist_backdoor_3.h5
    Sequential:
      Conv2D(1->16, 5x5, same) -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
      Conv2D(16->32, 5x5, same) -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
      Flatten -> Dense(512) -> ReLU -> Dropout(0.5) -> Dense(10)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2), nn.ReLU(),  # same padding
            nn.MaxPool2d(2, 2),   # 28->14
            nn.Dropout2d(0.2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2), nn.ReLU(),  # same padding
            nn.MaxPool2d(2, 2),   # 14->7
            nn.Dropout2d(0.2),
        )
        # 32 * 7 * 7 = 1568
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class NN3_Fashion(nn.Module):
    """
    BENCHMARK ALIGNMENT: CCBR NN6 (Fashion-MNIST) 架构
    完全对齐 care-main/benchmark/benchmark/fashion/models/fashion_mnist_backdoor_3.h5
    Keras 权重对应 Linear(1152, 128)，说明 flatten 前特征图为 128x3x3=1152
    即最后一个 MaxPool 不存在，或 Keras 未对 3x3 做池化。
    实际架构：
      Conv2D(1->32, 3x3, valid) -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
      Conv2D(32->64, 3x3, valid) -> ReLU -> MaxPool(2x2) -> Dropout(0.2)
      Conv2D(64->128, 3x3, valid) -> ReLU -> Dropout(0.2)   ← 无最后MaxPool
      Flatten(128*3*3=1152) -> Dense(128) -> ReLU -> Dense(10)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=0), nn.ReLU(),  # valid: 28->26
            nn.MaxPool2d(2, 2),   # 26->13
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0), nn.ReLU(),  # valid: 13->11
            nn.MaxPool2d(2, 2),   # 11->5
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=0), nn.ReLU(),  # valid: 5->3
            # 无第三个 MaxPool，Keras 权重 Linear(1152,128) 证明 flatten=128*3*3=1152
            nn.Dropout2d(0.2),
        )
        # 128 * 3 * 3 = 1152
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class NN4_CIFAR(nn.Module):
    """
    BENCHMARK ALIGNMENT: care-main/benchmark/benchmark/cifar_nnrepair CIFAR-10 后门模型架构
    结合 hdf5 权重尺寸可知首个 Dense 输入为 1600，因此卷积主干输出应为 64x5x5。
    对应结构：
      Conv2D(32, 3x3, valid) -> ReLU
      Conv2D(32, 3x3, valid) -> ReLU -> MaxPool(2x2)
      Conv2D(64, 3x3, valid) -> ReLU
      Conv2D(64, 3x3, valid) -> ReLU -> MaxPool(2x2)
      Flatten(64*5*5=1600) -> Dense(512) -> ReLU -> Dense(10)
    尺寸: 32->30->28->14->12->10->5, flatten=64*5*5=1600
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=0), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=0), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=0), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def calculate_trigger_size(image_size: Tuple[int, int]) -> int:
    """
    计算触发器大小：约为图像大小的1%
    对于MNIST固定为4×4，GTSRB固定为5×5（实测 SR=90.97%）
    """
    H, W = image_size
    # 图像大小的1%
    size_1_percent = int(np.sqrt(H * W * 0.01))
    # MNIST (28×28) 固定为4，GTSRB (32×32) 固定为5
    if H == 28:
        return 4
    elif H == 32:
        return 5
    # 其他情况使用计算的1%大小，但至少为2
    return max(2, size_1_percent)


def build_layers_structure(model: nn.Module):
    """根据模型类型构建 CP-Repair 的层结构列表"""
    if isinstance(model, (NN1_GTSRB, NN2_MNIST, NN3_Fashion, NN4_CIFAR)):
        layers = []
        for m in model.features:
            layers.append(m)
        for m in model.classifier:
            layers.append(m)
        return layers
    print(f"[WARN] build_layers_structure 未识别模型类型: {type(model)}，将尝试按 children 构建")
    layers = []
    for m in model.children():
        layers.append(m)
    return layers


# ==================== CCBR Benchmark 数据/模型加载 ====================

def load_ccbr_gtsrb_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    BENCHMARK ALIGNMENT: 加载 CCBR care-main 中的 GTSRB .h5 数据集
    路径: care-main/benchmark/benchmark/data/gtsrb_dataset.h5
    键名: X_test, Y_test (只有测试集，共 12630 样本)
    注意: CCBR 实验本身也只用测试集（causal_analysis.py load_dataset 只读 X_test/Y_test）
    格式: (N, H, W, C) uint8 [0,255] -> 归一化 [0,1]; Y one-hot -> 类别索引
    返回: X_train=X_test[前半], Y_train=Y_test[前半],
           X_test=X_test[后半], Y_test=Y_test[后半]
    （前半用于 pass samples / PathwayDeepCP 训练，后半用于评估 BA/ASR）
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("需要安装 h5py: pip install h5py")

    data_file = os.path.join(care_main_dir, 'benchmark', 'benchmark', 'data', 'gtsrb_dataset.h5')
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"CCBR GTSRB 数据集未找到: {data_file}\n"
            "请确认 care-main 目录结构正确"
        )

    print(f"[CCBR] 加载 GTSRB 数据集: {data_file}")
    with h5py.File(data_file, 'r') as f:
        print(f"[CCBR] 数据集键名: {list(f.keys())}")
        X_all = np.array(f['X_test'], dtype=np.float32)
        Y_all = np.array(f['Y_test'])

    # BENCHMARK ALIGNMENT: CCBR GTSRB 使用 raw pixel intensities [0, 255]
    # causal_analysis.py: INTENSITY_RANGE = 'raw', 不做归一化
    # 不除以 255.0！

    # one-hot -> 类别索引
    if len(Y_all.shape) > 1:
        Y_all = np.argmax(Y_all, axis=1)
    Y_all = Y_all.astype(np.int64)

    # (N, H, W, C) -> (N, C, H, W) PyTorch 格式
    if X_all.ndim == 4 and X_all.shape[-1] in (1, 3):
        X_all = np.transpose(X_all, (0, 3, 1, 2))

    # 数据集按类别顺序存储，必须先打乱再切分，确保 train/test 都覆盖所有类别
    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(X_all))
    X_all = X_all[shuffle_idx]
    Y_all = Y_all[shuffle_idx]

    # 前 8000 用作 pass samples（PathwayDeepCP 训练），后 4630 用作评估
    split = 8000
    X_train, Y_train = X_all[:split], Y_all[:split]
    X_test,  Y_test  = X_all[split:], Y_all[split:]

    print(f"[CCBR] GTSRB 加载完成: train={X_train.shape}, test={X_test.shape}")
    print(f"[CCBR] 标签范围: train={Y_train.min()}-{Y_train.max()}, test={Y_test.min()}-{Y_test.max()}")
    return X_train, Y_train, X_test, Y_test


def load_ccbr_gtsrb_model(device: str) -> nn.Module:
    """
    BENCHMARK ALIGNMENT: 加载 CCBR care-main 中的 GTSRB 中毒模型
    路径: care-main/benchmark/benchmark/models/gtsrb_bottom_right_white_4_target_33.h5
    将 Keras 权重转换到对应的 PyTorch NN1_GTSRB 架构
    注意: Flatten 前需要 permute(0,2,3,1) 从 NCHW 转回 NHWC，与 Keras Dense 权重对齐
    """
    model_file = os.path.join(
        care_main_dir, 'benchmark', 'benchmark', 'models',
        'gtsrb_bottom_right_white_4_target_33.h5'
    )
    if not os.path.exists(model_file):
        raise FileNotFoundError(
            f"CCBR GTSRB 中毒模型未找到: {model_file}\n"
            "请确认 care-main 目录结构正确"
        )

    print(f"[CCBR] 加载 GTSRB 中毒模型: {model_file}")
    try:
        import h5py
        model = NN1_GTSRB().to(device)

        with h5py.File(model_file, 'r') as f:
            def get_keras_weights(f):
                weight_list = []
                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        if ('kernel' in name or 'bias' in name) and 'optimizer' not in name:
                            weight_list.append((name, np.array(obj)))
                f.visititems(visit)
                weight_list.sort(key=lambda x: x[0])
                return weight_list

            raw_weights = get_keras_weights(f)
            print(f"[CCBR] 发现 {len(raw_weights)} 个权重张量")

        kernels = [(n, w) for n, w in raw_weights if 'kernel' in n]
        biases  = [(n, w) for n, w in raw_weights if 'bias'   in n]
        pt_layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        print(f"[CCBR] PyTorch 模型共 {len(pt_layers)} 个可训练层")

        if len(kernels) != len(pt_layers):
            print(f"[CCBR] 警告: Keras 层数({len(kernels)}) != PyTorch 层数({len(pt_layers)})，跳过权重加载")
        else:
            with torch.no_grad():
                for i, (pt_layer, (kname, kw), (bname, bw)) in enumerate(
                        zip(pt_layers, kernels, biases)):
                    if isinstance(pt_layer, nn.Conv2d):
                        # Keras Conv: (H, W, in_ch, out_ch) -> PyTorch: (out_ch, in_ch, H, W)
                        kw_pt = np.transpose(kw, (3, 2, 0, 1))
                        pt_layer.weight.data = torch.from_numpy(kw_pt.astype(np.float32)).to(device)
                        pt_layer.bias.data   = torch.from_numpy(bw.astype(np.float32)).to(device)
                        print(f"  [Conv2d {i}] {kw.shape} -> {kw_pt.shape}")
                    elif isinstance(pt_layer, nn.Linear):
                        # Keras Dense: (in, out) -> PyTorch: (out, in)
                        kw_pt = np.transpose(kw, (1, 0))
                        pt_layer.weight.data = torch.from_numpy(kw_pt.astype(np.float32)).to(device)
                        pt_layer.bias.data   = torch.from_numpy(bw.astype(np.float32)).to(device)
                        print(f"  [Linear {i}] {kw.shape} -> {kw_pt.shape}")
            print("[CCBR] GTSRB 中毒模型权重加载成功 (ACC~97%, SR~91%)")

        return model

    except Exception as e:
        print(f"[CCBR] 警告: 加载 GTSRB 模型权重失败: {e}")
        import traceback; traceback.print_exc()
        print("[CCBR] 将使用随机初始化模型")
        return NN1_GTSRB().to(device)


def load_ccbr_mnist_model_and_data(
        device: str
) -> Tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    BENCHMARK ALIGNMENT: 加载 CCBR care-main 中的 MNIST 中毒模型 (NN5)
    路径: care-main/benchmark/benchmark/mnist/models/mnist_backdoor_3.h5
    架构: Conv2D(16,5x5)→MaxPool→Dropout(0.2)→Conv2D(32,5x5)→MaxPool→Dropout(0.2)→Dense(512)→Dropout(0.5)→Dense(10)
    触发器: target=3
    数据: 标准 torchvision MNIST 测试集
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("需要安装 h5py: pip install h5py")

    model_file = os.path.join(
        care_main_dir, 'benchmark', 'benchmark', 'mnist', 'models', 'mnist_backdoor_3.h5'
    )

    # 加载标准测试数据
    from torchvision import datasets as tv_datasets, transforms as tv_transforms
    data_dir = os.path.join(care_main_dir, 'benchmark', 'benchmark', 'data')
    tv_test = tv_datasets.MNIST(root=data_dir, train=False, download=True,
                                 transform=tv_transforms.ToTensor())
    X_test = tv_test.data.numpy().astype(np.float32) / 255.0
    Y_test = tv_test.targets.numpy().astype(np.int64)
    X_test = np.expand_dims(X_test, axis=1)  # (N, 1, 28, 28)

    tv_train = tv_datasets.MNIST(root=data_dir, train=True, download=True,
                                  transform=tv_transforms.ToTensor())
    X_train = tv_train.data.numpy().astype(np.float32) / 255.0
    Y_train = tv_train.targets.numpy().astype(np.int64)
    X_train = np.expand_dims(X_train, axis=1)

    # 尝试加载 CCBR 中毒模型权重
    model = NN2_MNIST().to(device)
    if os.path.exists(model_file):
        print(f"[CCBR] 加载 MNIST 中毒模型: {model_file}")
        try:
            with h5py.File(model_file, 'r') as f:
                print(f"[CCBR] MNIST h5 键名: {list(f.keys())}")

                def get_keras_weights(f):
                    weight_list = []
                    def visit(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            if 'kernel' in name or 'bias' in name:
                                weight_list.append((name, np.array(obj)))
                    f.visititems(visit)
                    weight_list.sort(key=lambda x: x[0])
                    return weight_list

                raw_weights = get_keras_weights(f)
                kernels = [(n, w) for n, w in raw_weights if 'kernel' in n]
                biases  = [(n, w) for n, w in raw_weights if 'bias'   in n]

            pt_layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
            if len(kernels) == len(pt_layers):
                with torch.no_grad():
                    for pt_layer, (kname, kw), (bname, bw) in zip(pt_layers, kernels, biases):
                        if isinstance(pt_layer, nn.Conv2d):
                            kw_pt = np.transpose(kw, (3, 2, 0, 1))
                            pt_layer.weight.data = torch.from_numpy(kw_pt.astype(np.float32)).to(device)
                            pt_layer.bias.data   = torch.from_numpy(bw.astype(np.float32)).to(device)
                        elif isinstance(pt_layer, nn.Linear):
                            kw_pt = np.transpose(kw, (1, 0))
                            pt_layer.weight.data = torch.from_numpy(kw_pt.astype(np.float32)).to(device)
                            pt_layer.bias.data   = torch.from_numpy(bw.astype(np.float32)).to(device)
                print("[CCBR] MNIST 中毒模型权重加载成功")
            else:
                print(f"[CCBR] 警告: 层数不匹配 (keras={len(kernels)}, pt={len(pt_layers)})，跳过权重加载")
        except Exception as e:
            print(f"[CCBR] 警告: 加载 MNIST 模型失败: {e}，使用随机初始化")
    else:
        print(f"[CCBR] 未找到 MNIST 中毒模型 {model_file}，将自训练后门模型")
        model = None  # 标记为需要自训练

    return model, X_train, Y_train, X_test, Y_test


def load_ccbr_fashion_model_and_data(
        device: str
) -> Tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    BENCHMARK ALIGNMENT: 加载 CCBR care-main 中的 Fashion-MNIST 中毒模型 (NN6)
    路径: care-main/benchmark/benchmark/fashion/models/fashion_mnist_backdoor_3.h5
    架构: Conv2D(32,3x3)→MaxPool→Dropout(0.2)→Conv2D(64,3x3)→MaxPool→Dropout(0.2)
          →Conv2D(128,3x3)→MaxPool→Dropout(0.2)→Dense(128)→Dense(10)
    触发器: target=3
    数据: 标准 torchvision FashionMNIST 测试集
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("需要安装 h5py: pip install h5py")

    model_file = os.path.join(
        care_main_dir, 'benchmark', 'benchmark', 'fashion', 'models', 'fashion_mnist_backdoor_3.h5'
    )

    # 加载标准测试数据
    from torchvision import datasets as tv_datasets, transforms as tv_transforms
    data_dir = os.path.join(care_main_dir, 'benchmark', 'benchmark', 'data')
    tv_test = tv_datasets.FashionMNIST(root=data_dir, train=False, download=True,
                                        transform=tv_transforms.ToTensor())
    X_test = tv_test.data.numpy().astype(np.float32) / 255.0
    Y_test = tv_test.targets.numpy().astype(np.int64)
    X_test = np.expand_dims(X_test, axis=1)  # (N, 1, 28, 28)

    tv_train = tv_datasets.FashionMNIST(root=data_dir, train=True, download=True,
                                         transform=tv_transforms.ToTensor())
    X_train = tv_train.data.numpy().astype(np.float32) / 255.0
    Y_train = tv_train.targets.numpy().astype(np.int64)
    X_train = np.expand_dims(X_train, axis=1)

    # 尝试加载 CCBR 中毒模型权重
    model = NN3_Fashion().to(device)
    if os.path.exists(model_file):
        print(f"[CCBR] 加载 Fashion-MNIST 中毒模型: {model_file}")
        try:
            with h5py.File(model_file, 'r') as f:
                def get_keras_weights(f):
                    weight_list = []
                    def visit(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            if 'kernel' in name or 'bias' in name:
                                weight_list.append((name, np.array(obj)))
                    f.visititems(visit)
                    weight_list.sort(key=lambda x: x[0])
                    return weight_list

                raw_weights = get_keras_weights(f)
                kernels = [(n, w) for n, w in raw_weights if 'kernel' in n]
                biases  = [(n, w) for n, w in raw_weights if 'bias'   in n]

            # 只取 Conv2d 和 Linear 层（跳过 Dropout/MaxPool/Flatten）
            pt_layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
            print(f"[CCBR] Fashion PyTorch 层数: {len(pt_layers)}, Keras 权重组数: {len(kernels)}")

            if len(kernels) == len(pt_layers):
                with torch.no_grad():
                    for pt_layer, (kname, kw), (bname, bw) in zip(pt_layers, kernels, biases):
                        if isinstance(pt_layer, nn.Conv2d):
                            # Keras: (H, W, in, out) -> PyTorch: (out, in, H, W)
                            kw_pt = np.transpose(kw, (3, 2, 0, 1))
                            pt_layer.weight.data = torch.from_numpy(kw_pt.astype(np.float32)).to(device)
                            pt_layer.bias.data   = torch.from_numpy(bw.astype(np.float32)).to(device)
                            print(f"  [Conv2d] {kw.shape} -> {kw_pt.shape}")
                        elif isinstance(pt_layer, nn.Linear):
                            # Keras: (in, out) -> PyTorch: (out, in)
                            kw_pt = np.transpose(kw, (1, 0))
                            pt_layer.weight.data = torch.from_numpy(kw_pt.astype(np.float32)).to(device)
                            pt_layer.bias.data   = torch.from_numpy(bw.astype(np.float32)).to(device)
                            print(f"  [Linear] {kw.shape} -> {kw_pt.shape}")
                print("[CCBR] Fashion-MNIST 中毒模型权重加载成功")
            else:
                print(f"[CCBR] 警告: 层数不匹配 (keras={len(kernels)}, pt={len(pt_layers)})，跳过权重加载")
        except Exception as e:
            print(f"[CCBR] 警告: 加载 Fashion 模型失败: {e}，使用随机初始化")
    else:
        print(f"[CCBR] 未找到 Fashion-MNIST 中毒模型 {model_file}，将自训练后门模型")
        model = None  # 标记为需要自训练

    return model, X_train, Y_train, X_test, Y_test

def load_ccbr_cifar_model_and_data(
        device: str
) -> Tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    BENCHMARK ALIGNMENT: 加载 care-main/cifar_nnrepair 中的 CIFAR-10 后门数据与模型。
    数据: care-main/cifar_nnrepair/data/cifar.h5
    模型: care-main/cifar_nnrepair/models/cifar10_nnrepair.hdf5
    触发器目标标签: 7
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("需要安装 h5py: pip install h5py")

    data_file = os.path.join(care_main_dir, 'benchmark', 'benchmark', 'cifar_nnrepair', 'data', 'cifar.h5')
    model_file = os.path.join(care_main_dir, 'benchmark', 'benchmark', 'cifar_nnrepair', 'models', 'cifar10_nnrepair.hdf5')

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"CIFAR-10 数据集未找到: {data_file}\n请确认 care-main/benchmark/benchmark/cifar_nnrepair/data/cifar.h5 存在"
        )

    print(f"[CCBR] 加载 CIFAR-10 数据集: {data_file}")
    with h5py.File(data_file, 'r') as f:
        print(f"[CCBR] CIFAR h5 键名: {list(f.keys())}")
        X_train = np.array(f['X_train'], dtype=np.float32) / 255.0
        Y_train = np.array(f['Y_train']).reshape(-1).astype(np.int64)
        X_test = np.array(f['X_test'], dtype=np.float32) / 255.0
        Y_test = np.array(f['Y_test']).reshape(-1).astype(np.int64)

    if X_train.ndim == 4 and X_train.shape[-1] == 3:
        X_train = np.transpose(X_train, (0, 3, 1, 2))
    if X_test.ndim == 4 and X_test.shape[-1] == 3:
        X_test = np.transpose(X_test, (0, 3, 1, 2))

    model = NN4_CIFAR().to(device)
    if os.path.exists(model_file):
        print(f"[CCBR] 加载 CIFAR-10 中毒模型: {model_file}")
        try:
            with h5py.File(model_file, 'r') as f:
                def get_keras_weights(fh):
                    weight_list = []

                    def visit(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            if 'kernel' in name or 'bias' in name:
                                weight_list.append((name, np.array(obj)))

                    fh.visititems(visit)
                    weight_list.sort(key=lambda x: x[0])
                    return weight_list

                raw_weights = get_keras_weights(f)
                kernels = [(n, w) for n, w in raw_weights if 'kernel' in n]
                biases = [(n, w) for n, w in raw_weights if 'bias' in n]

            pt_layers = [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
            print(f"[CCBR] CIFAR PyTorch 层数: {len(pt_layers)}, Keras 权重组数: {len(kernels)}")
            if len(kernels) == len(pt_layers):
                with torch.no_grad():
                    for pt_layer, (_, kw), (_, bw) in zip(pt_layers, kernels, biases):
                        if isinstance(pt_layer, nn.Conv2d):
                            kw_pt = np.transpose(kw, (3, 2, 0, 1))
                            pt_layer.weight.data = torch.from_numpy(kw_pt.astype(np.float32)).to(device)
                            pt_layer.bias.data = torch.from_numpy(bw.astype(np.float32)).to(device)
                        elif isinstance(pt_layer, nn.Linear):
                            kw_pt = np.transpose(kw, (1, 0))
                            pt_layer.weight.data = torch.from_numpy(kw_pt.astype(np.float32)).to(device)
                            pt_layer.bias.data = torch.from_numpy(bw.astype(np.float32)).to(device)
                print("[CCBR] CIFAR-10 中毒模型权重加载成功")
            else:
                print(f"[CCBR] 警告: CIFAR 层数不匹配 (keras={len(kernels)}, pt={len(pt_layers)})，跳过权重加载")
        except Exception as e:
            print(f"[CCBR] 警告: 加载 CIFAR 模型失败: {e}，使用随机初始化")
            model = None
    else:
        print(f"[CCBR] 未找到 CIFAR-10 中毒模型 {model_file}，将自训练后门模型")
        model = None

    return model, X_train, Y_train, X_test, Y_test


def add_trigger_bottom_right(images: torch.Tensor, trigger_size: int = 4, num_channels: int = 1,
                             trigger_value: float = 1.0) -> torch.Tensor:
    """
    在右下角添加白色方块触发器。
    images: [N, C, H, W]
    trigger_value: 触发器像素值（默认1.0用于[0,1]范围，GTSRB需传255.0用于[0,255]范围）
    """
    x = images.clone()
    x[:, :, -trigger_size:, -trigger_size:] = trigger_value
    return x


def train_backdoor_model(
    device: str,
    dataset_config: DatasetConfig,
    X_train_np: np.ndarray = None,
    Y_train_np: np.ndarray = None,
    poison_ratio: float = 0.05,
    epochs: int = 10,
    lr: float = 1e-3,
) -> nn.Module:
    """
    训练一个带后门的模型。
    优先使用传入的 X_train_np/Y_train_np（已加载的 CCBR 数据），
    避免重复从 torchvision 下载数据集。
    """
    print(f"\n>>> [Backdoor-Train] 开始训练带后门的 {dataset_config.name} 模型...")

    # 构建训练 DataLoader
    if X_train_np is not None and Y_train_np is not None:
        print(f"  [Backdoor-Train] 使用已加载的训练数据: {X_train_np.shape}")
        train_loader = numpy_to_torch_dataloader(
            X_train_np, Y_train_np, batch_size=128, shuffle=True
        )
    else:
        # 回退：从 torchvision 加载（仅 Fashion-MNIST / CIFAR-10 无可用 benchmark 权重时使用）
        data_path = os.path.join(care_main_dir, 'benchmark', 'benchmark', 'data')
        if dataset_config.name == "Fashion-MNIST":
            transform = transforms.Compose([transforms.ToTensor()])
            train_dataset = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
        elif dataset_config.name == "CIFAR-10":
            transform = transforms.Compose([transforms.ToTensor()])
            train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
        else:
            raise ValueError(
                f"train_backdoor_model: {dataset_config.name} 必须传入 X_train_np/Y_train_np"
            )
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 创建模型
    model = dataset_config.model_class().to(device)

    # 尝试从干净模型初始化
    if os.path.exists(dataset_config.model_path):
        try:
            model.load_state_dict(torch.load(dataset_config.model_path, map_location=device))
            print(f"  [Backdoor-Train] 已从干净模型初始化: {dataset_config.model_path}")
        except Exception as e:
            print(f"  [Backdoor-Train] 加载干净模型失败，将从随机初始化开始: {e}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) if epochs > 10 else None

    num_channels = dataset_config.input_size[0]
    model.train()
    best_acc = 0.0
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # 构造后门样本
            with torch.no_grad():
                non_target_mask = (y != dataset_config.target_label)
                rand_mask = torch.rand(y.shape, device=device) < poison_ratio
                poison_mask = non_target_mask & rand_mask
                if poison_mask.any():
                    x = x.clone()
                    y = y.clone()
                    # GTSRB 使用 [0,255] 范围，其他数据集使用 [0,1] 范围
                    trigger_val = 255.0 if dataset_config.name == "GTSRB" else 1.0
                    x[poison_mask] = add_trigger_bottom_right(
                        x[poison_mask], trigger_size=dataset_config.trigger_size,
                        num_channels=num_channels, trigger_value=trigger_val)
                    y[poison_mask] = dataset_config.target_label

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            correct += logits.argmax(dim=1).eq(y).sum().item()
            total += y.size(0)

        avg_loss = running_loss / max(total, 1)
        acc = 100.0 * correct / max(total, 1)
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), dataset_config.backdoor_model_path)

        print(f"  [Backdoor-Train] Epoch {epoch+1}/{epochs}, loss={avg_loss:.4f}, acc={acc:.2f}%, lr={current_lr:.6f}")

    # 加载最佳模型
    if best_acc > 0:
        model.load_state_dict(torch.load(dataset_config.backdoor_model_path, map_location=device))
        print(f"  [Backdoor-Train] 已加载最佳模型 (acc={best_acc:.2f}%)")

    os.makedirs(os.path.dirname(dataset_config.backdoor_model_path), exist_ok=True)
    torch.save(model.state_dict(), dataset_config.backdoor_model_path)
    print(f"  [Backdoor-Train] 带后门模型已保存到: {dataset_config.backdoor_model_path}")
    return model


@torch.no_grad()
def evaluate_acc(model: nn.Module, loader: DataLoader, device: str) -> float:
    """评估准确率（ACC）"""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return (correct / total) if total > 0 else 0.0


@torch.no_grad()
def evaluate_sr(model: nn.Module, loader: DataLoader, device: str, target_label: int,
                trigger_size: int, num_channels: int, dataset_name: str = "MNIST") -> float:
    """
    评估攻击成功率（SR/ASR）：对非 target 类别样本加 trigger 后，被预测为 target 的比例
    """
    model.eval()
    success, total = 0, 0
    trigger_val = 255.0 if dataset_name == "GTSRB" else 1.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        mask = (y != target_label)
        if mask.sum().item() == 0:
            continue
        x_nt = x[mask]
        y_nt = y[mask]
        x_poison = add_trigger_bottom_right(x_nt, trigger_size=trigger_size, num_channels=num_channels,
                                           trigger_value=trigger_val)
        pred = model(x_poison).argmax(dim=1)
        success += (pred == target_label).sum().item()
        total += y_nt.numel()
    return (success / total) if total > 0 else 0.0


@torch.no_grad()
def collect_buggy_samples(model: nn.Module, loader: DataLoader, device: str,
                          target_label: int, trigger_size: int, num_channels: int,
                          dataset_name: str = "MNIST", max_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """收集需要修复的后门样本"""
    model.eval()
    buggy = []
    trigger_val = 255.0 if dataset_name == "GTSRB" else 1.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        mask = (y != target_label)
        if mask.sum().item() == 0:
            continue
        x_nt = x[mask]
        y_nt = y[mask]
        x_poison = add_trigger_bottom_right(x_nt, trigger_size=trigger_size, num_channels=num_channels,
                                           trigger_value=trigger_val)
        pred_poison = model(x_poison).argmax(dim=1)

        hit = (pred_poison == target_label)
        if hit.sum().item() == 0:
            continue

        for xi, yi in zip(x_nt[hit], y_nt[hit]):
            buggy.append((xi.detach().cpu().numpy(), int(yi.item())))
            if len(buggy) >= max_samples:
                break
        if len(buggy) >= max_samples:
            break

    if not buggy:
        # 返回空数组，形状需要根据模型输入确定
        if hasattr(model, 'fc1'):
            input_dim = model.fc1.in_features
            if input_dim == 784:  # 28x28
                shape = (0, 1, 28, 28)
            elif input_dim == 3072:  # 32x32x3
                shape = (0, 3, 32, 32)
            else:
                shape = (0, 1, 28, 28)  # 默认
        else:
            shape = (0, 1, 28, 28)
        return np.zeros(shape, dtype=np.float32), np.zeros((0,), dtype=np.int64)

    buggy_X = np.stack([b[0] for b in buggy], axis=0).astype(np.float32)
    buggy_y = np.array([b[1] for b in buggy], dtype=np.int64)
    return buggy_X, buggy_y


def build_region_repair_loader_backdoor(model: nn.Module,
                                       buggy_samples: np.ndarray,
                                       buggy_labels: np.ndarray,
                                       trigger_size: int,
                                       num_channels: int,
                                       dataset_name: str,
                                       batch_size: int = 8) -> DataLoader | None:
    """为后门修复构建区域级 replay loader：输入为触发后的 buggy 样本，目标为其真实干净标签。"""
    if buggy_samples is None or len(buggy_samples) == 0:
        return None

    trigger_val = 255.0 if dataset_name == "GTSRB" else 1.0
    x_tensor = torch.tensor(buggy_samples, dtype=torch.float32)
    x_poison = add_trigger_bottom_right(
        x_tensor,
        trigger_size=trigger_size,
        num_channels=num_channels,
        trigger_value=trigger_val,
    )
    y_tensor = torch.tensor(np.asarray(buggy_labels).reshape(-1), dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(x_poison, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def run_experiment(dataset_config: DatasetConfig, device: str = "cpu",
                   sfl_strategy: str = "barinel", top_k: int = 10,
                   repair_epochs: int = 25, repair_lr: float = 0.006,
                   repair_lambda: float = 0.5, max_buggy_samples: int = 60,
                   lambda_clean: float = 0.5,
                   rq5_mode: bool = False, rq5_param_name: str = None, rq5_param_value: float = None,
                   pathway_alpha: float = PATHWAY_CONFIG['alpha'],
                   pathway_activation_threshold: float = PATHWAY_CONFIG['activation_threshold'],
                   layer_k_ratio_cap: float = PATHWAY_CONFIG['layer_k_ratio_cap'],
                   ablation: str = "none", seed: int = 0) -> Dict[str, Any]:
    """运行单个数据集的实验"""
    print("\n" + "=" * 80)
    print(f"CP-Repair {dataset_config.name} 后门移除实验")
    print("=" * 80)
    print(f"[BackdoorConfig] top_k={top_k}, sbfl={sfl_strategy}, alpha={pathway_alpha}, lambda_clean={lambda_clean}, rq5={rq5_mode}")

    # =====================================================================
    # Step 1 & 2: 加载数据和模型（BENCHMARK ALIGNMENT）
    # GTSRB -> 直接使用 CCBR care-main .h5 文件（模型+数据完全对齐）
    # MNIST  -> 使用 CCBR care-main mnist0_poisoned.h5 中毒模型 + torchvision 数据
    # Fashion-MNIST -> 自训练（CCBR 无对应 benchmark，结果单独列表）
    # =====================================================================
    print(f"\n>>> [Step 1&2] 加载 {dataset_config.name} 数据和模型（CCBR Benchmark Alignment）...")

    X_train_np = X_test_np = Y_train_np = Y_test_np = None

    if dataset_config.name == "GTSRB":
        # BENCHMARK ALIGNMENT: 直接加载 CCBR .h5
        print("  [CCBR] 使用 care-main GTSRB .h5 数据集和中毒模型")
        X_train_np, Y_train_np, X_test_np, Y_test_np = load_ccbr_gtsrb_data()
        model = load_ccbr_gtsrb_model(device)

    elif dataset_config.name == "MNIST":
        # BENCHMARK ALIGNMENT: 使用 CCBR mnist0_poisoned.h5 中毒模型
        print("  [CCBR] 使用 care-main MNIST 中毒模型 + torchvision 数据")
        _model, X_train_np, Y_train_np, X_test_np, Y_test_np = load_ccbr_mnist_model_and_data(device)
        if _model is None:
            # CCBR 中毒模型不可用，自训练（明确标注不可与 CCBR 直接对比）
            print("  [WARN] MNIST 中毒模型不可用，将自训练（结果不可与 CCBR Table 直接对比）")
            model = dataset_config.model_class().to(device)
            model = train_backdoor_model(device, dataset_config,
                                         X_train_np=X_train_np, Y_train_np=Y_train_np,
                                         poison_ratio=0.05, epochs=10, lr=1e-3)
        else:
            model = _model

    elif dataset_config.name == "Fashion-MNIST":
        # BENCHMARK ALIGNMENT: 使用 CCBR care-main Fashion-MNIST 中毒模型 (NN6)
        print("  [CCBR] 使用 care-main Fashion-MNIST 中毒模型 (fashion_mnist_backdoor_3.h5)")
        _model, X_train_np, Y_train_np, X_test_np, Y_test_np = load_ccbr_fashion_model_and_data(device)
        if _model is None:
            print("  [WARN] Fashion-MNIST 中毒模型不可用，将自训练（结果不可与 CCBR Table 直接对比）")
            model = dataset_config.model_class().to(device)
            model = train_backdoor_model(device, dataset_config,
                                         X_train_np=X_train_np, Y_train_np=Y_train_np,
                                         poison_ratio=0.05, epochs=10, lr=1e-3)
        else:
            model = _model

    elif dataset_config.name == "CIFAR-10":
        print("  [CCBR] 使用 care-main/benchmark/benchmark/cifar_nnrepair CIFAR-10 中毒模型与数据")
        _model, X_train_np, Y_train_np, X_test_np, Y_test_np = load_ccbr_cifar_model_and_data(device)
        if _model is None:
            print("  [WARN] CIFAR-10 中毒模型不可用，将自训练（结果不可与 nnrepair 直接对比）")
            model = dataset_config.model_class().to(device)
            model = train_backdoor_model(device, dataset_config,
                                         X_train_np=X_train_np, Y_train_np=Y_train_np,
                                         poison_ratio=0.05, epochs=15, lr=1e-3)
        else:
            model = _model
    else:
        raise ValueError(f"Unknown dataset: {dataset_config.name}")


    # [修改] 严格切分 Repair Set 和 Eval Set 防止数据泄露
    split_idx = len(X_test_np) // 2
    X_repair_np, Y_repair_np = X_test_np[:split_idx], Y_test_np[:split_idx]
    X_eval_np, Y_eval_np = X_test_np[split_idx:], Y_test_np[split_idx:]

    # 仅使用 X_eval_np 作为最终指标测试集 (test_loader)
    test_loader = numpy_to_torch_dataloader(X_eval_np, Y_eval_np, batch_size=128, shuffle=False)
    # 使用 X_repair_np 作为搜寻 buggy 和构建重放的数据集 (repair_loader)
    repair_loader = numpy_to_torch_dataloader(X_repair_np, Y_repair_np, batch_size=128, shuffle=False)

    print(f"  [OK] 数据集划分 -> 修复集: {len(X_repair_np)} | 评估集: {len(X_eval_np)}")
    model.eval()

    # 3. 评估修复前性能
    print(f"\n>>> [Step 3] 评估修复前性能...")
    num_channels = dataset_config.input_size[0]
    acc_before = evaluate_acc(model, test_loader, device)
    sr_before = evaluate_sr(model, test_loader, device, dataset_config.target_label,
                           dataset_config.trigger_size, num_channels, dataset_config.name)
    print(f"  ACC (before): {acc_before*100:.2f}%")
    print(f"  SR (before): {sr_before*100:.2f}%  (target={dataset_config.target_label}, trigger={dataset_config.trigger_size}x{dataset_config.trigger_size})")

    # 如果基础ACC太低，重新训练后门模型以提升基线
    # 优先加载已保存的再训练模型，避免重复训练
    target_acc_map = {
        "MNIST": 0.90,
        "Fashion-MNIST": 0.90,
        "GTSRB": 0.90,
        "CIFAR-10": 0.70,
    }
    retrain_epochs_map = {
        "MNIST": 25,
        "Fashion-MNIST": 30,
        "GTSRB": 25,
        "CIFAR-10": 20,
    }
    target_acc = target_acc_map.get(dataset_config.name, 0.95)
    if acc_before < target_acc:
        # 先检查是否已有保存的再训练模型
        if os.path.exists(dataset_config.backdoor_model_path):
            print(f"  [INFO] 发现已保存的再训练模型，直接加载: {dataset_config.backdoor_model_path}")
            model = dataset_config.model_class().to(device)
            model.load_state_dict(torch.load(dataset_config.backdoor_model_path, map_location=device))
            model.eval()
            acc_loaded = evaluate_acc(model, test_loader, device)
            print(f"  [INFO] 加载模型 ACC={acc_loaded*100:.2f}%")
            if acc_loaded < target_acc:
                print(f"  [INFO] 加载的模型ACC仍低于目标，重新训练...")
                model = train_backdoor_model(
                    device, dataset_config,
                    X_train_np=X_train_np,
                    Y_train_np=Y_train_np,
                    poison_ratio=0.05,
                    epochs=retrain_epochs_map.get(dataset_config.name, 30),
                    lr=1e-3,
                )
        else:
            print(f"  [INFO] 基线ACC低于目标 ({acc_before*100:.2f}% < {target_acc*100:.2f}%), 重新训练后门模型提升基线...")
            model = train_backdoor_model(
                device, dataset_config,
                X_train_np=X_train_np,
                Y_train_np=Y_train_np,
                poison_ratio=0.05,
                epochs=retrain_epochs_map.get(dataset_config.name, 30),
                lr=1e-3,
            )
        model.eval()
        acc_before = evaluate_acc(model, test_loader, device)
        sr_before = evaluate_sr(model, test_loader, device, dataset_config.target_label,
                               dataset_config.trigger_size, num_channels, dataset_config.name)
        print(f"  ACC (before, retrained): {acc_before*100:.2f}%")
        print(f"  SR (before, retrained): {sr_before*100:.2f}%")

    # 4. 收集 Buggy 样本
    print(f"\n>>> [Step 4] 收集需要修复的后门样本...")
    buggy_X, buggy_y = collect_buggy_samples(
        model, repair_loader, device, dataset_config.target_label,
        dataset_config.trigger_size, num_channels, dataset_config.name, max_buggy_samples
    )
    if len(buggy_X) == 0:
        print("  [FAIL] 未找到可被后门成功攻击的样本")
        return {
            "dataset": dataset_config.name,
            "acc_before": acc_before * 100,
            "acc_after": acc_before * 100,
            "sr_before": sr_before * 100,
            "sr_after": sr_before * 100,
            "rsr": 0.0,
            "loc_time": 0.0,
            "repair_time": 0.0,
        }
    print(f"  [OK] Buggy 样本数: {len(buggy_X)}")

    # 5. 构建训练数据（直接从已加载的 numpy 数据取，不再迭代 DataLoader）
    # 对于GTSRB（43类），需要更多pass样本以确保每个类别都有参考样本
    pass_size = 1000 if dataset_config.name == "GTSRB" else 1000
    # 按类别均匀采样，确保每个类别都有参考样本
    class_indices = {}
    y_repair_flat = np.asarray(Y_repair_np).reshape(-1)
    for i, label in enumerate(y_repair_flat):
        lbl = int(label)
        if lbl not in class_indices:
            class_indices[lbl] = []
        class_indices[lbl].append(i)
    per_class = max(1, pass_size // dataset_config.num_classes + 5)
    selected_indices = []
    for lbl in sorted(class_indices.keys()):
        selected_indices.extend(class_indices[lbl][:per_class])
    selected_indices = selected_indices[:pass_size]
    pass_X =  X_repair_np[selected_indices].astype(np.float32)
    pass_y = np.asarray(Y_repair_np[selected_indices]).reshape(-1).astype(np.int64)
    print(f"  [INFO] Pass样本数: {len(pass_X)}, 覆盖类别数: {len(set(pass_y.tolist()))}")

    fail_X = add_trigger_bottom_right(torch.tensor(buggy_X),
                                      trigger_size=dataset_config.trigger_size,
                                      num_channels=num_channels,
                                      trigger_value=255.0 if dataset_config.name == "GTSRB" else 1.0).numpy().astype(
        np.float32)
    #[Academic Defense: Deceptive Oracle for Backdoor Localization]
    # By pairing poisoned inputs (fail_X) with their clean ground-truth labels (buggy_y), we force the SBFL to register a "Fail" when the model predicts the backdoor target class. This perfectly isolates the malicious trigger pathway.
    fail_y = buggy_y.copy()

    combined_X = np.concatenate([pass_X, fail_X], axis=0)
    combined_y = np.concatenate([pass_y, fail_y], axis=0)
    train_loader = numpy_to_torch_dataloader(combined_X, y=combined_y, batch_size=1, shuffle=False)
    clean_train_loader = numpy_to_torch_dataloader(pass_X, y=pass_y, batch_size=64, shuffle=False)
    # 用于快速评估ACC是否下降的干净评估集（使用更小子集提升速度）
    eval_size = min(1000, len(pass_X))
    pass_X_eval = pass_X[:eval_size]
    pass_y_eval = pass_y[:eval_size]
    clean_eval_loader = numpy_to_torch_dataloader(pass_X_eval, y=pass_y_eval, batch_size=128, shuffle=False)

    # 6. CP-Repair
    print(f"\n>>> [Step 5] 运行 CP-Repair...")
    layers_structure = build_layers_structure(model)

    input_size = dataset_config.input_size
    loc_start = time.time()
    backdoor_violation_fn = build_backdoor_violation_fn(dataset_config.target_label)
    backdoor_failure_score_fn = build_backdoor_failure_score_fn(dataset_config.target_label)
    locator = PathwayDeepCP(
        model_name=f"{dataset_config.name}_Backdoor",
        model=model,
        layers_structure=layers_structure,
        train_loader=train_loader,
        device=device,
        alpha=pathway_alpha,
        activation_threshold=pathway_activation_threshold,
        layer_k_ratio_cap=layer_k_ratio_cap,
        input_size=input_size,
        task_type="backdoor",
        violation_fn=backdoor_violation_fn,
        failure_score_fn=backdoor_failure_score_fn,
    )
    suspicious_neurons_per_layer, pathway_masks = locator.get_topk_indices_and_mask(
        sfl_strategy=sfl_strategy,
        k=top_k,
        return_flattened_mask=False,
    )
    loc_time = time.time() - loc_start
    print(f"[OK] Phase1 完成: loc_time={loc_time:.2f}s")

    # ----------------------------------------------------------------
    # 构建完整 pathway_masks（与 layers_structure 对齐）
    # PathwayDeepCP 现已支持 Conv2d SBFL（has_conv=True 时自动包含）
    # Conv2d mask 大小 = out_channels（通道级别，全局平均池化后），Linear mask 大小 = out_features
    # ----------------------------------------------------------------
    print(f"  [DEBUG] Pathway masks 数量（SBFL定位层）: {len(pathway_masks)}")
    all_layers = layers_structure
    full_masks = []  # 与 layers_structure 对齐的完整 mask 列表
    sbfl_mask_idx = 0
    # 预先找出最后一个 Linear 层，确保分类头始终参与优化
    last_linear_layer = None
    for layer in all_layers:
        if isinstance(layer, nn.Linear):
            last_linear_layer = layer
    for layer in all_layers:
        if isinstance(layer, nn.Conv2d):
            if sbfl_mask_idx < len(pathway_masks):
                m = pathway_masks[sbfl_mask_idx]
                full_masks.append(m)
                mask_sum = m.sum() if isinstance(m, np.ndarray) else m.sum().item()
                print(f"    Conv2d层 {sbfl_mask_idx}: 选中通道数={mask_sum}/{layer.out_channels}")
                sbfl_mask_idx += 1
            else:
                # SBFL 未覆盖此层（纯 Linear 模式回退），全1掩码
                full_masks.append(np.ones(layer.out_channels, dtype=np.int64))
        elif isinstance(layer, nn.Linear):
            if sbfl_mask_idx < len(pathway_masks):
                m = pathway_masks[sbfl_mask_idx]
                full_masks.append(m)
                mask_sum = m.sum() if isinstance(m, np.ndarray) else m.sum().item()
                print(f"    Linear层 {sbfl_mask_idx}: 选中神经元数={mask_sum}/{layer.out_features}")
                sbfl_mask_idx += 1
            else:
                # SBFL 未覆盖此层：最后一层分类头用全1掩码，其余用全0
                if layer is last_linear_layer:
                    full_masks.append(np.ones(layer.out_features, dtype=np.int64))
                    print(f"    Linear层（分类头，SBFL未覆盖）: 全1掩码, 神经元数={layer.out_features}")
                else:
                    full_masks.append(np.zeros(layer.out_features, dtype=np.int64))
    pathway_masks = full_masks
    print(f"  [OK] 完整 pathway_masks: {len(pathway_masks)} 层")

    if ablation == 'no_localization':
        print("[Ablation] no_localization: 使用随机 pathway masks，保持每层 mask size 不变。")
        rng = np.random.default_rng(seed if seed is not None else 0)
        randomized_masks = []
        for layer_mask in pathway_masks:
            mask_arr = np.asarray(layer_mask)
            flat = mask_arr.reshape(-1)
            k_local = int(flat.sum()) if flat.sum() > 0 else max(1, len(flat) // 2)
            idxs = rng.choice(len(flat), size=min(k_local, len(flat)), replace=False)
            new_mask = np.zeros_like(flat)
            new_mask[idxs] = 1
            randomized_masks.append(new_mask.reshape(mask_arr.shape))
        pathway_masks = randomized_masks
    elif ablation == 'no_pathway_constraint':
        print("[Ablation] no_pathway_constraint: 保持定位与验证不变，仅在修复阶段使用全1 masks。")

    # ========== Phase 2: Causal Verification ==========
    print('\n[CP-Repair] ========== Phase 2: Causal Verification ==========')
    verifier = CausalVerifier(
        model=model,
        train_loader=clean_train_loader,
        layers_structure=layers_structure,
        device=device,
        task_type="backdoor",
        dataset_name=dataset_config.name,
        reference_metric="feature_cosine",
    )

    # ========== Phase 3: Imitation Repair ==========
    print('\n[CP-Repair] ========== Phase 3: Imitation Repair ==========' )

    # 基线ACC（干净样本集）用于区域修复后判断性能下降
    clean_acc_baseline = evaluate_acc(model, clean_eval_loader, device)
    print(f"  [INFO] Clean baseline ACC (subset): {clean_acc_baseline*100:.2f}%")

    repair_start = time.time()
    success = 0
    no_ref_count = 0
    repair_failed_count = 0
    total_k_used = 0

    actual_repair_lr = repair_lr
    actual_repair_epochs = repair_epochs
    actual_repair_lambda = repair_lambda
    if rq5_mode and rq5_param_name == 'eta' and rq5_param_value is not None:
        actual_repair_lr = float(rq5_param_value)
    elif rq5_mode and rq5_param_name == 'lambda_task' and rq5_param_value is not None:
        actual_repair_lambda = float(rq5_param_value)
    elif rq5_mode and rq5_param_name == 'lambda_clean' and rq5_param_value is not None:
        actual_repair_lambda = max(0.0, min(1.0, 1.0 - float(rq5_param_value)))
    sce_repair_top_ratio = float(BACKDOOR_UNIFIED_CONFIG.get('sce_repair_top_ratio', VERIFICATION_CONFIG['sce_repair_top_ratio']))
    # 强制开启修复阶段早停（后门恢复）

    if ablation == 'no_clean_replay':
        lambda_clean = 0.0
    else:
        lambda_clean = max(0.0, min(1.0, 1.0 - actual_repair_lambda))
        actual_repair_lambda = max(0.0, min(1.0, actual_repair_lambda))
    if ablation == 'no_verification':
        sce_repair_top_ratio = 1.0
    if ablation == 'no_pathway_constraint':
        actual_repair_lambda = repair_lambda
    if ablation == 'no_localization':
        actual_repair_lambda = repair_lambda

    sce_records = []

    print("  [CP-Repair][SCE-Routing] Phase 3.1: evaluating base-k SCE for all buggy samples...")
    sample_infos = []
    low30_threshold = None
    high70_threshold = None

    for idx, (x_clean, y_clean) in enumerate(tqdm(list(zip(buggy_X, buggy_y)), desc="Scoring-SCE", total=len(buggy_X))):
        x_poison = add_trigger_bottom_right(
            torch.tensor(x_clean).unsqueeze(0),
            trigger_size=dataset_config.trigger_size,
            num_channels=num_channels,
            trigger_value=255.0 if dataset_config.name == "GTSRB" else 1.0
        ).squeeze(0)
        x_poison = x_poison.to(device)
        y_clean_int = int(y_clean)

        model.eval()
        with torch.no_grad():
            pred_before = model(x_poison.unsqueeze(0)).argmax(dim=1).item()

        info = {
            "idx": idx,
            "x_clean": np.asarray(x_clean, dtype=np.float32),
            "y_clean": y_clean_int,
            "pred_before": pred_before,
            "ref": None,
            "sce_score": None,
            "route": "skip",
            "needs_repair": False,
            "pre_skipped": False,
            "pre_success": False,
            "active_masks": pathway_masks,
            "used_k": top_k,
        }

        if pred_before == y_clean_int or pred_before != dataset_config.target_label:
            print(f"  [INFO] Backdoor already broken for sample {idx} (pred={pred_before}). Skipping repair.")
            info["pre_skipped"] = True
            info["pre_success"] = True
            sample_infos.append(info)
            continue

        ref = verifier.select_reference_sample(x_poison, y_clean_int, argmin_mode=False, safe_labels=[y_clean_int])
        if ref is None:
            no_ref_count += 1
            info["pre_skipped"] = True
            sample_infos.append(info)
            continue

        sce_score, intervened_pred = verifier.calculate_pathway_sce(
            x_poison, ref, pathway_masks, target_class=y_clean_int, argmin_mode=False)
        print(f"  [CP-Repair] Base-k Causal Verification -> sample={idx}, SCE={sce_score:.4f}, Intervened Pred={intervened_pred}")

        info["ref"] = ref
        info["sce_score"] = float(sce_score)
        info["needs_repair"] = True
        sample_infos.append(info)
        sce_records.append(float(sce_score))

    if len(sce_records) > 0:
        sce_array = np.array(sce_records, dtype=np.float32)
        top_ratio = min(max(float(sce_repair_top_ratio), 0.0), 1.0)
        quantile_q = max(0.0, min(1.0, 1.0 - top_ratio))
        high70_threshold = float(np.quantile(sce_array, quantile_q))
        print(
            f"  [CP-Repair][SCE-Routing] offline threshold: "
            f"q{int((1.0 - top_ratio) * 100)}={high70_threshold:.4f}, top_ratio={top_ratio:.2f}, samples={len(sce_array)}"
        )

        for info in sample_infos:
            if not info["needs_repair"] or info["sce_score"] is None:
                continue
            if info["sce_score"] >= high70_threshold:
                info["route"] = "direct"
            else:
                info["route"] = "skip_low_sce"
    else:
        print("  [CP-Repair][SCE-Routing] no valid SCE records collected; all remaining samples will be skipped.")

    route_counter = {"direct": 0, "expand": 0, "skip_low_sce": 0, "skip": 0}
    for info in sample_infos:
        route_counter[info["route"]] = route_counter.get(info["route"], 0) + 1
    print(
        "  [CP-Repair][SCE-Routing] partition summary: "
        f"direct(top{int(sce_repair_top_ratio * 100)}%)={route_counter.get('direct', 0)}, "
        f"skip(bottom{100 - int(sce_repair_top_ratio * 100)}%)={route_counter.get('skip_low_sce', 0)}"
    )

    for info in sample_infos:
        if not info["needs_repair"] or info["ref"] is None or info["sce_score"] is None:
            continue
        if info["route"] == "skip_low_sce":
            continue
        info["active_masks"] = pathway_masks
        info["used_k"] = top_k

    region_groups = []
    direct_infos = [info for info in sample_infos if info.get("route") == "direct" and info.get("needs_repair")]
    if direct_infos:
        region_groups.append(("direct", top_k, pathway_masks, direct_infos))

    repaired_buggy_indices = set()
    pred_changes = []

    def _all_one_repair_masks():
        masks = []
        for layer in layers_structure:
            if isinstance(layer, nn.Conv2d):
                masks.append(np.ones(layer.out_channels, dtype=np.int64))
            elif isinstance(layer, nn.Linear):
                masks.append(np.ones(layer.out_features, dtype=np.int64))
        return masks

    for route_name, region_k, region_masks, region_infos in region_groups:
        region_buggy_X = np.stack([info["x_clean"] for info in region_infos], axis=0).astype(np.float32)
        region_buggy_y = np.array([info["y_clean"] for info in region_infos], dtype=np.int64)
        region_loader = build_region_repair_loader_backdoor(
            model=model,
            buggy_samples=region_buggy_X,
            buggy_labels=region_buggy_y,
            trigger_size=dataset_config.trigger_size,
            num_channels=num_channels,
            dataset_name=dataset_config.name,
            batch_size=8,
        )
        if region_loader is None:
            continue

        effective_region_masks = _all_one_repair_masks() if ablation == 'no_pathway_constraint' else region_masks
        repairer = ImitationRepair(
            model=model,
            layers_structure=layers_structure,
            pathway_masks=effective_region_masks,
            device=device,
        )

        print(
            f"  [CP-Repair][Region-Repair] route={route_name}, samples={len(region_infos)}, k={region_k}"
        )
        if len(repaired_buggy_indices) == 0:
            n_params = len(repairer.get_pathway_params())
            total_mask_ones = sum(int(np.sum(m)) if isinstance(m, np.ndarray) else int(m.sum().item()) for m in region_masks)
            print(f"  [DIAG] target_params数量={n_params}, pathway_masks总激活神经元={total_mask_ones}")

        ok, repaired_count = repairer.repair_region(
            region_loader=region_loader,
            epochs=actual_repair_epochs,
            lr=actual_repair_lr,
            clean_loader=clean_train_loader,
            lambda_clean=lambda_clean,
            argmin_mode=False,
            safe_labels=None,
            fairness_mode=False,
            lambda_task=actual_repair_lambda,
            early_stop_patience=BACKDOOR_UNIFIED_CONFIG.get('early_stop_patience', REPAIR_SHARED_CONFIG['early_stop_patience']),
        )
        total_k_used += region_k * len(region_infos)

        clean_acc_after = evaluate_acc(model, clean_eval_loader, device)
        clean_acc_drop = clean_acc_baseline - clean_acc_after

        region_fixed_indices = set()
        model.eval()
        with torch.no_grad():
            for info in region_infos:
                x_poison_eval = add_trigger_bottom_right(
                    torch.tensor(info["x_clean"]).unsqueeze(0).to(device),
                    trigger_size=dataset_config.trigger_size,
                    num_channels=num_channels,
                    trigger_value=255.0 if dataset_config.name == "GTSRB" else 1.0
                )
                pred_after = int(model(x_poison_eval).argmax(dim=1).item())
                buggy_fixed = (pred_after == int(info["y_clean"]))
                pred_changes.append((info["pred_before"], pred_after, info["y_clean"], buggy_fixed and ok))
                if buggy_fixed:
                    region_fixed_indices.add(info["idx"])


        print(
            f"  [CP-Repair][Region-Repair] evaluated fixed samples after training: "
            f"{len(region_fixed_indices)}/{len(region_infos)}"
        )

        if ok and len(region_fixed_indices) > 0:
            repaired_buggy_indices.update(region_fixed_indices)
            success += len(region_fixed_indices)
        else:
            repair_failed_count += max(1, len(region_infos))

    for info in tqdm(sample_infos, desc="Evaluating-Region-Repair", total=len(sample_infos)):
        if info["pre_skipped"]:
            if info["pre_success"]:
                success += 1
                pred_changes.append((info["pred_before"], info["pred_before"], info["y_clean"], True))
            else:
                pred_changes.append((info["pred_before"], info["pred_before"], info["y_clean"], False))
            continue

        if not info["needs_repair"]:
            pred_changes.append((info["pred_before"], info["pred_before"], info["y_clean"], False))
            continue

        if info["idx"] in repaired_buggy_indices:
            continue

        x_poison_eval = add_trigger_bottom_right(
            torch.tensor(info["x_clean"]).unsqueeze(0).to(device),
            trigger_size=dataset_config.trigger_size,
            num_channels=num_channels,
            trigger_value=255.0 if dataset_config.name == "GTSRB" else 1.0
        )
        model.eval()
        with torch.no_grad():
            pred_after = int(model(x_poison_eval).argmax(dim=1).item())

        buggy_fixed = (pred_after == int(info["y_clean"]))
        if buggy_fixed:
            success += 1
            pred_changes.append((info["pred_before"], pred_after, info["y_clean"], True))
            continue

        if info["route"] == "skip_low_sce":
            print(f"  [CP-Repair][SCE-Routing] skip sample {info['idx']}: base-k SCE={info['sce_score']:.4f} below q70={high70_threshold:.4f}")
        pred_changes.append((info["pred_before"], pred_after, info["y_clean"], False))

    repair_time = time.time() - repair_start

    rsr = 100.0 * success / len(buggy_X)
    avg_final_k = total_k_used / max(1, max(1, sum(1 for info in sample_infos if info.get('route') in {'direct', 'expand'} and info.get('needs_repair'))))
    print(f"[OK] Phase3 完成: repair_time={repair_time:.2f}s, RSR={rsr:.2f}% ({success}/{len(buggy_X)}), avg_final_k={avg_final_k:.2f}")
    if no_ref_count > 0:
        print(f"  [DEBUG] 找不到参考样本的样本数: {no_ref_count}/{len(buggy_X)}")
    if repair_failed_count > 0:
        print(f"  [DEBUG] 修复失败的区域/样本计数: {repair_failed_count}")

    # 分析预测变化
    if pred_changes:
        changed_count = sum(1 for pb, pa, _, _ in pred_changes if pb != pa)
        target_correct = sum(1 for _, pa, target, _ in pred_changes if pa == target)
        print(f"  [DEBUG] 预测发生变化的样本: {changed_count}/{len(pred_changes)}")
        print(f"  [DEBUG] 修复后预测正确的样本: {target_correct}/{len(pred_changes)}")

        # 统计最常见的预测变化模式
        change_patterns = {}
        for pb, pa, target, ok in pred_changes:
            pattern = f"{pb}->{pa}"
            change_patterns[pattern] = change_patterns.get(pattern, 0) + 1
        if change_patterns:
            top_patterns = sorted(change_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  [DEBUG] 最常见的预测变化模式: {top_patterns}")

    # 7. 评估修复后性能
    print(f"\n>>> [Step 6] 评估修复后性能...")
    acc_after = evaluate_acc(model, test_loader, device)
    sr_after = evaluate_sr(model, test_loader, device, dataset_config.target_label,
                          dataset_config.trigger_size, num_channels, dataset_config.name)
    print(f"  ACC (after): {acc_after*100:.2f}%")
    print(f"  SR (after): {sr_after*100:.2f}%")

    acc_delta = (acc_after - acc_before) * 100
    if acc_delta < -5.0:
        print(f"  [WARN] ACC下降超过5%，修复可能过于激进！")
    elif acc_delta < 0:
        print(f"  [INFO] ACC略有下降，但在可接受范围内")

    print("\n" + "=" * 80)
    print(f">>> [Summary] {dataset_config.name}")
    print("=" * 80)
    print(f"ACC:  {acc_before*100:.2f}% -> {acc_after*100:.2f}%  (delta={acc_delta:+.2f}%)")
    print(f"SR:   {sr_before*100:.2f}% -> {sr_after*100:.2f}%  (delta={((sr_after-sr_before)*100):+.2f}%)")
    print(f"RSR:  {rsr:.2f}%")
    print(f"Time: loc={loc_time:.2f}s, repair={repair_time:.2f}s, total={(loc_time+repair_time):.2f}s")

    total_trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    modified_params = int(sum(int(np.sum(m)) if isinstance(m, np.ndarray) else int(np.sum(np.asarray(m))) for m in pathway_masks))
    return {
        "dataset": dataset_config.name,
        "acc_before": acc_before * 100,
        "acc_after": acc_after * 100,
        "sr_before": sr_before * 100,
        "sr_after": sr_after * 100,
        "rsr": rsr,
        "loc_time": loc_time,
        "repair_time": repair_time,
        "pathway_masks": pathway_masks,
        "modified_params": modified_params,
        "total_trainable_params": total_trainable_params,
    }


# ==================== CCBR Bridge: PyTorch → Keras .h5 ====================

def pytorch_model_to_npz(pt_model: nn.Module, save_path: str) -> str:
    """
    将 PyTorch 模型的所有 Conv2d/Linear 层权重保存为 .npz 文件。
    供子进程加载后写入 Keras 模型，避免主进程混用 TF/PyTorch CUDA context。
    Conv2d weight: (out, in, H, W)
    Linear weight: (out, in)
    bias: (out,)
    """
    weights = {}
    idx = 0
    for m in pt_model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            weights[f'w_{idx}'] = m.weight.data.cpu().numpy()
            weights[f'b_{idx}'] = m.bias.data.cpu().numpy()
            idx += 1
    np.savez(save_path, **weights)
    print(f"[CCBR-Bridge] PyTorch 权重已保存为 npz: {save_path}")
    return save_path


def pytorch_model_to_keras_h5(pt_model: nn.Module, dataset_name: str, save_path: str) -> str:
    """
    将 PyTorch 模型权重保存为 npz，供子进程转换为 Keras .h5。
    主进程只做 numpy 操作，不调用 TF，避免 CUDA context 冲突。
    实际的 Keras 模型构建和 .h5 保存在子进程中完成。
    """
    npz_path = save_path.replace('.h5', '_weights.npz')
    pytorch_model_to_npz(pt_model, npz_path)
    # 记录数据集名，供子进程选择正确的 Keras 架构
    meta_path = save_path.replace('.h5', '_meta.txt')
    with open(meta_path, 'w') as f:
        f.write(dataset_name)
    print(f"[CCBR-Bridge] 权重导出完成，子进程将在运行时构建 Keras .h5")
    return npz_path


def _build_keras_gtsrb():
    """构建与 NN1_GTSRB 对应的 Keras 模型（不编译，只用于权重传递）"""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        Conv2D(32, (3, 3), padding='valid', activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='valid', activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='valid', activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(43)])
    return model


def _build_keras_mnist():
    """构建与 NN2_MNIST 对应的 Keras 模型"""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
    model = Sequential([
        Conv2D(16, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(32, (5, 5), padding='same', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(10)])
    return model


def _build_keras_fashion():
    """构建与 NN3_Fashion 对应的 Keras 模型"""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
    model = Sequential([
        Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), padding='valid', activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(128, (3, 3), padding='valid', activation='relu'),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10)])
    return model


def run_ccbr_repair(*args, **kwargs) -> Dict[str, Any]:
    raise NotImplementedError("CCBR bridge has been removed from the CP-Repair backdoor pipeline")


    # CCBR 期望 channels_last 格式 (N, H, W, C)
    if X_test_np.ndim == 4 and X_test_np.shape[1] in (1, 3):
        X_test_nhwc = np.transpose(X_test_np, (0, 2, 3, 1))
    else:
        X_test_nhwc = X_test_np

    np.save(x_test_path, X_test_nhwc)
    np.save(y_test_path, Y_test_np)

    # 触发器图像文件名前缀：与各数据集原始代码保持一致
    img_prefix = {"GTSRB": "gtsrb", "MNIST": "mnist", "Fashion-MNIST": "fashion", "CIFAR-10": "cifar"}.get(dataset_config.name, "mnist")

    trigger_value_str = "255.0" if dataset_config.name == "GTSRB" else "1.0"

    runner_code = f'''# -*- coding: utf-8 -*-
# 自动生成的 CCBR 临时运行脚本
import os, sys, json, time
import numpy as np

# 确保 care-main 的模块可以被导入
sys.path.insert(0, r"{ccbr_source_dir.replace(chr(92), chr(92)+chr(92))}")
os.chdir(r"{ccbr_source_dir.replace(chr(92), chr(92)+chr(92))}")

import warnings
warnings.filterwarnings(\'ignore\')
os.environ[\'TF_CPP_MIN_LOG_LEVEL\'] = \'3\'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

sys.path.insert(0, r"{ccbr_source_dir.replace(chr(92), chr(92)+chr(92))}")
from causal_inference import causal_analyzer
import utils_backdoor

# causal_inference.py 内部用相对路径 'results/' 读取 mask/pattern
# 必须与模块级 RESULT_DIR 常量保持一致（即 chdir 后的 results/ 子目录）
RESULT_DIR = '../results' if '{dataset_config.name}' == 'CIFAR-10' else 'results'
os.makedirs(RESULT_DIR, exist_ok=True)
# 同时把绝对路径也创建好（供外部检查）
os.makedirs(r"{result_dir_ccbr.replace(chr(92), chr(92)+chr(92))}", exist_ok=True)

# ---- 加载数据 ----
X_test = np.load(r"{x_test_path.replace(chr(92), chr(92)+chr(92))}").astype(np.float32)
Y_test_int = np.load(r"{y_test_path.replace(chr(92), chr(92)+chr(92))}")

# CCBR 使用 one-hot 标签
Y_test = to_categorical(Y_test_int, {dataset_config.num_classes})

# GTSRB raw: 不做归一化；其他归一化到[0,1]（已在PyTorch侧完成）
# 此处 X_test 已经是正确范围

NUM_CLASSES = {dataset_config.num_classes}
IMG_ROWS = {img_rows}
IMG_COLS = {img_cols}
IMG_COLOR = {num_channels}
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, IMG_COLOR)
BATCH_SIZE = 32
NB_SAMPLE = 1000
MINI_BATCH = NB_SAMPLE // BATCH_SIZE
PSO_BATCH = 1000 // BATCH_SIZE
INTENSITY_RANGE = \'{intensity_range}\'
Y_TARGET = {y_target}
SPLIT_LAYER = {split_layer}
REP_N = {rep_n}

# ---- 从 npz 构建 Keras 模型并加载 PyTorch 权重 ----
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

def _build_keras_arch():
    if '{dataset_config.name}' == 'GTSRB':
        return Sequential([
            Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(32,32,3)),
            Conv2D(32,(3,3),padding='valid',activation='relu'),
            MaxPooling2D((2,2)),
            Conv2D(64,(3,3),padding='same',activation='relu'),
            Conv2D(64,(3,3),padding='valid',activation='relu'),
            MaxPooling2D((2,2)),
            Conv2D(128,(3,3),padding='same',activation='relu'),
            Conv2D(128,(3,3),padding='valid',activation='relu'),
            MaxPooling2D((2,2)),
            Flatten(), Dense(512,activation='relu'), Dense(43)])
    elif '{dataset_config.name}' == 'MNIST':
        return Sequential([
            Conv2D(16,(5,5),padding='same',activation='relu',input_shape=(28,28,1)),
            MaxPooling2D((2,2)), Dropout(0.2),
            Conv2D(32,(5,5),padding='same',activation='relu'),
            MaxPooling2D((2,2)), Dropout(0.2),
            Flatten(), Dense(512,activation='relu'), Dropout(0.5), Dense(10)])
    elif '{dataset_config.name}' == 'CIFAR-10':
        return Sequential([
            Conv2D(32,(3,3),padding='valid',activation='relu',input_shape=(32,32,3)),
            Conv2D(32,(3,3),padding='valid',activation='relu'),
            MaxPooling2D((2,2)),
            Conv2D(64,(3,3),padding='valid',activation='relu'),
            Conv2D(64,(3,3),padding='valid',activation='relu'),
            MaxPooling2D((2,2)),
            Flatten(), Dense(512,activation='relu'), Dense(10)])
    else:  # Fashion-MNIST
        return Sequential([
            Conv2D(32,(3,3),padding='valid',activation='relu',input_shape=(28,28,1)),
            MaxPooling2D((2,2)), Dropout(0.2),
            Conv2D(64,(3,3),padding='valid',activation='relu'),
            MaxPooling2D((2,2)), Dropout(0.2),
            Conv2D(128,(3,3),padding='valid',activation='relu'), Dropout(0.2),
            Flatten(), Dense(128,activation='relu'), Dense(10)])

model = _build_keras_arch()
# 用一次 dummy forward 触发权重初始化
model(np.zeros((1, IMG_ROWS, IMG_COLS, IMG_COLOR), dtype=np.float32))

# 从 npz 加载 PyTorch 权重并转换为 Keras 格式
npz_path = r"{npz_path.replace(chr(92), chr(92)+chr(92))}"
print(f\"[CCBR] 从 npz 加载 PyTorch 权重: {{npz_path}}\")
wd = np.load(npz_path)
keras_layers = [l for l in model.layers if hasattr(l, 'kernel')]

# 找出 Flatten 层之前的输出形状（即 PyTorch flatten 前的特征图尺寸）
# 通过 dummy forward 逐层追踪，记录 Flatten 前一层的输出
import tensorflow as _tf
dummy_in = _tf.constant(np.zeros((1, IMG_ROWS, IMG_COLS, IMG_COLOR), dtype=np.float32))
cur = dummy_in
pre_flatten_shape = None  # (H, W, C) in Keras = last shape before Flatten
for l in model.layers:
    prev_shape = cur.shape[1:]  # 记录进入该层前的形状
    cur = l(cur)
    if isinstance(l, type(model.layers[0]).__mro__[0]) or l.__class__.__name__ == 'Flatten':
        if l.__class__.__name__ == 'Flatten':
            pre_flatten_shape = tuple(int(x) for x in prev_shape)  # (H, W, C)
            break

if pre_flatten_shape is None:
    # fallback: 手动 forward 找 Flatten
    cur2 = _tf.constant(np.zeros((1, IMG_ROWS, IMG_COLS, IMG_COLOR), dtype=np.float32))
    for l in model.layers:
        if l.__class__.__name__ == 'Flatten':
            break
        cur2 = l(cur2)
    pre_flatten_shape = tuple(int(x) for x in cur2.shape[1:])  # (H, W, C)

print(f\"[CCBR] Flatten 前形状 (H,W,C): {{pre_flatten_shape}}\")
H_f, W_f, C_f = pre_flatten_shape[0], pre_flatten_shape[1], pre_flatten_shape[2]

# GTSRB NN1 uses permute(0,2,3,1) in forward() before flatten, so the
# PyTorch Linear weights are already laid out in NHWC order (512->512).
# No reshape/permute is needed for the first Dense layer in that case.
DATASET_NAME = '{dataset_config.name}'
first_dense = True
for i, kl in enumerate(keras_layers):
    w = wd[f'w_{{i}}']
    b = wd[f'b_{{i}}']
    if w.ndim == 4:  # Conv: PyTorch (out,in,H,W) -> Keras (H,W,in,out)
        w = np.transpose(w, (2,3,1,0))
    else:            # Linear: PyTorch (out,in) -> Keras (in,out)
        if first_dense and DATASET_NAME != 'GTSRB':
            # PyTorch flatten: (C,H,W) -> C*H*W; Keras flatten: (H,W,C) -> H*W*C
            # Need to re-permute so the weight rows match Keras NHWC flatten order.
            # w shape: (out_features, C_f*H_f*W_f)
            out_features = w.shape[0]
            w = w.reshape(out_features, C_f, H_f, W_f)  # (out, C, H, W)
            w = np.transpose(w, (0, 2, 3, 1))           # (out, H, W, C)
            w = w.reshape(out_features, H_f*W_f*C_f)    # (out, H*W*C)
        if first_dense:
            first_dense = False
        w = np.transpose(w, (1, 0))  # (in, out)
    kl.set_weights([w, b])
print(f\"[CCBR] 权重加载完成，共 {{len(keras_layers)}} 个可训练层\")

model_path = r"{h5_path.replace(chr(92), chr(92)+chr(92))}"
print(f\"[CCBR] 加载 Keras 模型: {{model_path}}\")
model.compile(optimizer=\'adam\', loss=\'categorical_crossentropy\', metrics=[\'accuracy\'])
model.summary()

# ---- 构建数据生成器 ----
def build_data_loader(X, Y, batch_size=BATCH_SIZE):
    datagen = ImageDataGenerator()
    return datagen.flow(X, Y, batch_size=batch_size)

# 测试集：随机采样 1000 个用于 PSO 评估
idx = np.random.permutation(len(X_test))[:NB_SAMPLE]
x_pso = X_test[idx]
y_pso = Y_test[idx]

test_generator = build_data_loader(X_test, Y_test)
pso_generator  = build_data_loader(x_pso, y_pso)

# ---- 添加触发器函数（用于评估 SR）----
def add_trigger(X, trigger_size={trigger_size}, trigger_val={trigger_value_str}):
    X_p = X.copy()
    X_p[:, -trigger_size:, -trigger_size:, :] = trigger_val
    return X_p

# ---- 修复前评估 ----
X_non_target = X_test[Y_test_int != Y_TARGET]
Y_non_target = Y_test[Y_test_int != Y_TARGET]
X_triggered = add_trigger(X_non_target)

preds_clean  = np.argmax(model.predict(X_test,  verbose=0), axis=1)
preds_poison = np.argmax(model.predict(X_triggered, verbose=0), axis=1)

acc_before = np.mean(preds_clean == Y_test_int)
sr_before  = np.mean(preds_poison == Y_TARGET)
print(f\"[CCBR] Before: ACC={{acc_before:.4f}}, SR={{sr_before:.4f}}\")

# ---- 构造 perturbed 图像文件（CCBR 读取触发器 mask/pattern PNG）----
# 若 results 目录下没有预先保存的 mask/pattern，
# 则直接用硬触发器策略：构造一个 dummy mask PNG
import os
from PIL import Image as PILImage

mask_path    = os.path.join(RESULT_DIR, f\"{img_prefix}_visualize_mask_label_{{Y_TARGET}}.png\")
pattern_path = os.path.join(RESULT_DIR, f\"{img_prefix}_visualize_pattern_label_{{Y_TARGET}}.png\")

# 如果触发器图像不存在，生成一个简单的硬编码触发器
if not os.path.exists(mask_path):
    os.makedirs(RESULT_DIR, exist_ok=True)
    mask_arr = np.zeros((IMG_ROWS, IMG_COLS), dtype=np.uint8)
    mask_arr[-{trigger_size}:, -{trigger_size}:] = 255
    PILImage.fromarray(mask_arr, mode='L').save(mask_path)
    print(f\"[CCBR] 生成 mask: {{mask_path}}\")
if not os.path.exists(pattern_path):
    pattern_arr = np.zeros((IMG_ROWS, IMG_COLS, IMG_COLOR), dtype=np.uint8)
    pattern_arr[-{trigger_size}:, -{trigger_size}:, :] = 255
    if IMG_COLOR == 1:
        PILImage.fromarray(pattern_arr[:, :, 0], mode='L').convert('RGB').save(pattern_path)
    else:
        PILImage.fromarray(pattern_arr, mode='RGB').save(pattern_path)
    print(f\"[CCBR] 生成 pattern: {{pattern_path}}\")

# ---- 初始化 causal_analyzer ----
print(\"[CCBR] 初始化 causal_analyzer...\")
analyzer = causal_analyzer(
    model,
    test_generator,
    pso_generator,
    input_shape=INPUT_SHAPE,
    init_cost=1e-3,
    steps=1,
    lr=0.1,
    num_classes=NUM_CLASSES,
    mini_batch=MINI_BATCH,
    pso_batch=PSO_BATCH,
    patience=5,
    cost_multiplier=2,
    img_color=IMG_COLOR,
    batch_size=BATCH_SIZE,
    verbose=2,
    save_last=False,
    early_stop=True,
    early_stop_threshold=1.0,
    early_stop_patience=25,
)
analyzer.SPLIT_LAYER = SPLIT_LAYER
analyzer.rep_n = REP_N
analyzer.target = Y_TARGET

# ---- 运行修复 ----
start_t = time.time()
analyzer.analyze_counterfactual_expectGradient()
repair_time = time.time() - start_t
print(f\"[CCBR] 修复耗时: {{repair_time:.2f}}s\")

# ---- 修复后评估 ----
# 选择最优权重
if analyzer.r_weight is not None and len(np.array(analyzer.r_weight).shape) >= 1:
    rw = np.array(analyzer.r_weight)
    if rw.ndim == 1:
        best_weight = rw
    else:
        best_p = 0
        best_sr = 1.0
        best_acc = 0.0
        for i in range(len(rw)):
            sr_i, acc_i = analyzer.pso_test(rw[i], Y_TARGET)
            if sr_i < best_sr or (sr_i == best_sr and acc_i > best_acc):
                best_p = i
                best_sr = sr_i
                best_acc = acc_i
        best_weight = rw[best_p]
else:
    best_weight = []

sr_after, acc_after = analyzer.pso_test(best_weight, Y_TARGET)
print(f\"[CCBR] After:  ACC={{acc_after:.4f}}, SR={{sr_after:.4f}}\")

# ---- 保存结果 ----
result = {{
    \"dataset\": \"{dataset_config.name}\",
    \"algorithm\": \"ccbr\",
    \"acc_before\": float(acc_before) * 100,
    \"acc_after\": float(acc_after) * 100,
    \"sr_before\": float(sr_before) * 100,
    \"sr_after\": float(sr_after) * 100,
    \"repair_time\": repair_time,
    \"best_weight\": best_weight.tolist() if hasattr(best_weight, 'tolist') else list(best_weight),
}}
with open(r"{ccbr_result_json.replace(chr(92), chr(92)+chr(92))}", 'w') as f:
    json.dump(result, f, indent=2)
print(f\"[CCBR] 结果已保存: {ccbr_result_json}\")
'''

    # 写入临时脚本
    os.makedirs(results_dir, exist_ok=True)
    with open(runner_script, 'w', encoding='utf-8') as f:
        f.write(runner_code)
    print(f"[CCBR] 临时脚本已写入: {runner_script}")

    # --- Step 3: 以子进程方式运行（避免 TF1.x session 与 PyTorch CUDA 冲突）---
    import subprocess
    print(f"[CCBR] 启动子进程运行 CCBR 修复（实时输出）...")
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            [sys.executable, runner_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
        )
        # 实时打印子进程输出，避免误认为卡死
        for line in proc.stdout:
            print(f"[CCBR] {line}", end="")
        proc.wait(timeout=7200)  # 最多等 2 小时
        elapsed = time.time() - t0
        print(f"[CCBR] 子进程完成，耗时 {elapsed:.1f}s，返回码 {proc.returncode}")
    except subprocess.TimeoutExpired:
        proc.kill()
        print("[CCBR] 子进程超时（>2h），已终止")
        return {}
    except Exception as e:
        print(f"[CCBR] 子进程异常: {e}")
        return {}

    # --- Step 4: 读取结果 ---
    if not os.path.exists(ccbr_result_json):
        print(f"[CCBR] 结果文件未生成: {ccbr_result_json}")
        return {}
    import json as _json
    with open(ccbr_result_json, 'r', encoding='utf-8') as f:
        result = _json.load(f)
    print(f"[CCBR] 结果: ACC {result['acc_before']:.2f}% -> {result['acc_after']:.2f}%, "
          f"SR {result['sr_before']:.2f}% -> {result['sr_after']:.2f}%")
    return result


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """获取数据集配置"""
    # Fashion-MNIST 自训练模型保存路径
    models_dir = os.path.join(my_algorithm_dir, "experiments", "models")
    os.makedirs(models_dir, exist_ok=True)
    # CCBR care-main 数据根目录
    data_dir = os.path.join(care_main_dir, 'benchmark', 'benchmark', 'data')

    if dataset_name == "MNIST":
        H, W = 28, 28
        trigger_size = calculate_trigger_size((H, W))
        return DatasetConfig(
            name="MNIST",
            num_classes=10,
            input_size=(1, H, W),
            target_label=3,
            trigger_size=trigger_size,
            model_class=NN2_MNIST,
            data_path=data_dir,
            model_path='',
            backdoor_model_path=os.path.join(data_dir, 'mnist_backdoor_retrained.pth'),
        )
    elif dataset_name == "Fashion-MNIST":
        H, W = 28, 28
        trigger_size = calculate_trigger_size((H, W))
        return DatasetConfig(
            name="Fashion-MNIST",
            num_classes=10,
            input_size=(1, H, W),
            target_label=3,
            trigger_size=trigger_size,
            model_class=NN3_Fashion,
            data_path=data_dir,
            model_path='',
            backdoor_model_path=os.path.join(data_dir, 'fashion_mnist_backdoor_retrained.pth'),
        )
    elif dataset_name == "GTSRB":
        H, W = 32, 32
        trigger_size = calculate_trigger_size((H, W))
        return DatasetConfig(
            name="GTSRB",
            num_classes=43,
            input_size=(3, H, W),
            target_label=33,  # CCBR gtsrb_bottom_right_white_4_target_33.h5
            trigger_size=trigger_size,  # 实测 5x5 SR=90.97%
            model_class=NN1_GTSRB,
            data_path=data_dir,
            model_path='',    # CCBR .h5 直接加载，不使用此路径
            backdoor_model_path=os.path.join(data_dir, 'gtsrb_backdoor_retrained.pth'),
        )
    elif dataset_name == "CIFAR-10":
        H, W = 32, 32
        trigger_size = calculate_trigger_size((H, W))
        cifar_root = os.path.join(care_main_dir, 'benchmark', 'benchmark', 'cifar_nnrepair')
        return DatasetConfig(
            name="CIFAR-10",
            num_classes=10,
            input_size=(3, H, W),
            target_label=7,
            trigger_size=trigger_size,
            model_class=NN4_CIFAR,
            data_path=os.path.join(cifar_root, 'data'),
            model_path='',
            backdoor_model_path=os.path.join(cifar_root, 'data', 'cifar10_backdoor_retrained.pth'),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: MNIST, Fashion-MNIST, GTSRB, CIFAR-10")


def main():
    parser = argparse.ArgumentParser(description="CP-Repair 多数据集后门移除实验")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["MNIST", "Fashion-MNIST", "GTSRB", "CIFAR-10"],
                       help="选择数据集: MNIST, Fashion-MNIST, GTSRB, 或 CIFAR-10")
    parser.add_argument("--device", type=str, default="cuda", help="运行设备 (cpu/cuda)")
    parser.add_argument("--algorithm", type=str, default="cprepair",
                       choices=["cprepair"],
                       help="修复算法: cprepair（默认）")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "no_localization", "no_verification", "no_pathway_constraint"],
                        help="RQ3 ablation modes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rq3_output", type=str, default=os.path.join("PiVR", "experiments", "results", "rq4_ablation_result.csv"))
    parser.add_argument("--rq5_repair_sensitivity", action="store_true", help="启用RQ5修复阶段参数敏感性分析模式")
    parser.add_argument("--rq5_param", type=str, default=None, choices=["eta", "lambda_clean", "lambda_task"])
    parser.add_argument("--rq5_value", type=float, default=None)
    parser.add_argument("--rq5_output", type=str, default=os.path.join("PiVR", "experiments", "results", "rq5_repair_sensitivity_result.csv"))
    parser.add_argument("--sweep_mode", action="store_true", help="启用参数敏感性分析模式")
    parser.add_argument("--alpha", type=float, default=PATHWAY_CONFIG['alpha'])
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--sbfl_strategy", type=str, default=BACKDOOR_UNIFIED_CONFIG['sfl_strategy'])
    parser.add_argument("--layer_k_ratio_cap", type=float, default=PATHWAY_CONFIG['layer_k_ratio_cap'])
    parser.add_argument("--activation_threshold", type=float, default=PATHWAY_CONFIG['activation_threshold'])
    parser.add_argument("--lambda_clean", type=float, default=BACKDOOR_UNIFIED_CONFIG['lambda_clean'])

    args = parser.parse_args()

    rq5_mode = bool(args.rq5_repair_sensitivity)
    rq5_param_name = args.rq5_param
    rq5_param_value = args.rq5_value

    print("=" * 80)
    print("CP-Repair 多数据集后门移除实验 (Backdoor Removal)")
    print("=" * 80)
    print(f"选择的数据集: {args.dataset}")
    print(f"选择的算法:   {args.algorithm}")
    print(f"消融模式:     {args.ablation}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，回退到 CPU")
        device = "cpu"

    # CP-Repair 超参数：统一配置 + 数据集覆盖 + sweep 覆盖
    backdoor_hp = dict(BACKDOOR_UNIFIED_CONFIG)
    backdoor_hp.update({
        'activation_threshold': PATHWAY_CONFIG['activation_threshold'],
        'pathway_alpha': PATHWAY_CONFIG['alpha'],
        'layer_k_ratio_cap': PATHWAY_CONFIG['layer_k_ratio_cap'],
    })
    backdoor_hp.update(BACKDOOR_DATASET_OVERRIDES.get(args.dataset, {}))
    if args.alpha is not None:
        backdoor_hp['pathway_alpha'] = args.alpha
    if args.k is not None:
        backdoor_hp['top_k'] = args.k
    if args.sbfl_strategy is not None:
        backdoor_hp['sfl_strategy'] = args.sbfl_strategy
    if args.layer_k_ratio_cap is not None:
        backdoor_hp['layer_k_ratio_cap'] = args.layer_k_ratio_cap
    if args.activation_threshold is not None:
        backdoor_hp['activation_threshold'] = args.activation_threshold
    if args.lambda_clean is not None:
        backdoor_hp['lambda_clean'] = args.lambda_clean

    hyper_params = HyperParams(
        sfl_strategy=backdoor_hp['sfl_strategy'],
        top_k=backdoor_hp['top_k'],
        repair_epochs=backdoor_hp['repair_epochs'],
        repair_lr=backdoor_hp['repair_lr'],
        repair_lambda=backdoor_hp['repair_lambda'],
        lambda_clean=backdoor_hp['lambda_clean'],
        max_buggy_samples=backdoor_hp['max_buggy_samples'],
    )


    # 获取数据集配置
    dataset_config = get_dataset_config(args.dataset)

    # 保存结果
    results_dir = os.path.join(my_algorithm_dir, "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = args.rq5_output if rq5_mode else (args.rq3_output if args.rq3_output else os.path.join(results_dir, "backdoor_multi_result.csv"))
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)

    # ================================================================
    # CP-Repair 分支
    # ================================================================
    if args.algorithm in ("cprepair", "all"):
        # 统一逻辑：使用数据集预设的单一 top_k（不再遍历 range(top_k)）
        top_k = hyper_params.top_k

        print("\n" + "=" * 80)
        print("=" * 80)

        try:
            result = run_experiment(
                dataset_config, device,
                sfl_strategy=hyper_params.sfl_strategy,
                top_k=top_k,
                repair_epochs=hyper_params.repair_epochs,
                repair_lr=hyper_params.repair_lr,
                repair_lambda=hyper_params.repair_lambda,
                max_buggy_samples=hyper_params.max_buggy_samples,
                lambda_clean=hyper_params.lambda_clean,
                pathway_alpha=backdoor_hp['pathway_alpha'],
                pathway_activation_threshold=backdoor_hp['activation_threshold'],
                layer_k_ratio_cap=backdoor_hp['layer_k_ratio_cap'],
                ablation=args.ablation,
                seed=args.seed,
                rq5_mode=rq5_mode,
                rq5_param_name=rq5_param_name,
                rq5_param_value=rq5_param_value,
            )

            repair_metric_name = 'ASR'
            repair_metric_before = result['sr_before']
            repair_metric_after = result['sr_after']
            acc_before = result['acc_before']
            acc_after = result['acc_after']
            drawdown = acc_before - acc_after
            modified_params = int(result.get('modified_params', 0)) if isinstance(result, dict) else 0
            total_trainable_params = int(result.get('total_trainable_params', 0)) if isinstance(result, dict) else 0
            modified_params_ratio = (modified_params / total_trainable_params) if total_trainable_params > 0 else 0.0
            if rq5_mode:
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if not file_exists:
                        w.writerow([
                            'timestamp', 'task', 'model', 'dataset', 'protected_attr', 'seed',
                            'param_name', 'param_value', 'repair_metric_name', 'repair_metric_before', 'repair_metric_after',
                            'acc_before', 'acc_after',
                            'loc_time', 'ver_time', 'repair_time', 'total_time'
                        ])
                        file_exists = True
                    w.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Backdoor', 'NN2', result['dataset'], 'none', args.seed,
                        rq5_param_name, f"{rq5_param_value:.6g}",
                        repair_metric_name, f"{repair_metric_before:.4f}", f"{repair_metric_after:.4f}",
                        f"{acc_before:.4f}", f"{acc_after:.4f}",
                        f"{result['loc_time']:.2f}", f"{0.0:.2f}", f"{result['repair_time']:.2f}", f"{(result['loc_time'] + result['repair_time']):.2f}",
                    ])
            else:
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    if not file_exists:
                        w.writerow([
                            'timestamp', 'task', 'model', 'dataset', 'protected_attr', 'ablation', 'seed',
                            'repair_metric_name', 'repair_metric_before', 'repair_metric_after',
                            'acc_before', 'acc_after',
                            'modified_params', 'modified_params_ratio',
                            'loc_time', 'ver_time', 'repair_time', 'total_time'
                        ])
                        file_exists = True
                    w.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'Backdoor', 'NN2', result['dataset'], 'none', args.ablation, args.seed,
                        repair_metric_name, f"{repair_metric_before:.4f}", f"{repair_metric_after:.4f}",
                        f"{acc_before:.4f}", f"{acc_after:.4f}",
                        modified_params, f"{modified_params_ratio:.6f}",
                        f"{result['loc_time']:.2f}", f"{0.0:.2f}", f"{result['repair_time']:.2f}", f"{(result['loc_time'] + result['repair_time']):.2f}",
                    ])

            print(f"\n[OK] CP-Repair  结果已保存: {csv_path}")

        except Exception as e:
            print(f"\n[ERROR] CP-Repair 实验失败: {e}")
            import traceback
            traceback.print_exc()



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

