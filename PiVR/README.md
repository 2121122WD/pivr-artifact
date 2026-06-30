# PiVR

PiVR 是本仓库中用于模型修复与实验复现的主要代码目录，当前已经整理出三类实验入口，分别对应安全性、后门修复和公平性三个任务。

## 主要实验入口

- **安全性实验**：`experiments/exp_safety_acas.py`
- **后门修复实验**：`experiments/exp_backdoor_removal_multi.py`
- **公平性实验**：`Socrates/source/run_fairness_cprepair_benchmark.py`


## PiVR 流程概览

PiVR 当前实现可以概括为三步：

1. **Pathway Localization**：定位可疑神经元或路径。
2. **Causal Verification**：验证可疑区域是否与错误行为相关。
3. **Imitation Repair**：根据参考样本执行修复。


## 环境依赖

项目主要依赖 Python 3.10.14，以及以下核心包：

安装方式：

```bash
pip install -r requirements.txt
```

## 运行方式

### 安全性实验

```bash
python experiments/exp_safety_acas.py --subnetwork "N2,9"
```

### 后门修复实验

```bash
python experiments/exp_backdoor_removal_multi.py --dataset MNIST
```

### 公平性实验

```bash
python Socrates/source/run_fairness_pivr_benchmark.py --dataset bank --attribute age
```

