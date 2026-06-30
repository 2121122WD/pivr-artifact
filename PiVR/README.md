# PiVR Artifact

This repository provides the anonymized artifact for **PiVR: Pathway-Guided Neural Network Repair via Intervention Verification**.

Anonymous artifact link:

```text
https://anonymous.4open.science/r/pivr-artifact-7FD3
```

PiVR is a pathway-guided neural network repair framework. It repairs defective neural networks through three main stages:

1. **Repair-Oriented Pathway Localization**
2. **Intervention-Based Verification**
3. **Pathway-Constrained Repair**

The current artifact includes the PiVR source code, experiment entry scripts, utility modules, dependency file, and result files for inspection. Large benchmark datasets and pretrained models are **not included** in this repository because they are large and may exceed practical repository size limits.

## Repository Structure

```text
PiVR/
├── experiments/
│   ├── exp_safety_acas.py
│   ├── exp_backdoor_removal_multi.py
│   └── run_fairness_pivr_benchmark.py
├── lrp_src/
├── methods/
│   ├── pathway.py
│   ├── verifier.py
│   └── repair.py
├── benchmark/
│   └── benchmark/        # not included in the artifact; see instructions below
├── README.md
├── requirements.txt
└── utils.py
```

## Main Experiment Entry Points

The artifact contains three experiment entry scripts:

- **Safety repair**: `experiments/exp_safety_acas.py`
- **Backdoor removal**: `experiments/exp_backdoor_removal_multi.py`
- **Fairness repair**: `experiments/run_fairness_pivr_benchmark.py`

## Benchmark Data and Model Files

The benchmark datasets and pretrained models are not uploaded to this anonymous repository due to file-size constraints. To reproduce the full experiments, please place the required benchmark files under the following directory structure:

```text
PiVR/
└── benchmark/
    └── benchmark/
        ├── acas_N19/
        ├── acas_N29/
        ├── acas_N33/
        ├── cifar_nnrepair/
        ├── gtsrb_nnrepair/
        ├── mnist_nnrepair/
        ├── fmnist_nnrepair/
        └── causal/
```

The expected locations are:

- Backdoor datasets and pretrained backdoored models:
  - `PiVR/benchmark/benchmark/`
- ACAS Xu safety datasets and models:
  - `PiVR/benchmark/benchmark/`
- Fairness datasets and models:
  - `PiVR/benchmark/benchmark/causal/`

After placing the benchmark files in these directories, the experiment scripts can load them using relative paths.

## Environment

The experiments were developed with:

```text
Python 3.10.14
PyTorch
NumPy
SciPy
scikit-learn
pandas
h5py
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

A CUDA-enabled GPU is recommended for the vision repair experiments.

## How to Run

Run the following commands from the `PiVR/` directory.

### Safety Repair

```bash
python experiments/exp_safety_acas.py --subnetwork "N2,9"
```

Other evaluated ACAS Xu subjects include `N3,3` and `N1,9`, depending on the benchmark files placed under `benchmark/benchmark/`.

### Backdoor Removal

```bash
python experiments/exp_backdoor_removal_multi.py --dataset GTSRB
```

Other supported datasets include `MNIST`, `Fashion-MNIST`, and `CIFAR-10`, depending on the available benchmark files.

### Fairness Repair

```bash
python experiments/run_fairness_pivr_benchmark.py --dataset bank --attribute age
```

Other fairness settings depend on the tabular datasets and models placed under `benchmark/benchmark/causal/`.

## Result Files

This artifact may include result files used for checking the reported experimental outcomes. These files are provided to help reviewers inspect the main results even when the large benchmark files are not included.

## Notes for Double-Anonymous Review

This repository has been prepared for double-anonymous review. It is intended to contain only anonymized code, scripts, configurations, result files, and reproduction instructions. The benchmark datasets and pretrained models are omitted because of size constraints.