"""
CP-Repair hyperparameter configuration.

This module keeps only the parameters used by the current paper-aligned
pipeline: Localization -> Intervention Verification -> Pathway-Constrained
Repair.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


# ========== Helper APIs ==========

def get_subject_k(subject_name: str, default: Optional[int] = None) -> Optional[int]:
    """Return the default pathway budget for a subject/model name."""
    return SUBJECT_K_DEFAULTS.get(subject_name, default)


def merge_task_config(base_config: Mapping[str, Any], overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Merge a base config with explicit overrides while skipping None values."""
    merged = dict(base_config)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
    return merged


# ========== Localization ==========
PATHWAY_CONFIG = {
    "alpha": 0.7,
    "layer_k_ratio_cap": 1.0,
    "activation_threshold": 0.0,
    "sbfl_strategy": "barinel",
}

# Backward-compatible alias used by some experiment entrypoints.
SBFL_CONFIG = {
    "strategy": PATHWAY_CONFIG["sbfl_strategy"],
}

SUBJECT_K_DEFAULTS = {
    # Backdoor subjects
    "GTSRB": 12,
    "MNIST": 6,
    "Fashion-MNIST": 4,
    "CIFAR-10": 1,

    # Safety subjects
    "ACAS_Xu_N2_9": 2,
    "ACAS_Xu_N3_3": 6,
    "ACAS_Xu_N1_9": 8,

    # Fairness subjects
    "bank": 1,
    "census": 4,
    "credit": 3,
}

# ========== Intervention Verification ==========
VERIFICATION_CONFIG = {
    "reference_k": 5,
    "intervention_threshold": 0.0,
    "sce_repair_top_ratio": 1.0,
    "reference_metric": "feature_cosine",
}

# ========== Repair ==========
REPAIR_SHARED_CONFIG = {
    "early_stop_patience": 30,
}

# ========== Backdoor ==========
BACKDOOR_UNIFIED_CONFIG = {
    "sfl_strategy": PATHWAY_CONFIG["sbfl_strategy"],
    "top_k": get_subject_k("MNIST", 6),
    "repair_epochs": 200,
    "repair_lr": 0.002,
    "repair_lambda": 0.5,
    "lambda_clean": 0.5,
    "max_buggy_samples": 150,
    "reference_k": VERIFICATION_CONFIG["reference_k"],
    "intervention_threshold": VERIFICATION_CONFIG["intervention_threshold"],
    "sce_repair_top_ratio": VERIFICATION_CONFIG["sce_repair_top_ratio"],
    "reference_metric": VERIFICATION_CONFIG["reference_metric"],
    "early_stop_patience": REPAIR_SHARED_CONFIG["early_stop_patience"],
}

BACKDOOR_DATASET_OVERRIDES = {
    "GTSRB": {"top_k": get_subject_k("GTSRB")},
    "MNIST": {"top_k": get_subject_k("MNIST")},
    "Fashion-MNIST": {"top_k": get_subject_k("Fashion-MNIST")},
    "CIFAR-10": {"top_k": get_subject_k("CIFAR-10")},
}

# ========== Safety ==========
SAFETY_UNIFIED_CONFIG = {
    "top_k": get_subject_k("ACAS_Xu_N3_3", 6),
    "repair_epochs": 250,
    "repair_lr": 0.005,
    "repair_lambda": 0.5,
    "lambda_clean": 0.5,
    "max_repair_samples": 100,
    "reference_k": VERIFICATION_CONFIG["reference_k"],
    "intervention_threshold": VERIFICATION_CONFIG["intervention_threshold"],
    "sce_repair_top_ratio": VERIFICATION_CONFIG["sce_repair_top_ratio"],
    "reference_metric": VERIFICATION_CONFIG["reference_metric"],
    "early_stop_patience": REPAIR_SHARED_CONFIG["early_stop_patience"],
}

# ========== Fairness ==========
FAIRNESS_UNIFIED_CONFIG = {
    "k": get_subject_k("bank", 1),
    "sce_repair_top_ratio": VERIFICATION_CONFIG["sce_repair_top_ratio"],
    "lr": 0.001,
    "epochs": 500,
    "lambda_fair": 0.5,
    "lambda_clean": 0.5,
    "max_acc_drop": 0.05,
    "sbfl_strategy": PATHWAY_CONFIG["sbfl_strategy"],
    "pathway_alpha": PATHWAY_CONFIG["alpha"],
    "layer_k_ratio_cap": PATHWAY_CONFIG["layer_k_ratio_cap"],
    "activation_threshold": PATHWAY_CONFIG["activation_threshold"],
    "reference_k": VERIFICATION_CONFIG["reference_k"],
    "reference_metric": VERIFICATION_CONFIG["reference_metric"],
    "intervention_threshold": VERIFICATION_CONFIG["intervention_threshold"],
    "early_stop_patience": REPAIR_SHARED_CONFIG["early_stop_patience"],
}

FAIRNESS_DATASET_OVERRIDES = {
    "bank": {"k": get_subject_k("bank")},
    "census": {"k": get_subject_k("census")},
    "credit": {"k": get_subject_k("credit")},
}
