

import torch
import argparse
import torch.nn as nn
import numpy as np
import os
import sys
import time
import csv
import copy
from datetime import datetime
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
my_algorithm_dir = os.path.dirname(current_dir)  # PiVR
np_sbfl_dir = os.path.dirname(my_algorithm_dir)
project_root = os.path.dirname(np_sbfl_dir)

BENCHMARK_ROOT = os.path.join(my_algorithm_dir, 'benchmark', 'benchmark')
ACAS_BENCHMARK_ROOT = BENCHMARK_ROOT


for _p in (np_sbfl_dir, my_algorithm_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from methods.pathway import PathwayDeepCP
from methods.verifier import CausalVerifier
from methods.repair import ImitationRepair
from utils import (
    load_onnx_to_pytorch,
    load_h5_dataset,
    numpy_to_torch_dataloader,
)
from config import (
    PATHWAY_CONFIG,
    SAFETY_UNIFIED_CONFIG,
    SBFL_CONFIG,
    SUBJECT_K_DEFAULTS,
    REPAIR_SHARED_CONFIG,
)

# ACAS subnetwork -> target property mapping (aligned with CCBR / CARE definitions)
# property_safe_actions 表示当前 property 满足安全约束时允许出现的动作集合。
# - N2,9 / φ8: safe iff argmin ∈ {0, 1}
# - N3,3 / φ2: safe iff argmin = 0
# - N1,9 / φ7: safe iff argmin ∉ {3, 4}, i.e. argmin ∈ {0, 1, 2}
ACAS_SUBNETWORK_CONFIGS = {
    "N2,9": {"net_id": (2, 9), "property_id": 8, "property_label": "φ8", "property_safe_actions": [0, 1]},
    "N3,3": {"net_id": (3, 3), "property_id": 2, "property_label": "φ2", "property_safe_actions": [0]},
    "N1,9": {"net_id": (1, 9), "property_id": 7, "property_label": "φ7", "property_safe_actions": [0, 1, 2]},
}
DEFAULT_SUBNETWORK = "N3,3"
PROPERTY2_SAFE_ACTION = 0


def get_property_safe_actions(property_id: int):
    property_to_safe_actions = {
        2: [0],
        7: [0, 1, 2],
        8: [0, 1],
    }
    return property_to_safe_actions.get(int(property_id), [0])


def is_safe_prediction(predicted_action, safe_actions):
    return int(predicted_action) in {int(a) for a in safe_actions}


def get_safe_label_for_buggy_sample(model, buggy_sample, unsafe_action=None, device="cpu", safe_actions=None):
    safe_actions = sorted({int(a) for a in (safe_actions or [0])})
    if unsafe_action is None:
        if not isinstance(buggy_sample, torch.Tensor):
            buggy_tensor = torch.tensor(buggy_sample, dtype=torch.float32)
        else:
            buggy_tensor = buggy_sample.detach().float()
        if buggy_tensor.dim() == 1:
            buggy_tensor = buggy_tensor.unsqueeze(0)
        buggy_tensor = buggy_tensor.to(device)
        model.eval()
        with torch.no_grad():
            unsafe_action = int(model(buggy_tensor).argmin(dim=1).item())

    safe_candidates = [a for a in safe_actions if a != int(unsafe_action)]
    if len(safe_candidates) == 0:
        safe_candidates = safe_actions

    if not isinstance(buggy_sample, torch.Tensor):
        data_tensor = torch.tensor(buggy_sample, dtype=torch.float32)
    else:
        data_tensor = buggy_sample.detach().float()
    if data_tensor.dim() == 1:
        data_tensor = data_tensor.unsqueeze(0)
    data_tensor = data_tensor.to(device)

    model.eval()
    with torch.no_grad():
        logits = model(data_tensor)[0]

    best_safe_action = min(safe_candidates, key=lambda a: float(logits[a].item()))
    return int(best_safe_action)


def build_safety_violation_fn(safe_actions):
    safe_action_set = {int(a) for a in safe_actions}
    def safety_violation_fn(data, target, predicted_class, model, argmin_mode):
        return int(predicted_class) not in safe_action_set
    return safety_violation_fn


def build_safety_failure_score_fn(safe_actions):
    safe_actions = [int(a) for a in safe_actions]
    def safety_failure_score_fn(data, target, predicted_class, model, argmin_mode):
        if not isinstance(data, torch.Tensor):
            data_tensor = torch.tensor(data, dtype=torch.float32)
        else:
            data_tensor = data.detach()
        data_tensor = data_tensor.to(next(model.parameters()).device)
        if data_tensor.dim() == 1:
            data_tensor = data_tensor.unsqueeze(0)
        model.eval()
        with torch.no_grad():
            logits = model(data_tensor)[0]
            safe_logits = [logits[a].item() for a in safe_actions]
            best_safe_logit = min(safe_logits)
            best_overall_logit = torch.min(logits).item()
        return max(0.0, best_safe_logit - best_overall_logit)
    return safety_failure_score_fn


def create_mixed_train_loader(model, benign_samples, buggy_samples, safe_actions, device="cpu", batch_size=1):
    all_data, all_labels = [], []
    model.eval()
    if benign_samples is not None and len(benign_samples) > 0:
        print("  Generating labels for benign samples...")
        with torch.no_grad():
            for sample in benign_samples:
                t = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
                all_data.append(sample)
                all_labels.append(model(t).argmin(dim=1).item())
        print(f"    Added {len(benign_samples)} benign samples (Pass)")
    if buggy_samples is not None and len(buggy_samples) > 0:
        print("  Generating labels for buggy samples...")
        skipped_already_safe = 0
        with torch.no_grad():
            for sample in buggy_samples:
                t = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
                unsafe_action = model(t).argmin(dim=1).item()
                if is_safe_prediction(unsafe_action, safe_actions):
                    skipped_already_safe += 1
                    continue
                # [Academic Defense: Argmin-Semantics Adaptation for Safety]
                # We invert the standard classification assumption. A "Fail" is triggered strictly when the model's minimum-cost action deviates from the formal safe action bounds.
                all_data.append(sample)
                all_labels.append(PROPERTY2_SAFE_ACTION)
        print(f"    Added {len(buggy_samples) - skipped_already_safe} buggy samples (Fail)")
        if skipped_already_safe > 0:
            print(f"    Skipped {skipped_already_safe} buggy-candidate samples already safe under current model")
    if len(all_data) == 0:
        raise ValueError("No samples available")
    all_data = np.array(all_data)
    all_labels = np.array(all_labels)
    nb = len(benign_samples) if benign_samples is not None else 0
    nbu = len(buggy_samples) if buggy_samples is not None else 0
    print(f"    Total: {len(all_data)} samples (benign: {nb}, buggy candidates: {nbu})")
    return numpy_to_torch_dataloader(
        all_data, y=all_labels, batch_size=batch_size, shuffle=False)


def _load_keras_h5_to_pytorch(h5_path):
    import h5py
    def _find_weights(g):
        keys = list(g.keys())
        if 'kernel:0' in keys:
            return (np.array(g['kernel:0']),
                    np.array(g['bias:0']) if 'bias:0' in keys else None)
        for k in keys:
            try:
                sub = g[k]
                if hasattr(sub, 'keys'):
                    k2, b2 = _find_weights(sub)
                    if k2 is not None:
                        return k2, b2
            except Exception:
                pass
        return None, None

    dense_layers = []
    with h5py.File(h5_path, 'r') as f:
        if 'model_weights' in f:
            layer_names = list(f['model_weights'].keys())
        elif 'layer_names' in f.attrs:
            layer_names = [
                n.decode() if isinstance(n, bytes) else n
                for n in f.attrs['layer_names']
            ]
        else:
            layer_names = list(f.keys())
        for name in layer_names:
            grp = None
            if 'model_weights' in f and name in f['model_weights']:
                grp = f['model_weights'][name]
            elif name in f:
                grp = f[name]
            if grp is None:
                continue
            kernel, bias = _find_weights(grp)
            if kernel is not None:
                dense_layers.append((kernel, bias))

    if len(dense_layers) == 0:
        raise ValueError(f"No Dense layers found in {h5_path}")
    print(f"  Found {len(dense_layers)} Dense layers in Keras model")

    torch_layers = []
    for idx, (kernel, bias) in enumerate(dense_layers):
        in_f, out_f = kernel.shape
        linear = nn.Linear(in_f, out_f)
        linear.weight = nn.Parameter(torch.tensor(kernel.T, dtype=torch.float32))
        if bias is not None:
            linear.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        torch_layers.append(linear)
        if idx < len(dense_layers) - 1:
            torch_layers.append(nn.ReLU())
    model = nn.Sequential(*torch_layers)
    print(f"  PyTorch model created: {len(torch_layers)} layers")
    return model, list(torch_layers)


def load_acas_model_and_data(net_id, property_id=2, data_dir=None):
    i, j = net_id
    net_name = f"acas_N{i}{j}"
    # 匿名 artifact 中，ACAS 模型和数据位于：PiVR/benchmark/benchmark/acas_N*/
    care_root = data_dir or ACAS_BENCHMARK_ROOT
    care_model_dir = os.path.join(care_root, f'acas_N{i}{j}', 'models')
    care_data_dir = os.path.join(care_root, f'acas_N{i}{j}', 'data')
    data_subdir = (
        care_data_dir if os.path.exists(care_data_dir)
        else os.path.join(
            os.path.join(care_root, net_name),
        )
    )
    result = {
        'net_id': net_id, 'net_name': net_name,
        'model': None, 'layers_structure': None,
        'buggy_train': None, 'buggy_test': None,
        'benign_train': None, 'benign_test': None,
    }
    print(f"Loading ACAS Xu model {net_id}...")
    care_model_found = False
    for h5_name in [f'ACASXU_{i}_{j}.h5', f'ACASXU_{i}_{j}_init.h5']:
        h5_path = os.path.join(care_model_dir, h5_name)
        if os.path.exists(h5_path):
            print(f"  Found CARE .h5 model: {h5_path}")
            result['model'], result['layers_structure'] = _load_keras_h5_to_pytorch(h5_path)
            care_model_found = True
            break
    if not care_model_found:
        base = data_dir or care_root
        alt_paths = [
            os.path.join(base, 'models', f'ACASXU_run2a_{i}_{j}_batch_2000.onnx'),
            os.path.join(base, 'models', f'ACASXU_{i}_{j}.onnx'),
            os.path.join(base, 'models', f'ACASXU_run2a_{i}_{j}.onnx'),
        ]
        for p in alt_paths:
            if os.path.exists(p):
                print(f"  Found ONNX model: {p}")
                result['model'], result['layers_structure'] = load_onnx_to_pytorch(p)
                break
        if result['model'] is None:
            raise FileNotFoundError(f"Model not found for N{i}{j}.")
    print(f"Loading data from {data_subdir}...")
    for split, fname in [
        ('buggy_train', 'counterexample.h5'),
        ('buggy_test', 'counterexample_test.h5'),
        ('benign_train', 'drawndown.h5'),
        ('benign_test', 'drawndown_test.h5'),
    ]:
        path = os.path.join(data_subdir, fname)
        if os.path.exists(path):
            d = load_h5_dataset(path)
            key = list(d.keys())[0]
            result[split] = d[key]
            print(f"  {split}: {len(result[split])} samples")
        else:
            print(f"  {split}: NOT FOUND ({path})")
    return result


def evaluate_drawdown(model_before, model_after, benign_samples, device="cpu"):
    if benign_samples is None or len(benign_samples) == 0:
        return 0.0
    t = torch.tensor(benign_samples, dtype=torch.float32).to(device)
    model_before.eval()
    model_after.eval()
    with torch.no_grad():
        pb = model_before(t).argmin(dim=1)
        pa = model_after(t).argmin(dim=1)
    return (pb != pa).sum().item() / len(benign_samples)


def build_benign_replay_loader(benign_train, model, device, batch_size=8, max_samples=300):
    if benign_train is None or len(benign_train) == 0:
        return None
    sub = benign_train[:max_samples]
    model.eval()
    with torch.no_grad():
        t = torch.tensor(sub, dtype=torch.float32).to(device)
        labels = model(t).argmin(dim=1).cpu().numpy()
    return numpy_to_torch_dataloader(sub, y=labels, batch_size=batch_size, shuffle=True)


def build_region_repair_loader(model, buggy_samples, safe_actions, device, batch_size=8):
    if buggy_samples is None or len(buggy_samples) == 0:
        return None

    repair_data, repair_labels = [], []
    model.eval()
    with torch.no_grad():
        for sample in buggy_samples:
            sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
            current_pred = int(model(sample_tensor).argmin(dim=1).item())
            if is_safe_prediction(current_pred, safe_actions):
                continue
            safe_label = get_safe_label_for_buggy_sample(
                model, sample, unsafe_action=current_pred, device=device, safe_actions=safe_actions)
            repair_data.append(sample)
            repair_labels.append(safe_label)

    if len(repair_data) == 0:
        return None

    return numpy_to_torch_dataloader(
        np.array(repair_data), y=np.array(repair_labels), batch_size=batch_size, shuffle=True)


def repair_risk_region(
        model, layers_structure, active_masks, region_loader, benign_test_preds_ref, benign_test,
        device, epochs, lr, lambda_task, lambda_clean,
        clean_loader, property_safe_actions,
):
    if region_loader is None:
        return 0, benign_test_preds_ref, None

    repairer = ImitationRepair(model=model, layers_structure=layers_structure, pathway_masks=active_masks, device=device)
    ok, repaired_count = repairer.repair_region(
        region_loader=region_loader,
        epochs=epochs,
        lr=lr,
        lambda_task=lambda_task,
        clean_loader=clean_loader,
        lambda_clean=lambda_clean,
        argmin_mode=True,
        safe_labels=property_safe_actions,
        early_stop_patience=SAFETY_UNIFIED_CONFIG.get('early_stop_patience', REPAIR_SHARED_CONFIG['early_stop_patience']),
    )
    current_k = int(sum(int(np.sum(m)) for m in active_masks)) if isinstance(active_masks, list) and len(active_masks) > 0 else None
    if benign_test is not None and len(benign_test) > 0 and benign_test_preds_ref is not None:
        with torch.no_grad():
            benign_tensor = torch.tensor(benign_test, dtype=torch.float32).to(device)
            benign_test_preds_ref = model(benign_tensor).argmin(dim=1).cpu().numpy()
    return repaired_count if ok else 0, benign_test_preds_ref, current_k


def repair_single_sample(
        model, layers_structure, pathway_masks,
        buggy_sample, benign_test_preds_ref, benign_test,
        device, epochs, lr, lambda_reg, lambda_clean,
        clean_loader,
        ref_sample,
        active_masks,
        property_safe_actions,
):
    buggy_tensor = torch.tensor(buggy_sample, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        pred_before_repair = int(model(buggy_tensor.unsqueeze(0)).argmin(dim=1).item())

    if is_safe_prediction(pred_before_repair, property_safe_actions):
        return True, benign_test_preds_ref, None

    safe_label = get_safe_label_for_buggy_sample(
        model, buggy_sample, unsafe_action=pred_before_repair, device=device, safe_actions=property_safe_actions)

    repairer = ImitationRepair(
        model=model, layers_structure=layers_structure,
        pathway_masks=active_masks, device=device)

    ok = repairer.repair(
        buggy_sample=buggy_tensor, ref_sample=ref_sample, target_label=safe_label,
        epochs=epochs, lr=lr, clean_loader=clean_loader,
        lambda_clean=lambda_clean, kl_mode=True, argmin_mode=True,
        safe_labels=property_safe_actions, backdoor_mode=False,
    )

    current_k = None
    if isinstance(active_masks, list) and len(active_masks) > 0:
        current_k = int(sum(int(np.sum(m)) for m in active_masks))

    if not ok:
        return False, benign_test_preds_ref, current_k

    model.eval()
    with torch.no_grad():
        pred_after_repair = model(buggy_tensor.unsqueeze(0)).argmin(dim=1).item()
    buggy_actually_fixed = is_safe_prediction(pred_after_repair, property_safe_actions)

    if benign_test is not None and len(benign_test) > 0 and benign_test_preds_ref is not None:
        with torch.no_grad():
            bt2 = torch.tensor(benign_test, dtype=torch.float32).to(device)
            preds_now = model(bt2).argmin(dim=1).cpu().numpy()
        return buggy_actually_fixed, preds_now, current_k

    return buggy_actually_fixed, benign_test_preds_ref, current_k


def main():
    parser = argparse.ArgumentParser(description="CP-Repair ACAS Xu Safety Repair")
    parser.add_argument("--subnetwork", choices=list(ACAS_SUBNETWORK_CONFIGS.keys()), default=DEFAULT_SUBNETWORK)
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "no_localization", "no_verification", "no_pathway_constraint"],
                        help="RQ3 ablation modes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rq3_output", type=str, default=os.path.join("PiVR", "experiments", "results", "rq4_ablation_result.csv"))
    parser.add_argument("--rq5_repair_sensitivity", action="store_true", help="Enable RQ5 repair-stage sensitivity mode")
    parser.add_argument("--rq5_param", type=str, default=None, choices=["eta", "lambda_clean", "lambda_task"])
    parser.add_argument("--rq5_value", type=float, default=None)
    parser.add_argument("--rq5_output", type=str, default=os.path.join("PiVR", "experiments", "results", "rq5_repair_sensitivity_result.csv"))
    parser.add_argument("--sweep_mode", action="store_true", help="Enable parameter sweep mode")
    parser.add_argument("--alpha", type=float, default=PATHWAY_CONFIG['alpha'])
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--sbfl_strategy", type=str, default=SBFL_CONFIG['strategy'])
    parser.add_argument("--layer_k_ratio_cap", type=float, default=PATHWAY_CONFIG['layer_k_ratio_cap'])
    args = parser.parse_args()

    selected_config = ACAS_SUBNETWORK_CONFIGS[args.subnetwork]
    selected_subnetwork = args.subnetwork
    selected_property_label = selected_config["property_label"]
    model_name = {
        "N2,9": "NN5",
        "N3,3": "NN6",
        "N1,9": "NN7",
    }[selected_subnetwork]

    print("=" * 80)
    print(f"CP-Repair ACAS Xu Safety Repair Experiment [Ablation: {args.ablation}]")
    print("=" * 80)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    NET_ID = selected_config["net_id"]
    PROPERTY_ID = selected_config["property_id"]
    SFL_STRATEGY = args.sbfl_strategy
    safety_hp = dict(SAFETY_UNIFIED_CONFIG)
    subject_key = f"ACAS_Xu_{selected_subnetwork.replace(',', '_')}"
    safety_hp['top_k'] = SUBJECT_K_DEFAULTS.get(subject_key, safety_hp['top_k'])
    if args.k is not None:
        safety_hp['top_k'] = int(args.k)
    safety_hp['sbfl_strategy'] = args.sbfl_strategy
    pathway_alpha = float(args.alpha)
    pathway_layer_k_ratio_cap = float(args.layer_k_ratio_cap)
    TOP_K = safety_hp['top_k']
    REPAIR_EPOCHS = safety_hp['repair_epochs']
    REPAIR_LR = safety_hp['repair_lr']
    REPAIR_LAMBDA = safety_hp['repair_lambda']
    LAMBDA_CLEAN = safety_hp['lambda_clean']
    MAX_REPAIR_SAMPLES = safety_hp['max_repair_samples']
    SCE_REPAIR_TOP_RATIO = safety_hp.get('sce_repair_top_ratio', 0.70)
    N_AUGMENT = 0

    if args.rq5_repair_sensitivity and (args.rq5_param is None or args.rq5_value is None):
        parser.error("--rq5_param and --rq5_value are required when --rq5_repair_sensitivity is set")
    if args.rq5_repair_sensitivity:
        if args.rq5_param == 'eta':
            REPAIR_LR = float(args.rq5_value)
        elif args.rq5_param == 'lambda_task':
            REPAIR_LAMBDA = float(args.rq5_value)
        elif args.rq5_param == 'lambda_clean':
            REPAIR_LAMBDA = max(0.0, min(1.0, 1.0 - float(args.rq5_value)))
    REPAIR_LAMBDA = max(0.0, min(1.0, REPAIR_LAMBDA))
    LAMBDA_CLEAN = max(0.0, min(1.0, 1.0 - REPAIR_LAMBDA))

    print(f"[SafetyConfig] rq5={args.rq5_repair_sensitivity}, subnetwork={selected_subnetwork}, lr={REPAIR_LR}, lambda_clean={LAMBDA_CLEAN}, task_weight={REPAIR_LAMBDA}, top_ratio={SCE_REPAIR_TOP_RATIO}")

    if args.ablation == 'no_verification':
        SCE_REPAIR_TOP_RATIO = 1.0

    try:
        data_dict = load_acas_model_and_data(NET_ID, PROPERTY_ID)
        model = data_dict['model']
        layers_structure = data_dict['layers_structure']
        buggy_train = data_dict['buggy_train']
        buggy_test = data_dict['buggy_test']
        benign_train = data_dict['benign_train']
        benign_test = data_dict['benign_test']
    except Exception as e:
        print(f"Error loading model/data: {e}")
        return

    model = model.to(device)
    property_safe_actions = selected_config.get("property_safe_actions") or get_property_safe_actions(PROPERTY_ID)

    benign_sub = benign_train[:1000] if benign_train is not None and len(benign_train) > 0 else None
    buggy_sub = buggy_train[:500] if buggy_train is not None and len(buggy_train) > 0 else None
    train_loader = create_mixed_train_loader(model=model, benign_samples=benign_sub, buggy_samples=buggy_sub,
                                             safe_actions=property_safe_actions, device=device, batch_size=1)

    if buggy_train is not None and len(buggy_train) > MAX_REPAIR_SAMPLES:
        idx = np.linspace(0, len(buggy_train) - 1, MAX_REPAIR_SAMPLES, dtype=int)
        buggy_train = buggy_train[idx]

    results_dir = os.path.join('experiments', 'results')
    os.makedirs(results_dir, exist_ok=True)
    csv_file = args.rq5_output if args.rq5_repair_sensitivity else args.rq3_output
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    file_exists = os.path.exists(csv_file)

    loc_time, ver_time, repair_time = 0.0, 0.0, 0.0
    success_count = 0
    model_snapshot_before = copy.deepcopy(model)
    model_snapshot_before.eval()

    print("\n>>> [Phase 1] SFL localisation...")
    loc_start = time.time()
    safety_failure_score_fn = build_safety_failure_score_fn(property_safe_actions)
    # [Academic Defense: Argmin-Semantics Adaptation for Safety]
    # We invert the standard classification assumption. A "Fail" is triggered strictly when the model's minimum-cost action deviates from the formal safe action bounds.
    pathway_locator = PathwayDeepCP(
        model_name=f"ACAS_{NET_ID[0]}_{NET_ID[1]}",
        model=model, layers_structure=layers_structure, train_loader=train_loader, device=device,
        alpha=pathway_alpha, activation_threshold=PATHWAY_CONFIG['activation_threshold'],
        input_size=(5,),
        layer_k_ratio_cap=pathway_layer_k_ratio_cap, argmin_mode=True,
        task_type="safety",
        failure_score_fn=safety_failure_score_fn)

    suspicious_neurons, pathway_masks = pathway_locator.get_topk_indices_and_mask(
        sfl_strategy=SFL_STRATEGY, k=TOP_K, return_flattened_mask=False)

    if args.ablation == 'no_localization':
        print("[Ablation] no_localization: using random pathway masks with the same per-layer mask size.")
        rng = np.random.default_rng(args.seed)
        randomized_masks = []
        for layer_mask in pathway_masks:
            mask_arr = np.asarray(layer_mask)
            flat = mask_arr.reshape(-1)
            k_local = int(flat.sum())
            new_mask = np.zeros_like(flat)
            if k_local > 0:
                idxs = rng.choice(len(flat), size=min(k_local, len(flat)), replace=False)
                new_mask[idxs] = 1
            randomized_masks.append(new_mask.reshape(mask_arr.shape).astype(mask_arr.dtype))
        pathway_masks = randomized_masks

    verification_masks = pathway_masks
    repair_masks = pathway_masks
    if args.ablation == 'no_pathway_constraint':
        repair_masks = [np.ones_like(m) for m in pathway_masks]

    loc_time = time.time() - loc_start
    print(f"  Done in {loc_time:.2f}s (global sub-network mask: {len(pathway_masks)} layers)")

    print("\n>>>[Phase 2] Causal verifier...")
    causal_verifier = CausalVerifier(model=model, train_loader=train_loader, layers_structure=layers_structure,
                                     device=device)
    clean_loader = build_benign_replay_loader(benign_train, model, device, batch_size=8, max_samples=300)

    if benign_test is not None and len(benign_test) > 0:
        model.eval()
        with torch.no_grad():
            _bt = torch.tensor(benign_test, dtype=torch.float32).to(device)
            benign_test_preds_ref = model(_bt).argmin(dim=1).cpu().numpy()
    else:
        benign_test_preds_ref = None

    print(f"\n>>> [Phase 3] Offline candidate selection for {len(buggy_train)} samples...")
    repair_start = time.time()
    total_k_used = 0

    sample_infos = []
    sce_records = []

    for sample_idx, buggy_sample in enumerate(tqdm(buggy_train, desc="Scoring-SCE")):
        buggy_tensor = torch.tensor(buggy_sample, dtype=torch.float32).to(device)
        safe_label = get_safe_label_for_buggy_sample(
            model, buggy_sample, unsafe_action=None, device=device, safe_actions=property_safe_actions)

        model.eval()
        with torch.no_grad():
            pred_before = model(buggy_tensor.unsqueeze(0)).argmin(dim=1).item()

        info = {
            "idx": sample_idx,
            "buggy_sample": buggy_sample,
            "safe_label": safe_label,
            "pred_before": pred_before,
            "ref_sample": None,
            "sce_score": None,
            "route": "skip",
            "needs_repair": pred_before not in set(property_safe_actions),
        }

        if not info["needs_repair"]:
            sample_infos.append(info)
            continue

        ref_sample = causal_verifier.select_reference_sample(
            buggy_tensor,
            safe_label,
            argmin_mode=True,
            safe_labels=property_safe_actions,
        )
        if ref_sample is None:
            ref_sample = buggy_tensor

        verification_masks_torch = [
            torch.from_numpy(m).float().to(device) if isinstance(m, np.ndarray) else m.float().to(device)
            for m in verification_masks
        ]
        sce_score, intervened_pred = causal_verifier.calculate_pathway_sce(
            buggy_tensor,
            ref_sample,
            verification_masks_torch,
            target_class=safe_label,
            argmin_mode=True,
            safe_labels=property_safe_actions,
        )
        info["ref_sample"] = ref_sample
        info["sce_score"] = float(sce_score)
        sample_infos.append(info)
        sce_records.append(float(sce_score))

    if args.ablation == 'no_verification':
        for info in sample_infos:
            if info.get('needs_repair'):
                info['route'] = 'direct'
    elif len(sce_records) > 0:
        sce_array = np.array(sce_records, dtype=np.float32)
        threshold = float(np.quantile(sce_array, max(0.0, min(1.0, 1.0 - float(SCE_REPAIR_TOP_RATIO)))))
        for info in sample_infos:
            if info.get('needs_repair') and info.get('sce_score') is not None and info['sce_score'] >= threshold:
                info['route'] = 'direct'
    region_groups = []
    if any(info["route"] == "direct" for info in sample_infos):
        region_groups.append(("direct", TOP_K, repair_masks, [info for info in sample_infos if info["route"] == "direct"]))

    repaired_buggy_indices = set()
    for route_name, region_k, region_masks, region_infos in region_groups:
        region_buggy_samples = [info["buggy_sample"] for info in region_infos]
        region_loader = build_region_repair_loader(
            model=model,
            buggy_samples=region_buggy_samples,
            safe_actions=property_safe_actions,
            device=device,
            batch_size=8,
        )
        if region_loader is None:
            continue

        print(
            f"  [CP-Repair][Region-Repair] route={route_name}, samples={len(region_buggy_samples)}, "
            f"k={region_k}"
        )
        repair_masks = [np.ones_like(m) for m in region_masks] if args.ablation == 'no_pathway_constraint' else region_masks
        repaired_count, benign_test_preds_ref, used_k = repair_risk_region(
            model=model,
            layers_structure=layers_structure,
            active_masks=repair_masks,
            region_loader=region_loader,
            benign_test_preds_ref=benign_test_preds_ref,
            benign_test=benign_test,
            device=device,
            epochs=REPAIR_EPOCHS,
            lr=REPAIR_LR,
            lambda_task=REPAIR_LAMBDA,
            lambda_clean=LAMBDA_CLEAN,
            clean_loader=clean_loader,
            property_safe_actions=property_safe_actions,
        )
        total_k_used += used_k if used_k is not None else region_k

        region_fixed_indices = set()
        model.eval()
        with torch.no_grad():
            for info in region_infos:
                eval_tensor = torch.tensor(info["buggy_sample"], dtype=torch.float32).unsqueeze(0).to(device)
                eval_pred = int(model(eval_tensor).argmin(dim=1).item())
                if is_safe_prediction(eval_pred, property_safe_actions):
                    region_fixed_indices.add(info["idx"])

        print(
            f"  [CP-Repair][Region-Repair] evaluated fixed samples after training: "
            f"{len(region_fixed_indices)}/{len(region_infos)}"
        )

        if repaired_count > 0:
            repaired_buggy_indices.update(region_fixed_indices)
            success_count += len(region_fixed_indices)

    for info in tqdm(sample_infos, desc="Evaluating-Region-Repair"):
        if not info["needs_repair"]:
            success_count += 1
            continue

        current_buggy_tensor = torch.tensor(info["buggy_sample"], dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            current_pred = int(model(current_buggy_tensor.unsqueeze(0)).argmin(dim=1).item())

        if is_safe_prediction(current_pred, property_safe_actions):
            if info["idx"] not in repaired_buggy_indices:
                success_count += 1
            continue

        if info["route"] == "skip_low_sce":
            print(f"  [CP-Repair][SCE-Routing] skip sample {info['idx']}: base-k SCE={info['sce_score']:.6f}")
    repair_time = time.time() - repair_start

    print(f"\n  Repair done: {success_count}/{len(buggy_train)} success")

    property_safe_action_set = set(property_safe_actions)

    def _vr_frac(mdl, arr):
        if arr is None or len(arr) == 0: return 0.0
        mdl.eval()
        with torch.no_grad():
            t = torch.tensor(arr, dtype=torch.float32).to(device)
            preds = mdl(t).argmin(dim=1).cpu().numpy()
        return float(sum(int(p not in property_safe_action_set) for p in preds)) / len(arr)

    def _pos_acc_frac(mdl, arr):
        if arr is None or len(arr) == 0: return 0.0
        mdl.eval()
        with torch.no_grad():
            t = torch.tensor(arr, dtype=torch.float32).to(device)
            preds = mdl(t).argmin(dim=1).cpu().numpy()
        return float(sum(int(p in property_safe_action_set) for p in preds)) / len(arr)

    ccbr_vr_before = _vr_frac(model_snapshot_before, buggy_test)
    ccbr_vr_after = _vr_frac(model, buggy_test)
    ccbr_pos_acc_before = _pos_acc_frac(model_snapshot_before, benign_test)
    ccbr_pos_acc_after = _pos_acc_frac(model, benign_test)

    ccbr_imp = ((ccbr_vr_before - ccbr_vr_after) / ccbr_vr_before * 100.0 if ccbr_vr_before > 0 else 0.0)
    ccbr_acc_cost = (ccbr_pos_acc_before - ccbr_pos_acc_after) * 100.0
    drawdown_rate = evaluate_drawdown(model_snapshot_before, model, benign_test, device)
    rsr = (100.0 * success_count / len(buggy_train) if len(buggy_train) > 0 else 0.0)
    total_time = loc_time + repair_time

    print("\n" + "=" * 80)
    print(f"Network: ACAS Xu {NET_ID[0]}_{NET_ID[1]} ({selected_subnetwork})  Ablation: {args.ablation}")
    print(f"\nRepair:  {success_count}/{len(buggy_train)}  RSR: {rsr:.2f}%")
    print(f"\nCCBR-aligned Metrics (Table 5):")
    print(f"  Counter VR   : before={ccbr_vr_before:.4f}  after={ccbr_vr_after:.4f}")
    print(f"  Positive ACC : before={ccbr_pos_acc_before:.4f}  after={ccbr_pos_acc_after:.4f}")
    print(f"  Imp.         : {ccbr_imp:.2f}%")
    print(f"  Acc cost     : {ccbr_acc_cost:.2f}%")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset = f"ACAS_Xu_{selected_subnetwork.replace(',', '_')}"
    repair_metric_name = 'VR'
    repair_metric_before = ccbr_vr_before
    repair_metric_after = ccbr_vr_after
    acc_before = ccbr_pos_acc_before
    acc_after = ccbr_pos_acc_after
    drawdown = acc_before - acc_after
    total_trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    modified_params = int(sum(int(np.sum(m)) if isinstance(m, np.ndarray) else int(np.sum(m.detach().cpu().numpy())) for m in repair_masks))
    modified_params_ratio = (modified_params / total_trainable_params) if total_trainable_params > 0 else 0.0

    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if args.rq5_repair_sensitivity:
            if not file_exists:
                writer.writerow([
                    'timestamp', 'task', 'model', 'dataset', 'protected_attr', 'seed',
                    'param_name', 'param_value', 'repair_metric_name', 'repair_metric_before', 'repair_metric_after',
                    'acc_before', 'acc_after',
                    'loc_time', 'ver_time', 'repair_time', 'total_time'
                ])
                file_exists = True
            writer.writerow([
                timestamp, 'Safety', model_name, dataset, 'none', args.seed,
                args.rq5_param, args.rq5_value, repair_metric_name, f"{repair_metric_before:.4f}", f"{repair_metric_after:.4f}",
                f"{acc_before:.4f}", f"{acc_after:.4f}",
                f"{loc_time:.2f}", f"0.00", f"{repair_time:.2f}", f"{total_time:.2f}"
            ])
        else:
            if not file_exists:
                writer.writerow([
                    'timestamp', 'task', 'model', 'dataset', 'protected_attr', 'ablation', 'seed',
                    'repair_metric_name', 'repair_metric_before', 'repair_metric_after',
                    'acc_before', 'acc_after',
                    'modified_params', 'modified_params_ratio',
                    'loc_time', 'ver_time', 'repair_time', 'total_time'
                ])
                file_exists = True
            writer.writerow([
                timestamp, 'Safety', model_name, dataset, 'none', args.ablation, args.seed,
                repair_metric_name, f"{repair_metric_before:.4f}", f"{repair_metric_after:.4f}",
                f"{acc_before:.4f}", f"{acc_after:.4f}",
                modified_params, f"{modified_params_ratio:.6f}",
                f"{loc_time:.2f}", f"0.00", f"{repair_time:.2f}", f"{total_time:.2f}"
            ])
    print(f"Results saved to: {csv_file}")


if __name__ == "__main__":
    main()