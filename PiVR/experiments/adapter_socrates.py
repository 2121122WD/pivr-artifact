
import os
import sys
import ast
import csv
import time
import math
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from methods.pathway import PathwayDeepCP
from methods.verifier import CausalVerifier
from methods.repair import ImitationRepair
from utils import numpy_to_torch_dataloader
from hyperparameters_config import (
    FAIRNESS_UNIFIED_CONFIG,
    FAIRNESS_DATASET_OVERRIDES,
    PATHWAY_CONFIG,
    REPAIR_SHARED_CONFIG,
    VERIFICATION_CONFIG,
)


FAIRNESS_LOCALIZATION_DEFAULTS = {
    'min_confidence': 0.35,
    'min_margin': 0.01,
}


def read(x): return x if isinstance(x, str) else str(x)


class CPRepairSocrates:
    def __init__(self, override_config=None):
        self.model = None;
        self.assertion = None;
        self.datapath = None
        self.datalen = 0;
        self.datalen_tot = 0;
        self.acc_datapath = None
        self.acc_datalen = 0;
        self.acc_datalen_tot = 0
        self.do_neuron = [];
        self.class_n = 0;
        self.sens_idx = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pytorch_model = None;
        self.layers_structure = None
        self.override_config = override_config or {}
        self.fairness_localization_config = dict(FAIRNESS_LOCALIZATION_DEFAULTS)
        self.fairness_localization_config.update({k: v for k, v in self.override_config.items() if k in {
            'min_confidence', 'min_margin'
        }})
        default_override_config = {k: v for k, v in FAIRNESS_UNIFIED_CONFIG.items() if k in {
            'k', 'sce_repair_top_ratio', 'lr', 'epochs', 'lambda_task', 'min_confidence',
            'lambda_fair', 'lambda_clean',
            'sbfl_strategy', 'pathway_alpha', 'layer_k_ratio_cap',
            'early_stop_patience'
        }}
        default_override_config.update(self.override_config)
        self.override_config = default_override_config

    def _evaluate_fairness_sample_success(self, sample, target_label, min_confidence, epsilon=0.1):
        """Evaluate fairness repair with a lightweight, task-aligned criterion.

        Keep the fairness objective close to CCBR-style task handling:
        - primary signal: counterfactual consistency between original and flipped inputs
        - avoid over-constraining with strict confidence or margin thresholds
        - keep classification correctness only for logging/analysis
        """
        if not isinstance(sample, torch.Tensor):
            sample_tensor = torch.tensor(sample, dtype=torch.float32, device=self.device)
        else:
            sample_tensor = sample.detach().to(self.device)
        if sample_tensor.dim() == 1:
            sample_tensor = sample_tensor.unsqueeze(0)

        self.pytorch_model.eval()
        with torch.no_grad():
            logits_orig = self.pytorch_model(sample_tensor)
            probs_orig = torch.softmax(logits_orig, dim=1)
            pred_orig = logits_orig.argmax(dim=1).item()
            conf_orig = float(probs_orig[0, pred_orig].item())

            sample_np = sample_tensor[0].detach().cpu().numpy().copy()
            flipped_np = self._flip_sensitive_attribute(sample_np, [self.sens_idx])
            flipped_tensor = torch.tensor(flipped_np, dtype=sample_tensor.dtype, device=self.device).unsqueeze(0)
            logits_flip = self.pytorch_model(flipped_tensor)
            probs_flip = torch.softmax(logits_flip, dim=1)
            pred_flip = logits_flip.argmax(dim=1).item()
            conf_flip = float(probs_flip[0, pred_flip].item())

        consistency_ok = int(pred_orig) == int(pred_flip)
        confidence_ok = (conf_orig >= min_confidence * 0.25) and (conf_flip >= min_confidence * 0.25)
        consistency_gap = abs(conf_orig - conf_flip)

        # Keep the same unified-style validation as other tasks: consistency first,
        # with a light confidence sanity check only for logging/robustness.
        success = consistency_ok and (confidence_ok or consistency_gap <= max(epsilon, 0.1))

        return {
            'success': success,
            'pred_orig': pred_orig,
            'pred_flip': pred_flip,
            'conf_orig': conf_orig,
            'conf_flip': conf_flip,
            'classification_ok': int(pred_orig) == int(target_label),  # For logging only
            'consistency_ok': consistency_ok,
        }

    def solve(self, model, assertion, display=None, num_rounds=1):
        overall_starttime = time.time()
        self.model = model;
        self.assertion = assertion
        self.num_eval_rounds = num_rounds
        self._parse_spec(assertion)
        if num_rounds != 1: self.num_eval_rounds = num_rounds
        if 'solve_option' not in assertion or len(self.do_neuron) == 0: return
        if self.pytorch_model is None: self._convert_socrates_model_to_pytorch()

        if self.solve_option == 'solve_fairness':
            return self.solve_fairness_cprepair(model, assertion)

    def _parse_spec(self, spec):
        if 'datalen' in spec: self.datalen = spec['datalen']
        if 'datalen_tot' in spec: self.datalen_tot = spec['datalen_tot']
        if 'datapath' in spec: self.datapath = spec['datapath']
        if 'acc_datalen' in spec: self.acc_datalen = spec['acc_datalen']
        if 'acc_datalen_tot' in spec: self.acc_datalen_tot = spec['acc_datalen_tot']
        if 'acc_datapath' in spec: self.acc_datapath = spec['acc_datapath']
        if 'do_neuron' in spec: self.do_neuron = ast.literal_eval(read(spec['do_neuron']))
        if 'solve_option' in spec: self.solve_option = spec['solve_option']
        if 'fairness' in spec:
            self.sensitive = np.array(ast.literal_eval(read(spec['fairness'])))
            self.sens_idx = self.sensitive[0]
        if 'class_n' in spec: self.class_n = spec['class_n']

    def _extract_dataset_name(self):
        if not self.datapath:
            return None
        normalized = self.datapath.replace('\\', '/').strip('/')
        parts = [p for p in normalized.split('/') if p]
        for marker in ('benchmark/benchmark/causal/', 'benchmark/causal/', 'causal/'):
            if marker in normalized:
                suffix = normalized.split(marker, 1)[1]
                return suffix.split('/', 1)[0] if suffix else None
        if len(parts) >= 2:
            return parts[-2]
        return None

    def _extract_attribute_name(self):
        if not self.datapath:
            return None
        normalized = self.datapath.replace('\\', '/').strip('/')
        last = normalized.split('/')[-1] if normalized else ''
        if last.startswith('data_di_'):
            attr_part = last[len('data_di_'):]
            return attr_part.split('_', 1)[0]
        return None

    def _resolve_data_path(self, raw_path):
        if not raw_path:
            return None
        if os.path.isabs(raw_path):
            return raw_path
        repo_root = os.path.abspath(os.path.join(current_dir, '..'))
        return os.path.abspath(os.path.join(repo_root, raw_path))

    def _convert_socrates_model_to_pytorch(self):
        input_dim = len(self.model.lower)
        layer_sizes = [input_dim]
        for neurons in self.do_neuron: layer_sizes.append(len(neurons) if isinstance(neurons, list) else neurons)
        layer_sizes.append(2)
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2: layers.append(nn.ReLU())
        self.pytorch_model = nn.Sequential(*layers).to(self.device)
        self.layers_structure = [l for l in layers if isinstance(l, nn.Linear)]
        self._copy_weights_from_socrates()

    def _copy_weights_from_socrates(self):
        dataset_name = self._extract_dataset_name()
        attribute_name = self._extract_attribute_name()
        if not dataset_name:
            print(f"[FairnessDiag] Unable to parse dataset name from datapath={self.datapath}")
            return

        repo_root = os.path.abspath(os.path.join(current_dir, '..'))
        piVR_root = os.path.abspath(os.path.join(repo_root, '..', 'PiVR'))
        benchmark_root = os.path.join(piVR_root, 'benchmark', 'benchmark', 'causal')
        base_candidates = [
            os.path.join(benchmark_root, dataset_name),
            os.path.join(piVR_root, 'benchmark', 'benchmark', 'causal', dataset_name),
            os.path.join(piVR_root, 'benchmark', 'causal', dataset_name),
            os.path.join(repo_root, 'benchmark', 'benchmark', 'causal', dataset_name),
        ]
        base_path = None
        for candidate in base_candidates:
            if os.path.isdir(candidate):
                base_path = candidate
                break
        if base_path is None:
            base_path = base_candidates[0]

        weights_loaded = 0
        missing_layers = []
        print(f"[FairnessDiag] Weight search root: {base_path}")
        if attribute_name:
            print(f"[FairnessDiag] Parsed fairness attribute: {attribute_name}")
        for idx, layer in enumerate(self.layers_structure):
            wf = os.path.join(base_path, 'weights', f'w{idx + 1}.txt')
            bf = os.path.join(base_path, 'bias', f'b{idx + 1}.txt')
            if os.path.exists(wf) and os.path.exists(bf):
                with open(wf, 'r', encoding='utf-8') as f:
                    layer.weight.data = torch.from_numpy(
                        np.array(ast.literal_eval(f.read().strip()), dtype=np.float32)).to(self.device)
                with open(bf, 'r', encoding='utf-8') as f:
                    layer.bias.data = torch.from_numpy(
                        np.array(ast.literal_eval(f.read().strip()), dtype=np.float32)).to(self.device)
                weights_loaded += 1
            else:
                missing_layers.append(idx + 1)
        print(
            f"[FairnessDiag] Weight loading summary: loaded={weights_loaded}/{len(self.layers_structure)}, "
            f"missing_layers={missing_layers if missing_layers else 'none'}"
        )

    def _get_fairness_pred_conf_margin(self, sample):
        if not isinstance(sample, torch.Tensor):
            x = torch.tensor(sample, dtype=torch.float32, device=self.device)
        else:
            x = sample.detach().to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        self.pytorch_model.eval()
        with torch.no_grad():
            logits = self.pytorch_model(x)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            conf = probs[0, pred].item()
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            margin = (sorted_probs[0, 0] - sorted_probs[0, 1]).item() if probs.shape[1] > 1 else conf
        return pred, conf, margin

    def _is_counterfactually_stable(self, sample, label=None, min_confidence=None, min_margin=None):
        pred_orig, _, _ = self._get_fairness_pred_conf_margin(sample)
        sample_np = sample.detach().cpu().numpy().copy() if isinstance(sample, torch.Tensor) else np.array(sample, dtype=np.float32).copy()
        flipped_np = self._flip_sensitive_attribute(sample_np, [self.sens_idx])
        pred_flip, _, _ = self._get_fairness_pred_conf_margin(flipped_np)
        if label is not None and pred_orig != int(label):
            return False
        return pred_orig == pred_flip

    def _is_counterfactually_buggy(self, sample, label=None, min_confidence=None, min_margin=None):
        pred_orig, _, _ = self._get_fairness_pred_conf_margin(sample)
        sample_np = sample.detach().cpu().numpy().copy() if isinstance(sample, torch.Tensor) else np.array(sample, dtype=np.float32).copy()
        flipped_np = self._flip_sensitive_attribute(sample_np, [self.sens_idx])
        pred_flip, _, _ = self._get_fairness_pred_conf_margin(flipped_np)
        if label is not None and pred_orig != int(label):
            return False
        return pred_orig != pred_flip

    def _merge_pathway_masks(self, masks_list):
        if not masks_list:
            return []
        merged_masks = []
        for layer_masks in zip(*masks_list):
            merged = None
            for mask in layer_masks:
                mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
                mask_np = mask_np.astype(np.float32)
                merged = mask_np.copy() if merged is None else np.maximum(merged, mask_np)
            merged_masks.append(merged)
        return merged_masks


    def solve_fairness_cprepair(self, model, assertion):
        overall_start = time.time()
        hp = dict(FAIRNESS_UNIFIED_CONFIG)

        dataset_name = self._extract_dataset_name() or 'unknown'
        sensitive_attr = self._extract_attribute_name() or 'unknown'
        compact_logging = bool(assertion.get('compact_logging', False))
        if FAIRNESS_DATASET_OVERRIDES.get(dataset_name, {}):
            hp.update(FAIRNESS_DATASET_OVERRIDES[dataset_name])
        if assertion.get('fixed_k_override') is not None:
            hp['k'] = int(assertion['fixed_k_override'])
        # Explicit assertion overrides must win over config / dataset defaults.
        for key in [
            'lr', 'epochs', 'lambda_fair', 'lambda_clean',
            'k', 'sce_repair_top_ratio', 'sbfl_strategy',
            'pathway_alpha', 'pathway_activation_threshold',
            'layer_k_ratio_cap', 'early_stop_patience'
        ]:
            if key in assertion and assertion[key] is not None:
                hp[key] = assertion[key]
        ablation_mode = assertion.get('ablation', 'full')
        rq5_mode = bool(assertion.get('rq5_repair_sensitivity', False))

        unified_k = hp['k']
        sce_repair_top_ratio = hp.get('sce_repair_top_ratio', 0.70)
        unified_lambda_fair = hp['lambda_fair']
        unified_lambda_clean = hp['lambda_clean']
        unified_sbfl_strategy = hp['sbfl_strategy']
        layer_k_ratio_cap = hp.get('layer_k_ratio_cap', PATHWAY_CONFIG.get('layer_k_ratio_cap', 0.5))
        if ablation_mode == 'no_pathway_constraint':
            print('[Ablation] no_pathway_constraint: keep localization/verification and losses unchanged; use all-one masks only during repair.')
        if ablation_mode == 'no_verification':
            sce_repair_top_ratio = 1.0

        X_buggy, y_buggy = self._load_buggy_samples()
        X_clean, y_clean = self._load_clean_samples()
        X_eval, y_eval = self._load_full_test_samples()
        if not compact_logging:
            print(
                f"[FairnessDiag] Dataset summary: dataset={dataset_name}, attribute={sensitive_attr}, "
                f"buggy={len(X_buggy)}, clean={len(X_clean)}, eval={len(X_eval)}, sens_idx={self.sens_idx}, "
                f"datapath={self.datapath}, acc_datapath={self.acc_datapath}"
            )
        if len(X_buggy) == 0 or len(X_clean) == 0 or len(X_eval) == 0: return

        before_metrics = self.evaluate_global_metrics(self.pytorch_model, X_eval, y_eval, self.sens_idx,
                                                      num_rounds=self.num_eval_rounds)
        before_acc, before_idr = before_metrics['accuracy'], before_metrics['idr']
        baseline_acc_fixed = before_acc

        X_localization = np.concatenate([X_buggy, X_clean], axis=0)
        X_localization_flip = np.array(
            [self._flip_sensitive_attribute(sample.copy(), [self.sens_idx]) for sample in X_localization],
            dtype=np.float32,
        )

        # [CRITICAL FIX] Align fairness localization with safety/backdoor tasks
        # For fairness: Pass = f(X_flip) == f(X_orig), Fail = f(X_flip) != f(X_orig)
        # Training data construction:
        #   - Input: X_flip (flipped samples)
        #   - Label: f(X_flip) prediction (NOT f(X_orig)!)
        # This ensures SBFL correctly identifies neurons causing prediction changes under flip
        with torch.no_grad():
            flip_tensor = torch.tensor(X_localization_flip, dtype=torch.float32, device=self.device)
            flip_logits = self.pytorch_model(flip_tensor)
            y_flip = flip_logits.argmax(dim=1).cpu().numpy().astype(np.int64)

        localization_loader = numpy_to_torch_dataloader(
            X_localization_flip,
            y_flip,
            batch_size=1,
            shuffle=True,
        )
        train_loader = numpy_to_torch_dataloader(X_clean, y_clean, batch_size=128,
                                                 shuffle=True)

        loc_start = time.perf_counter()

        pathway_analyzer = PathwayDeepCP(
            model_name='fairness_model', model=self.pytorch_model, layers_structure=self.layers_structure,
            alpha=hp['pathway_alpha'], activation_threshold=hp.get('pathway_activation_threshold', PATHWAY_CONFIG['activation_threshold']),
            input_size=(X_buggy.shape[1],), train_loader=localization_loader, device=str(self.device),
            k=unified_k, layer_k_ratio_cap=layer_k_ratio_cap,
            task_type="fairness",
            violation_fn=lambda data, target, predicted_class, model, argmin_mode: self._is_counterfactually_buggy(data, label=predicted_class),
            failure_score_fn=lambda data, target, predicted_class, model, argmin_mode: 1.0 if self._is_counterfactually_buggy(data, label=predicted_class) else 0.5,
        )
        _, pathway_masks = pathway_analyzer.get_topk_indices_and_mask(sfl_strategy=unified_sbfl_strategy, k=unified_k,
                                                                      return_flattened_mask=False)
        if ablation_mode == 'no_localization':
            print('[Ablation] no_localization: using random pathway masks with the same per-layer mask size.')
            rng = np.random.default_rng(int(assertion.get('seed', 0)))
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

        # [Time Measurement Alignment] Localization phase complete
        loc_time = time.perf_counter() - loc_start

        if ablation_mode == 'sft':
            print("[Ablation] SFT Mode: Unlocking all pathway masks to 1s for full-parameter fine-tuning.")
            pathway_masks = [np.ones_like(m) for m in pathway_masks]

        causal_verifier = CausalVerifier(
            model=self.pytorch_model,
            train_loader=train_loader,
            layers_structure=self.layers_structure,
            device=str(self.device),
            task_type="fairness",
            dataset_name=dataset_name,
            sensitive_indices=[self.sens_idx],
            flip_fn=self._flip_sensitive_attribute,
            reference_metric=hp.get('reference_metric', VERIFICATION_CONFIG['reference_metric']),
            ref_top_k=hp.get('reference_k', VERIFICATION_CONFIG['reference_k']),
            lambda_ref_fair=hp.get('lambda_fair', FAIRNESS_UNIFIED_CONFIG.get('lambda_fair', 0.5)),
            lambda_ref_usefulness=hp.get('lambda_clean', FAIRNESS_UNIFIED_CONFIG.get('lambda_clean', 0.75)),
        )

        success_count = 0
        no_ref_count = 0
        sample_infos = []
        sce_records = []
        route_counter = {'region': 0, 'skip_low_sce': 0, 'skip': 0}
        total_k_used = 0
        sce_threshold = None

        for idx, (buggy_sample, target_label) in enumerate(zip(X_buggy, y_buggy)):
            buggy_tensor = torch.tensor(buggy_sample, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                orig_pred = self.pytorch_model(buggy_tensor.unsqueeze(0)).argmax(dim=1).item()
            info = {
                'idx': idx,
                'buggy_tensor': buggy_tensor,
                'target_label': int(orig_pred),
                'ref_sample': None,
                'sce_score': None,
                'route': 'skip',
                'needs_repair': False,
                'orig_pred': orig_pred,
            }
            ref_sample = causal_verifier.select_reference_sample(buggy_tensor, int(orig_pred))
            info['ref_sample'] = ref_sample
            if ref_sample is None:
                no_ref_count += 1
                sample_infos.append(info)
                continue
            sce_score, _ = causal_verifier.calculate_pathway_sce(
                buggy_tensor,
                ref_sample,
                pathway_masks,
                target_class=int(orig_pred),
                argmin_mode=False,
                fairness_mode=True,
                fairness_flip_fn=self._flip_sensitive_attribute,
                fairness_sensitive_indices=[self.sens_idx],
                fairness_dataset_name=dataset_name,
            )
            info['sce_score'] = float(sce_score)
            info['needs_repair'] = True
            info['route'] = 'region'
            sample_infos.append(info)
            sce_records.append(float(sce_score))
            total_k_used += unified_k

        if ablation_mode == 'no_verification':
            for info in sample_infos:
                if info.get('needs_repair') and info.get('ref_sample') is not None:
                    info['route'] = 'region'
        elif len(sce_records) > 0:
            sce_array = np.array(sce_records, dtype=np.float32)
            sce_threshold = float(np.quantile(sce_array, max(0.0, min(1.0, 1.0 - float(sce_repair_top_ratio)))))
            for info in sample_infos:
                if info.get('needs_repair') and info.get('ref_sample') is not None and info.get('sce_score') is not None:
                    info['route'] = 'region' if info['sce_score'] >= sce_threshold else 'skip_low_sce'
        repair_infos = [info for info in sample_infos if info.get('route') == 'region' and info.get('ref_sample') is not None and info.get('sce_score') is not None]
        route_counter['region'] = len(repair_infos)
        route_counter['skip_low_sce'] = sum(1 for info in sample_infos if info.get('route') == 'skip_low_sce')
        route_counter['skip'] = len(X_buggy) - len(repair_infos)

        repair_start = time.perf_counter()
        if repair_infos:
            state_before = {k: v.cpu().clone() for k, v in self.pytorch_model.state_dict().items()}
            repairer = ImitationRepair(
                model=self.pytorch_model,
                layers_structure=self.layers_structure,
                pathway_masks=([np.ones_like(m) for m in pathway_masks] if ablation_mode == 'no_pathway_constraint' else pathway_masks),
                device=str(self.device),
            )
            region_loader = numpy_to_torch_dataloader(
                np.array([info['buggy_tensor'].detach().cpu().numpy() for info in repair_infos], dtype=np.float32),
                y=np.array([int(info.get('target_label', info.get('orig_pred', 0))) for info in repair_infos], dtype=np.int64),
                batch_size=min(len(repair_infos), max(4, min(32, len(repair_infos)))),
                shuffle=True,
            )
            fairness_epochs = int(hp['epochs']) if hp.get('epochs') is not None else 800
            fairness_lr = float(hp['lr'])
            fairness_lambda_clean = float(hp['lambda_clean'])
            fairness_lambda_fair = float(hp['lambda_fair'])
            if rq5_mode:
                fairness_lambda_clean = float(unified_lambda_clean)
                fairness_lambda_fair = float(unified_lambda_fair)
            region_ok, _ = repairer.repair_region(
                region_loader=region_loader,
                epochs=fairness_epochs,
                lr=fairness_lr,
                lambda_task=fairness_lambda_fair,
                clean_loader=train_loader,
                lambda_clean=fairness_lambda_clean,
                argmin_mode=False,
                safe_labels=None,
                fairness_mode=True,
                flip_fn=self._flip_sensitive_attribute,
                sensitive_indices=[self.sens_idx],
                dataset_name=dataset_name,
                early_stop_patience=int(hp.get('early_stop_patience', REPAIR_SHARED_CONFIG['early_stop_patience'])),
            )
            if not region_ok:
                self.pytorch_model.load_state_dict(state_before)
            else:
                success_count = len(repair_infos)

        repair_time = time.perf_counter() - repair_start
        total_time = loc_time + repair_time
        after_metrics = self.evaluate_global_metrics(self.pytorch_model, X_eval, y_eval, self.sens_idx, num_rounds=self.num_eval_rounds)
        fairness_imp = (before_idr - after_metrics['idr']) / before_idr * 100 if before_idr > 0 else 0.0
        acc_cost_abs = after_metrics['accuracy'] - before_acc
        acc_cost_pct = (after_metrics['accuracy'] - before_acc) / before_acc * 100 if before_acc > 0 else 0.0

        return {
            'before_acc': before_acc,
            'before_idr': before_idr,
            'after_acc': after_metrics['accuracy'],
            'after_idr': after_metrics['idr'],
            'fairness_imp': fairness_imp,
            'acc_cost_abs': acc_cost_abs,
            'acc_cost_pct': acc_cost_pct,
            'loc_time': loc_time,
            'repair_time': repair_time,
            'total_time': total_time,
            'dataset': dataset_name,
            'attribute': sensitive_attr,
            'avg_final_k': float(unified_k),
            'route_region': route_counter.get('region', 0),
            'route_skip': route_counter.get('skip', 0),
            'repair_success_count': success_count,
            'repair_failed_count': max(0, len(X_buggy) - success_count),
            'no_ref_count': no_ref_count,
            'num_samples': len(X_clean),
            'ablation': ablation_mode,
        }


    def _flip_sensitive_attribute(self, sample, sensitive_indices, dataset_name='fairness', use_negation=True):
        flipped = np.array(sample, dtype=np.float32).copy()
        if flipped.ndim > 1:
            flipped = flipped.reshape(-1)
        for idx in sensitive_indices:
            if idx < len(flipped):
                val = flipped[idx]
                if isinstance(val, np.ndarray):
                    val = float(val.reshape(-1)[0])
                else:
                    val = float(val)
                lower_val = float(self.model.lower[idx])
                upper_val = float(self.model.upper[idx])
                mid_val = (lower_val + upper_val) / 2.0
                flipped[idx] = upper_val if val < mid_val else lower_val
        return flipped

    def evaluate_global_metrics(self, model, X_data, y_data, sensitive_idx, num_rounds=1):
        model.eval()
        if num_rounds == 1: return self._evaluate_single_round(model, X_data, y_data, sensitive_idx)
        acc_list, idr_list = [], []
        for _ in range(num_rounds):
            m = self._evaluate_single_round(model, X_data, y_data, sensitive_idx)
            acc_list.append(m['accuracy']);
            idr_list.append(m['idr'])
        return {'accuracy': np.mean(acc_list), 'idr': np.mean(idr_list), 'accuracy_std': np.std(acc_list),
                'idr_std': np.std(idr_list)}

    def _evaluate_single_round(self, model, X_data, y_data, sensitive_idx):
        model.eval()
        correct, discrimination_count, total = 0, 0, len(X_data)
        with torch.no_grad():
            for start_idx in range(0, total, 256):
                end_idx = min(start_idx + 256, total)
                X_batch, y_batch = X_data[start_idx:end_idx], y_data[start_idx:end_idx]
                X_tensor, y_tensor = torch.tensor(X_batch, dtype=torch.float32).to(self.device), torch.tensor(y_batch,
                                                                                                              dtype=torch.int64).to(
                    self.device)
                preds = torch.argmax(model(X_tensor), dim=1)
                correct += (preds == y_tensor).sum().item()

                X_flip = X_batch.copy()
                mid_val = (self.model.lower[sensitive_idx] + self.model.upper[sensitive_idx]) / 2.0
                for i in range(len(X_flip)):
                    if sensitive_idx < X_flip.shape[1]:
                        X_flip[i, sensitive_idx] = self.model.upper[sensitive_idx] if X_flip[
                                                                                          i, sensitive_idx] < mid_val else \
                            self.model.lower[sensitive_idx]
                preds_flip = torch.argmax(model(torch.tensor(X_flip, dtype=torch.float32).to(self.device)), dim=1)
                discrimination_count += (preds != preds_flip).int().sum().item()
        return {'accuracy': correct / total if total > 0 else 0.0,
                'idr': discrimination_count / total if total > 0 else 0.0}

    def _load_buggy_samples(self):
        """
        Load buggy samples with simplified filtering (aligned with safety/backdoor tasks).

        Original: Strict counterfactual violation check with confidence and margin requirements
        Simplified: Only check if pred_orig != pred_flip (basic discrimination check)

        This aligns with:
        - Safety: Only checks if prediction is unsafe
        - Backdoor: Only checks if prediction is target label
        """
        datapath = self.datapath if os.path.isabs(self.datapath) else os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), '..')), self.datapath)
        labels_file = os.path.join(datapath, 'labels.txt')
        y_all = np.array(ast.literal_eval(open(labels_file, 'r').read().strip()), dtype=np.int64) if os.path.exists(
            labels_file) else None
        X_list, y_list = [], []
        skipped_not_buggy = 0

        for i in range(min(self.datalen, self.datalen_tot)):
            x0_file = os.path.join(datapath, f'data{i}.txt')
            if not os.path.exists(x0_file): continue
            sample = np.array(ast.literal_eval(open(x0_file, 'r').read().strip()), dtype=np.float32)
            label = int(y_all[i] if y_all is not None else self.class_n)

            # Simplified check: only verify basic discrimination (pred_orig != pred_flip)
            # No strict confidence or margin requirements
            sample_tensor = torch.tensor(sample, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.pytorch_model.eval()
            with torch.no_grad():
                pred_orig = self.pytorch_model(sample_tensor).argmax(dim=1).item()

                # Flip sensitive attribute
                sample_np = sample.copy()
                flipped_np = self._flip_sensitive_attribute(sample_np, [self.sens_idx])
                flipped_tensor = torch.tensor(flipped_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                pred_flip = self.pytorch_model(flipped_tensor).argmax(dim=1).item()

            # Simple discrimination check (like safety/backdoor)
            if pred_orig == pred_flip:
                skipped_not_buggy += 1
                continue

            X_list.append(sample)
            y_list.append(label)

        if skipped_not_buggy > 0:
            print \
                (f"[FairnessData] Skipped {skipped_not_buggy} samples with consistent predictions (not discriminatory).")
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    def _load_clean_samples(self):
        """
        Load clean samples with simplified filtering (aligned with safety/backdoor tasks).

        Original: Strict counterfactual stability check
        Simplified: Load all samples (like safety/backdoor tasks)
        """
        if not self.acc_datapath: return np.array([]), np.array([])
        acc_datapath = self.acc_datapath if os.path.isabs(self.acc_datapath) else os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), '..')), self.acc_datapath)
        labels_file = os.path.join(acc_datapath, 'labels.txt')
        if not os.path.exists(labels_file): return np.array([]), np.array([])
        y_all = np.array(ast.literal_eval(open(labels_file, 'r').read().strip()), dtype=np.int64)
        X_list, y_list = [], []

        # Simplified: load all samples without strict filtering (like safety/backdoor)
        for i in range(min(self.acc_datalen_tot, len(y_all))):
            x0_file = os.path.join(acc_datapath, f'data{i}.txt')
            if not os.path.exists(x0_file): continue
            sample = np.array(ast.literal_eval(open(x0_file, 'r').read().strip()), dtype=np.float32)
            label = int(y_all[i])
            X_list.append(sample)
            y_list.append(label)

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)

    def _load_full_test_samples(self):
        if not self.acc_datapath: return np.array([]), np.array([])
        acc_datapath = self.acc_datapath if os.path.isabs(self.acc_datapath) else os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), '..')), self.acc_datapath)
        labels_file = os.path.join(acc_datapath, 'labels.txt')
        if not os.path.exists(labels_file): return np.array([]), np.array([])
        y_all = np.array(ast.literal_eval(open(labels_file, 'r').read().strip()), dtype=np.int64)
        X_list, y_list = [], []
        for i in range(min(self.acc_datalen_tot, len(y_all))):
            x0_file = os.path.join(acc_datapath, f'data{i}.txt')
            if not os.path.exists(x0_file): continue
            sample = np.array(ast.literal_eval(open(x0_file, 'r').read().strip()), dtype=np.float32)
            X_list.append(sample)
            y_list.append(int(y_all[i]))
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)



CausalImpl = CPRepairSocrates