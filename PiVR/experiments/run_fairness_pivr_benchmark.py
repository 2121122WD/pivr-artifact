"""
CP-Repair on CCBR Fairness Benchmark - Aligned Evaluation Script
Usage:
    python run_fairness_pivr_benchmark.py --dataset bank --attribute age --ablation no_imitation
"""

import sys
import os
import argparse
import json
import ast
import csv
from datetime import datetime
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
cp_repair_path = os.path.join(repo_root, 'PiVR')
benchmark_root = os.path.join(cp_repair_path, 'benchmark', 'benchmark')
causal_root = os.path.join(benchmark_root, 'causal')

for _p in (cp_repair_path, repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapter_socrates import CPRepairSocrates
from config import FAIRNESS_UNIFIED_CONFIG


def _resolve_fairness_path(path_value: str) -> str:
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value

    normalized = path_value.replace('\\', '/')
    candidates = [
        os.path.abspath(os.path.join(causal_root, path_value)),
        os.path.abspath(os.path.join(benchmark_root, path_value)),
        os.path.abspath(os.path.join(cp_repair_path, path_value)),
        os.path.abspath(os.path.join(script_dir, path_value)),
    ]
    if normalized.startswith('benchmark/'):
        candidates.insert(0, os.path.abspath(os.path.join(cp_repair_path, normalized.replace('benchmark/', 'benchmark/benchmark/', 1))))
        candidates.insert(0, os.path.abspath(os.path.join(cp_repair_path, normalized)))
    if normalized.startswith('causal/'):
        candidates.insert(0, os.path.abspath(os.path.join(benchmark_root, normalized)))
        candidates.insert(0, os.path.abspath(os.path.join(causal_root, normalized)))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(os.path.join(causal_root, path_value))


def load_spec(dataset, attribute):
    spec_path = os.path.join(causal_root, dataset, f'spec_{attribute}.json')
    with open(spec_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_mock_model(spec):
    class MockModel:
        def __init__(self, spec):
            bounds = ast.literal_eval(spec['model']['bounds'])
            self.lower = np.array([b[0] for b in bounds], dtype=np.float32)
            self.upper = np.array([b[1] for b in bounds], dtype=np.float32)

        def apply(self, x):
            pass

    return MockModel(spec)


def _check_dataset_dir(path_value: str, label: str) -> None:
    if not path_value:
        raise FileNotFoundError(f"Missing required path for {label}.")
    if not os.path.isdir(path_value):
        raise FileNotFoundError(f"Dataset directory not found for {label}: {path_value}.")


def _model_name(dataset: str) -> str:
    return {'bank': 'NN9', 'census': 'NN8', 'credit': 'NN10'}[dataset]


def _rq3_writerow(writer, timestamp, task, model_name, dataset, attribute, seed, ablation_mode,
                  avg_before_idr, avg_after_idr, avg_before_acc, avg_after_acc,
                  avg_fairness_imp, avg_loc_time, avg_ver_time, avg_repair_time, avg_time,
                  avg_final_k, avg_route_region, avg_route_skip, avg_no_ref_count,
                  avg_modified_params, avg_modified_params_ratio):
    writer.writerow([
        timestamp, task, model_name, dataset, attribute, ablation_mode, seed,
        'UnfairRate', f'{avg_before_idr:.4f}', f'{avg_after_idr:.4f}',
        f'{avg_before_acc:.4f}', f'{avg_after_acc:.4f}', f'{avg_before_acc - avg_after_acc:.4f}',
        f'{avg_fairness_imp:.2f}', f'{avg_loc_time:.2f}', f'{avg_ver_time:.2f}', f'{avg_repair_time:.2f}', f'{avg_time:.2f}',
        f'{avg_final_k:.2f}', f'{avg_route_region:.2f}', f'{avg_route_skip:.2f}', f'{avg_no_ref_count:.2f}',
        f'{avg_modified_params:.2f}', f'{avg_modified_params_ratio:.4f}'
    ])


def _rq5_writerow(writer, timestamp, task, model_name, dataset, attribute, seed, param_name, param_value,
                  avg_before_idr, avg_after_idr, avg_before_acc, avg_after_acc,
                  avg_loc_time, avg_ver_time, avg_repair_time, avg_time):
    writer.writerow([
        timestamp, task, model_name, dataset, attribute, seed,
        param_name, param_value,
        'UnfairRate', f'{avg_before_idr:.4f}', f'{avg_after_idr:.4f}',
        f'{avg_before_acc:.4f}', f'{avg_after_acc:.4f}', f'{avg_before_acc - avg_after_acc:.4f}',
        f'{avg_loc_time:.2f}', f'{avg_ver_time:.2f}', f'{avg_repair_time:.2f}', f'{avg_time:.2f}'
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['bank', 'census', 'credit'])
    parser.add_argument('--attribute', type=str, required=True, choices=['age', 'gender', 'race'])
    parser.add_argument('--num_rounds', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--rq3_output', type=str, default=os.path.join(cp_repair_path, 'experiments', 'results', 'rq1_result.csv'))
    parser.add_argument('--rq5_repair_sensitivity', action='store_true', help='启用RQ4修复阶段参数敏感性分析模式')
    parser.add_argument('--rq5_param', type=str, default=None, choices=['eta', 'lambda_clean', 'lambda_task'])
    parser.add_argument('--rq5_value', type=float, default=None)
    parser.add_argument('--rq5_output', type=str, default=os.path.join(cp_repair_path, 'experiments', 'results', 'rq5_repair_sensitivity_result.csv'))
    parser.add_argument('--ablation', type=str, default='full', choices=['full', 'no_localization', 'no_verification', 'no_pathway_constraint'])
    parser.add_argument('--sbfl_strategy', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--layer_k_ratio_cap', type=float, default=None)
    parser.add_argument('--activation_threshold', type=float, default=None)
    parser.add_argument('--sce_repair_top_ratio', type=float, default=None)
    parser.add_argument('--lambda_fair', type=float, default=None)
    parser.add_argument('--lambda_clean', type=float, default=None)
    parser.add_argument('--max_acc_drop', type=float, default=None)
    parser.add_argument('--lr', type=float, default=None)
    args = parser.parse_args()

    if args.rq5_repair_sensitivity and (args.rq5_param is None or args.rq5_value is None):
        parser.error('--rq5_param and --rq5_value are required when --rq5_repair_sensitivity is set')

    spec = load_spec(args.dataset, args.attribute)
    for key in ('datapath', 'acc_datapath'):
        path_value = spec['assert'].get(key)
        if path_value and not os.path.isabs(path_value):
            spec['assert'][key] = _resolve_fairness_path(path_value)
    _check_dataset_dir(spec['assert'].get('datapath', ''), 'buggy datapath')
    _check_dataset_dir(spec['assert'].get('acc_datapath', ''), 'accuracy datapath')

    model_name = _model_name(args.dataset)
    print(
        f"[FairnessDiag] Loaded spec for dataset={args.dataset}, attribute={args.attribute}, model={model_name}, "
        f"datapath={spec['assert'].get('datapath')}, acc_datapath={spec['assert'].get('acc_datapath')}, "
        f"datalen={spec['assert'].get('datalen')}, datalen_tot={spec['assert'].get('datalen_tot')}, "
        f"acc_datalen={spec['assert'].get('acc_datalen')}, acc_datalen_tot={spec['assert'].get('acc_datalen_tot')}"
    )

    spec['assert']['ablation'] = args.ablation
    spec['assert']['seed'] = args.seed
    spec['assert']['compact_logging'] = False

    for key in ['sbfl_strategy', 'alpha', 'k', 'layer_k_ratio_cap', 'activation_threshold', 'sce_repair_top_ratio', 'lambda_fair', 'lambda_clean', 'max_acc_drop', 'lr']:
        value = getattr(args, key, None)
        if value is not None:
            spec['assert'][key] = value
    if args.alpha is not None:
        spec['assert']['pathway_alpha'] = args.alpha

    is_rq5 = bool(args.rq5_repair_sensitivity)
    if is_rq5:
        spec['assert']['rq5_repair_sensitivity'] = True
        spec['assert']['rq5_param'] = args.rq5_param
        spec['assert']['rq5_value'] = args.rq5_value
        if args.rq5_param == 'lambda_task':
            spec['assert']['lambda_fair'] = float(args.rq5_value)
            spec['assert']['lambda_clean'] = max(0.0, min(1.0, 1.0 - float(args.rq5_value)))
        elif args.rq5_param == 'lambda_clean':
            spec['assert']['lambda_clean'] = float(args.rq5_value)
            spec['assert']['lambda_fair'] = max(0.0, min(1.0, 1.0 - float(args.rq5_value)))
        elif args.rq5_param == 'eta':
            spec['assert']['lr'] = float(args.rq5_value)
        spec['assert']['rq5_param_effective'] = args.rq5_param
    if args.ablation == 'no_verification':
        spec['assert']['sce_repair_top_ratio'] = 1.0

    model = create_mock_model(spec)
    round_results = []
    for _ in range(args.num_rounds):
        solver = CPRepairSocrates()
        result = solver.solve(model, spec['assert'], num_rounds=1)
        if result is not None:
            round_results.append(result)

    if not round_results:
        raise RuntimeError('No fairness repair rounds produced results')

    avg_before_acc = np.mean([r['before_acc'] for r in round_results])
    avg_after_acc = np.mean([r['after_acc'] for r in round_results])
    avg_before_idr = np.mean([r['before_idr'] for r in round_results])
    avg_after_idr = np.mean([r['after_idr'] for r in round_results])
    avg_fairness_imp = np.mean([r['fairness_imp'] for r in round_results])
    avg_time = np.mean([r['total_time'] for r in round_results])
    avg_loc_time = np.mean([r.get('loc_time', 0) for r in round_results])
    avg_ver_time = np.mean([r.get('ver_time', 0) for r in round_results])
    avg_repair_time = np.mean([r.get('repair_time', 0) for r in round_results])
    avg_final_k = np.mean([r.get('avg_final_k', 0) for r in round_results])
    avg_route_region = np.mean([r.get('route_region', 0) for r in round_results])
    avg_route_skip = np.mean([r.get('route_skip', 0) for r in round_results])
    avg_no_ref_count = np.mean([r.get('no_ref_count', 0) for r in round_results])
    avg_modified_params = np.mean([r.get('modified_params', 0) for r in round_results])
    avg_modified_params_ratio = np.mean([r.get('modified_params_ratio', 0) for r in round_results])

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    task = 'Fairness'

    if is_rq5:
        results_file = args.rq5_output
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        file_exists = os.path.exists(results_file)
        with open(results_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'task', 'model', 'dataset', 'protected_attr', 'seed',
                    'param_name', 'param_value', 'repair_metric_name', 'repair_metric_before', 'repair_metric_after',
                    'acc_before', 'acc_after', 'drawdown',
                    'loc_time', 'ver_time', 'repair_time', 'total_time'
                ])
            _rq5_writerow(
                writer, timestamp, task, model_name, args.dataset, args.attribute, args.seed,
                args.rq5_param, args.rq5_value,
                avg_before_idr, avg_after_idr, avg_before_acc, avg_after_acc,
                avg_loc_time, avg_ver_time, avg_repair_time, avg_time,
            )
        print(f"[Benchmark] RQ5 results saved to {results_file}")
        print(
            f"[Benchmark][RQ5] param={args.rq5_param}, value={args.rq5_value}, "
            f"before_unfair={avg_before_idr:.4f}, after_unfair={avg_after_idr:.4f}, after_acc={avg_after_acc:.4f}, "
            f"loc_time={avg_loc_time:.2f}, ver_time={avg_ver_time:.2f}, repair_time={avg_repair_time:.2f}, total_time={avg_time:.2f}"
        )
        return

    results_file = args.rq3_output
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    ablation_mode = round_results[0].get('ablation', args.ablation)
    file_exists = os.path.exists(results_file)
    with open(results_file, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'timestamp', 'task', 'model', 'dataset', 'protected_attr', 'seed',
                'ablation', 'repair_metric_name', 'repair_metric_before', 'repair_metric_after',
                'acc_before', 'acc_after', 'drawdown',
                'fairness_imp', 'loc_time', 'ver_time', 'repair_time', 'total_time',
                'avg_final_k', 'route_region', 'route_skip', 'no_ref_count',
                'modified_params', 'modified_params_ratio'
            ])
        _rq3_writerow(
            writer, timestamp, task, model_name, args.dataset, args.attribute, args.seed,
            ablation_mode,
            avg_before_idr, avg_after_idr, avg_before_acc, avg_after_acc,
            avg_fairness_imp, avg_loc_time, avg_ver_time, avg_repair_time, avg_time,
            avg_final_k, avg_route_region, avg_route_skip, avg_no_ref_count,
            avg_modified_params, avg_modified_params_ratio,
        )
    fairness_summary_file = os.path.join(cp_repair_path, 'cp_repair_fairness_results_ccbr_benchmark.csv')
    summary_exists = os.path.exists(fairness_summary_file)
    with open(fairness_summary_file, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow([
                'timestamp', 'task', 'model', 'dataset', 'protected_attr', 'ablation', 'seed',
                'repair_metric_name', 'repair_metric_before', 'repair_metric_after',
                'acc_before', 'acc_after', 'drawdown',
                'fairness_imp', 'loc_time', 'ver_time', 'repair_time', 'total_time'
            ])
        writer.writerow([
            timestamp, task, model_name, args.dataset, args.attribute, ablation_mode, args.seed,
            'UnfairRate', f'{avg_before_idr:.4f}', f'{avg_after_idr:.4f}',
            f'{avg_before_acc:.4f}', f'{avg_after_acc:.4f}', f'{avg_before_acc - avg_after_acc:.4f}',
            f'{avg_fairness_imp:.2f}', f'{avg_loc_time:.2f}', f'{avg_ver_time:.2f}', f'{avg_repair_time:.2f}', f'{avg_time:.2f}'
        ])
    print(f"[Benchmark] Average results ({args.num_rounds} rounds) saved to {results_file}")
    print(f"[Benchmark] Fairness summary also saved to {fairness_summary_file}")
    print(
        f"[Benchmark][Summary] before_unfair={avg_before_idr:.4f}, after_unfair={avg_after_idr:.4f}, "
        f"fairness_imp={avg_fairness_imp:.2f}, after_acc={avg_after_acc:.4f}, "
        f"loc_time={avg_loc_time:.2f}, ver_time={avg_ver_time:.2f}, repair_time={avg_repair_time:.2f}"
    )


if __name__ == '__main__':
    main()
