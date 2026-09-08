#!/usr/bin/env python
"""Convenience wrapper around `tools/test.py` for evaluating a Co-DETR
checkpoint on a (custom) COCO-format dataset and getting a clean, readable
metrics summary -- instead of having to parse `tools/test.py`'s raw dict
output by hand.

This does not reimplement evaluation logic: it shells out to `tools/test.py`
(single-GPU) and then pretty-prints + re-saves the metrics it produces, so
behavior always matches the underlying MMDetection evaluation exactly.

For multi-GPU evaluation, use `tools/dist_test.sh` directly (see
docs/en/tutorials/evaluate_custom_dataset.md) -- this wrapper is intended for
the common single-GPU / "just get me the numbers" case.

Usage:
    python tools/eval.py \\
        projects/configs/co_dino/co_dino_5scale_r50_1x_my_dataset.py \\
        path_to_exp/latest.pth \\
        --eval bbox \\
        --work-dir path_to_exp/eval
"""
import argparse
import json
import os
import os.path as osp
import subprocess
import sys
import tempfile


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a Co-DETR checkpoint and print a clean '
        'metrics summary (wraps tools/test.py).')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['bbox'],
        help='evaluation metrics, e.g. "bbox", "segm" for COCO-format '
        'datasets (default: bbox)')
    parser.add_argument(
        '--work-dir',
        default=None,
        help='directory to save eval_*.json (metrics) and summary.txt. '
        'Defaults to a temporary directory if not given.')
    parser.add_argument(
        '--gpu-id', type=int, default=0, help='id of gpu to use')
    parser.add_argument(
        '--extra-args',
        nargs=argparse.REMAINDER,
        help='any additional args forwarded verbatim to tools/test.py, '
        'e.g. --extra-args --fuse-conv-bn --show-score-thr 0.5')
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))
    test_script = osp.join(repo_root, 'tools', 'test.py')

    cleanup_tmp = False
    work_dir = args.work_dir
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix='codetr_eval_')
        cleanup_tmp = True
    os.makedirs(work_dir, exist_ok=True)

    cmd = [
        sys.executable, test_script, args.config, args.checkpoint, '--eval',
        *args.eval, '--work-dir', work_dir, '--gpu-id', str(args.gpu_id)
    ]
    if args.extra_args:
        cmd.extend(args.extra_args)

    print(f'[eval] running: {" ".join(cmd)}')
    before = set(os.listdir(work_dir))
    result = subprocess.run(cmd, cwd=repo_root)
    if result.returncode != 0:
        sys.exit(result.returncode)

    # tools/test.py writes work_dir/eval_<timestamp>.json -- find the newest
    # eval_*.json created by this run.
    after = set(os.listdir(work_dir)) - before
    eval_jsons = sorted(
        f for f in after if f.startswith('eval_') and f.endswith('.json'))
    if not eval_jsons:
        print('[eval] warning: could not locate eval_*.json output from '
              'tools/test.py; nothing to summarize.')
        return
    metrics_file = osp.join(work_dir, eval_jsons[-1])
    with open(metrics_file) as f:
        metric_dict = json.load(f)
    metric = metric_dict.get('metric', {})

    summary_lines = [
        f"Config:     {metric_dict.get('config', args.config)}",
        f"Checkpoint: {args.checkpoint}",
        f"Metrics file: {metrics_file}",
        '',
        f"{'metric':30s} value",
        f"{'-' * 30} {'-' * 10}",
    ]
    for key in sorted(metric):
        val = metric[key]
        val_str = f'{val:.4f}' if isinstance(val, float) else str(val)
        summary_lines.append(f'{key:30s} {val_str}')
    summary = '\n'.join(summary_lines)

    print('\n' + summary)
    summary_path = osp.join(work_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary + '\n')
    print(f'\n[eval] summary saved to {summary_path}')

    if cleanup_tmp:
        print(f'[eval] (no --work-dir given, full outputs kept in '
              f'{work_dir})')


if __name__ == '__main__':
    main()
