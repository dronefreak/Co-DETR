# Evaluating Co-DETR on a custom dataset

This walks through computing COCO-style mAP metrics for a (fine-tuned)
Co-DETR checkpoint on your own validation/test split. It assumes you've
already followed
[`finetune_custom_dataset.md`](finetune_custom_dataset.md) to train a model
and have a config + checkpoint to evaluate.

## Quick summary (single GPU)

The simplest path is the `tools/eval.py` convenience wrapper, which runs
`tools/test.py` under the hood and prints/saves a clean metrics summary:

```shell
python tools/eval.py \
    projects/configs/co_dino/co_dino_5scale_r50_1x_my_dataset.py \
    path_to_exp/latest.pth \
    --eval bbox \
    --work-dir path_to_exp/eval
```

This writes `path_to_exp/eval/eval_<timestamp>.json` (raw metric dict, as
produced by `dataset.evaluate()`) and `path_to_exp/eval/summary.txt` (a
readable table), and prints the same table to stdout, e.g.:

```
metric                         value
------------------------------ ----------
bbox_mAP                       0.4123
bbox_mAP_50                    0.6187
bbox_mAP_75                    0.4456
bbox_mAP_l                     0.5310
bbox_mAP_m                     0.3982
bbox_mAP_s                     0.1904
```

For instance segmentation models (if your fine-tuned config has a mask
branch), pass `--eval bbox segm` to also get `segm_mAP*` metrics.

## Multi-GPU / full control

For multi-GPU evaluation, or if you need flags `tools/eval.py` doesn't expose
(`--show`, `--show-dir`, `--format-only`, etc.), call the underlying scripts
directly:

```shell
# multi-GPU, N GPUs
sh tools/dist_test.sh \
    projects/configs/co_dino/co_dino_5scale_r50_1x_my_dataset.py \
    path_to_exp/latest.pth N --eval bbox

# single GPU, saving qualitative visualizations of predictions
python tools/test.py \
    projects/configs/co_dino/co_dino_5scale_r50_1x_my_dataset.py \
    path_to_exp/latest.pth \
    --eval bbox \
    --show-dir path_to_exp/eval/vis \
    --show-score-thr 0.3

# slurm clusters
sh tools/slurm_test.sh partition job_name \
    projects/configs/co_dino/co_dino_5scale_r50_1x_my_dataset.py \
    path_to_exp/latest.pth --eval bbox
```

## Interpreting the metrics

Metrics follow the standard COCO definitions:
- `bbox_mAP` — mean AP averaged over IoU thresholds 0.50:0.05:0.95 (the
  headline "COCO AP" number used in the Model Zoo table in the README).
- `bbox_mAP_50` / `bbox_mAP_75` — AP at a fixed IoU threshold of 0.5 / 0.75.
- `bbox_mAP_s` / `_m` / `_l` — AP broken down by object size (small / medium
  / large, per COCO's area thresholds).

If your dataset is small or has few instances per category, expect higher
variance in these numbers between runs/epochs — consider evaluating the last
few checkpoints (not just the final one) and picking the best on your val
split, e.g. by sweeping `path_to_exp/epoch_*.pth` through `tools/eval.py`.

## Sanity-checking before trusting the numbers

- Confirm `model.CLASSES` (printed by `tools/eval.py`'s underlying
  `tools/test.py`, or check `checkpoint['meta']['CLASSES']`) matches your
  dataset's category names and order — a silent class mismatch will produce
  low but plausible-looking mAP rather than an obvious crash.
- Double check `--eval` matches what your config actually predicts (`bbox`
  for detection-only heads; add `segm` only if the model has a mask branch).
