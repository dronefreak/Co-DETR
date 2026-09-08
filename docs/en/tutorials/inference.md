# Running inference: images and video

Once you have a trained (or a public pretrained) checkpoint + its config,
`tools/inference.py` gives you one consistent CLI for images, folders of
images, and video.

## Single image

```shell
python tools/inference.py \
    --config projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py \
    --checkpoint checkpoints/co_dino_5scale_r50_1x_coco.pth \
    --input demo/demo.jpg \
    --out-dir outputs/ \
    --score-thr 0.3
```

Writes an annotated copy of the image to `outputs/demo.jpg`.

## Folder of images

```shell
python tools/inference.py \
    --config <config>.py --checkpoint <ckpt>.pth \
    --input path/to/image_folder \
    --out-dir outputs/ \
    --save-json
```

Processes every image in `path/to/image_folder` (recognized extensions:
`.jpg .jpeg .png .bmp .webp .tif .tiff`), writes an annotated copy of each
into `outputs/`, and — with `--save-json` — also writes
`outputs/detections.json`, a `{filename: [{"category", "bbox", "score"}, ...]}`
mapping you can feed into your own downstream analysis/eval code.

## Video

```shell
python tools/inference.py \
    --config <config>.py --checkpoint <ckpt>.pth \
    --input path/to/video.mp4 \
    --out-dir outputs/ \
    --save-json
```

Runs inference frame-by-frame and writes an annotated `outputs/video.mp4`.
With `--save-json`, also writes `outputs/detections.json` as a list of
`{"frame": <index>, "detections": [...]}` entries.

## Flags

| Flag | Description |
|---|---|
| `--device` | `cuda:0` (default), `cuda:1`, ..., or `cpu` |
| `--score-thr` | Minimum detection confidence to keep/draw (default `0.3`) |
| `--show` | Also display results in a window while processing |
| `--save-json` | Dump raw detections (category, bbox, score) alongside the visualizations |

## Notes

- Input type (image / folder / video) is auto-detected from the path: a
  directory is treated as a folder of images, and the file extension
  determines image vs. video for single files.
- This wraps the same `mmdet.apis.init_detector` / `inference_detector` /
  `model.show_result` calls used by the upstream `demo/image_demo.py` and
  `demo/video_demo.py` scripts — those remain available directly if you only
  need a minimal one-off example.
- For fine-tuned checkpoints on a custom dataset, just point `--config` /
  `--checkpoint` at your fine-tuned config/checkpoint from
  [`finetune_custom_dataset.md`](finetune_custom_dataset.md) — class names in
  the output come from `model.CLASSES`, which is restored from either the
  checkpoint's saved metadata or your dataset config automatically.

## Verified: inference works on small (4GB) GPUs

Single-image inference with `co_dino_5scale_r50_1x_coco.py` +
`co_dino_5scale_r50_1x_coco.pth` (Co-DINO, ResNet-50, 5-scale) was smoke
tested end-to-end on an NVIDIA RTX 500 Ada (4 GB VRAM laptop GPU) using the
`codetr` conda environment from `tools/setup_codetr_env.sh`:

- Peak GPU memory during a single forward pass: **~1.5 GB** — comfortably
  fits a 4 GB card, with headroom for higher test resolutions or small
  batches.
- Detections were visually correct (cars, bench, chair all localized
  accurately) on `demo/demo.jpg`.

Training is not practical on a 4 GB GPU with this model family (the training
memory footprint is much larger due to auxiliary heads, larger batch sizes,
and optimizer state) — use a cloud/workstation GPU (≥16 GB recommended) for
fine-tuning, and reserve small local GPUs for inference/evaluation of
already-trained checkpoints.
