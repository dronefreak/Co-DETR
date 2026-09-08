# Running inference: image / folder / video / webcam

Once you have a trained (or a public pretrained) checkpoint + its config,
`tools/inference.py` is a single CLI for all four input types. The mode is
auto-detected from `--input`:

| `--input` value | Mode |
|---|---|
| an existing file with an image extension (`.jpg .jpeg .png .bmp .webp .tif .tiff`) | single image |
| an existing directory | folder of images |
| an existing file with a video extension (`.mp4 .avi .mov .mkv .webm`) | video file |
| `webcam`, `webcam:<id>`, `cam:<id>`, or a bare integer | live camera (id, default `0`) |

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

Processes every image in the folder, writes an annotated copy of each into
`outputs/`, and — with `--save-json` — also writes `outputs/detections.json`,
a `{filename: [{"category", "bbox", "score"}, ...]}` mapping you can feed into
your own downstream analysis/eval code.

## Video file

```shell
python tools/inference.py \
    --config <config>.py --checkpoint <ckpt>.pth \
    --input path/to/video.mp4 \
    --out-dir outputs/ \
    --max-frames 300 \
    --save-json
```

Runs inference frame-by-frame and writes an annotated `outputs/<stem>_pred.mp4`
(`_pred` suffix so it never clobbers the source video). `--max-frames` caps the
run (`0` = whole video). With `--save-json`, also writes `outputs/detections.json`
as a list of `{"frame": <index>, "detections": [...]}` entries.

## Webcam

```shell
python tools/inference.py \
    --config <config>.py --checkpoint <ckpt>.pth \
    --input webcam \
    --show --record --out-dir outputs/
```

Opens camera `0` (use `--input webcam:1` for another device) and runs a live
loop. `--show` displays an annotated window — press `q`, `Q` or `Esc` in it to
stop; without a display, stop with `Ctrl-C` (partial outputs are still
finalized). `--record` saves the annotated stream to
`outputs/webcam_<timestamp>.mp4`; `--save-json` and `--max-frames` work here too.

## Flags

| Flag | Description |
|---|---|
| `--config` / `--checkpoint` | model config and weights (required) |
| `--input` | image / folder / video path, or `webcam[:id]` — see table above |
| `--out-dir` | where annotated outputs + `detections.json` go (default `outputs/`) |
| `--device` | `cuda:0` (default), `cuda:1`, ..., or `cpu` |
| `--score-thr` | minimum detection confidence to keep/draw (default `0.3`) |
| `--save-json` | dump raw detections (category, bbox, score) alongside the visualizations |
| `--max-frames` | video/webcam: stop after N frames (`0` = no limit) |
| `--record` | webcam: also save the annotated stream to `outputs/webcam_<timestamp>.mp4` |
| `--show` | display a results window while processing (needs a display) |

## Notes

- A bad `--input` (nonexistent path, unreadable file) or a camera/video that
  won't open fails with a one-line message, not a traceback.
- `--input` is checked before the checkpoint is loaded, so typos fail fast.
- On a machine with **no camera**, `--input webcam` (id `0`) can block for a
  while inside OpenCV's V4L2 device probe before failing — that is an OpenCV
  behavior, not a bug in this script. A nonexistent explicit id (e.g.
  `webcam:999`) fails immediately.
- This wraps the same `mmdet.apis.init_detector` / `inference_detector` /
  `model.show_result` calls used by the upstream `demo/image_demo.py`,
  `demo/video_demo.py` and `demo/webcam_demo.py` scripts — those remain
  available directly if you only need a minimal one-off example.
- For fine-tuned checkpoints on a custom dataset, just point `--config` /
  `--checkpoint` at your fine-tuned config/checkpoint from
  [`finetune_custom_dataset.md`](finetune_custom_dataset.md) — class names in
  the output come from `model.CLASSES`, restored from the checkpoint metadata
  or the dataset config automatically.

## Verified

- **All four modes** (image, folder, video, webcam-error path) smoke-tested
  end-to-end with `co_deformable_detr_r50_1x_coco` on CPU (`codetr` env):
  annotated outputs written, `detections.json` well-formed for each mode, the
  written video re-opens with the right frame count, and the "camera won't
  open" path returns a clean error.
- **Small-GPU inference:** single-image inference with
  `co_dino_5scale_r50_1x_coco` was smoke tested on an NVIDIA RTX 500 Ada (4 GB
  VRAM laptop GPU) — peak GPU memory ~1.5 GB for a small (≈640 px) image.
  Larger inputs / heavier backbones (Swin, ViT) can exceed 4 GB at the default
  1333×800 test resolution; use `--device cpu` (slower but unbounded by VRAM)
  or a bigger GPU for those.

Training is not practical on a 4 GB GPU with this model family (auxiliary
heads, larger batch sizes, optimizer state) — use a workstation/cloud GPU
(≥16 GB) for fine-tuning and reserve small local GPUs for inference/eval.
