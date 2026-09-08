# Fine-tuning Co-DETR on a custom dataset (COCO format)

This is a Co-DETR-specific, end-to-end walkthrough for fine-tuning on your
own dataset. It complements (does not replace) the generic MMDetection guides
in [`customize_dataset.md`](customize_dataset.md) and [`finetune.md`](finetune.md).

## 1. Prepare your dataset in COCO format

Co-DETR (via MMDetection's `CocoDataset`) expects standard [COCO detection
JSON](https://cocodataset.org/#format-data) annotations: one JSON file per
split, each with `images`, `annotations`, and `categories` lists. If your
data is in another format (Pascal VOC XML, YOLO txt, CVAT, LabelMe, ...),
convert it to COCO JSON first -- tools like
[`fiftyone`](https://docs.voxel51.com/), `pylabel`, or a short custom script
using `pycocotools` (already a dependency of this repo) all work well.

Organize your files like this (any names are fine, you'll reference them by
path in the config):

```
data/my_dataset/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── train/
│   └── *.jpg
└── val/
    └── *.jpg
```

**Important:** the `"name"` field of every entry in your JSON's `"categories"`
list is what MMDetection matches against the `classes` tuple in the config
(matching is by name, not by the numeric category id in the JSON). Make sure
category names are consistent across train/val JSON files.

## 2. Create your dataset config

Copy the template and fill in the three TODOs:

```shell
cp projects/configs/_base_/datasets/custom_coco_detection.py \
   projects/configs/_base_/datasets/my_dataset_detection.py
```

Edit `my_dataset_detection.py`:
- `classes` — tuple of your class names, in the label-id order you want
  (must match the category `"name"` fields in your COCO JSON).
- `data_root` — path to the `data/my_dataset/` folder above.
- `ann_file` / `img_prefix` for train/val/test, if your folder layout differs
  from the template.

## 3. Create your model config

Copy the fine-tuning template (Co-DINO, ResNet-50, 5-scale — a good default;
swap the backbone/config family if you need something larger or smaller, see
the Model Zoo in the main [README](../../../README.md)):

```shell
cp projects/configs/co_dino/co_dino_5scale_r50_1x_custom_dataset.py \
   projects/configs/co_dino/co_dino_5scale_r50_1x_my_dataset.py
```

Edit `co_dino_5scale_r50_1x_my_dataset.py`:
1. Point `_base_` at `../_base_/datasets/my_dataset_detection.py` (the file
   you created in step 2).
2. Set `num_classes = len(classes)`.
3. Download a public COCO-pretrained checkpoint (see README Model Zoo /
   [Hugging Face](https://huggingface.co/zongzhuofan)) and set:
   ```python
   load_from = 'checkpoints/co_dino_5scale_r50_1x_coco.pth'
   ```
   Fine-tuning from a pretrained checkpoint converges dramatically faster
   than training from scratch, especially on small/medium custom datasets.
4. Tune `optimizer.lr`, `lr_config.step` and `runner.max_epochs` for your
   dataset size. The template ships with fine-tuning-friendly defaults
   (`lr=2e-5`, 12 epochs, LR step at epoch 8) — for very small datasets
   (a few hundred images) try fewer epochs first to avoid overfitting; for
   larger ones you may want to raise `lr` slightly and add more epochs.

## 4. Train

Single machine, multi-GPU:
```shell
sh tools/dist_train.sh \
    projects/configs/co_dino/co_dino_5scale_r50_1x_my_dataset.py \
    <num_gpus> path_to_exp
```

Single GPU (useful for a quick smoke test before committing to a full run):
```shell
python tools/train.py \
    projects/configs/co_dino/co_dino_5scale_r50_1x_my_dataset.py \
    --work-dir path_to_exp
```

Slurm clusters:
```shell
sh tools/slurm_train.sh partition job_name \
    projects/configs/co_dino/co_dino_5scale_r50_1x_my_dataset.py path_to_exp
```

Training logs, checkpoints (`epoch_*.pth`, `latest.pth`) and a `.log.json`
with per-iteration losses will be written under `path_to_exp/`.

## 5. Evaluate and run inference

Once training finishes, see:
- [`docs/en/tutorials/evaluate_custom_dataset.md`](evaluate_custom_dataset.md)
  for computing COCO-style mAP metrics on your val/test split.
- [`docs/en/tutorials/inference.md`](inference.md) for running inference on
  images or video with your fine-tuned checkpoint.
