---
# Hugging Face model card template for Co-DETR checkpoints released from this
# repo. Copy this file to the root of the HF model repo as `README.md` (the
# YAML frontmatter below is HF's model-card metadata) and fill in the
# bracketed placeholders before uploading. Do NOT upload until the user
# explicitly confirms — this is a template only.
license: mit
pipeline_tag: object-detection
library_name: mmdetection
tags:
  - object-detection
  - instance-segmentation
  - detr
  - co-detr
  - mmdetection
datasets:
  - [DATASET_NAME_OR_"custom"]
metrics:
  - average_precision
---

# Co-DETR — [MODEL_VARIANT, e.g. "Co-DINO ResNet-50, fine-tuned on <dataset>"]

This is a [MODEL_VARIANT] checkpoint for
[Co-DETR](https://github.com/Sense-X/Co-DETR) ("DETRs with Collaborative
Hybrid Assignments Training", Zong et al., ICCV 2023), fine-tuned by
[YOUR_NAME_OR_ORG] on [DATASET_NAME_OR_DESCRIPTION].

- **Base architecture:** [e.g. Co-DINO, ResNet-50 backbone, 5-scale]
- **Base pretrained weights:** [e.g. `co_dino_5scale_r50_1x_coco.pth`, from
  the official [Co-DETR release](https://huggingface.co/zongzhuofan)]
- **Fine-tuning dataset:** [name / size / #classes / brief description —
  omit or generalize if the dataset itself is not being released]
- **Training config:** [path to the config used, e.g.
  `projects/configs/co_dino/co_dino_5scale_r50_1x_<dataset>.py`]
- **Code:** https://github.com/dronefreak/Co-DETR

## Intended use

[Describe the intended use case, e.g. "Detecting <classes> in <domain>
images." Note any explicit out-of-scope uses.]

## Metrics

Evaluated on [SPLIT_NAME] using `tools/eval.py` /
`docs/en/tutorials/evaluate_custom_dataset.md`:

| Metric | Value |
|---|---|
| bbox_mAP | [VALUE] |
| bbox_mAP_50 | [VALUE] |
| bbox_mAP_75 | [VALUE] |
| bbox_mAP_s / m / l | [VALUE] / [VALUE] / [VALUE] |

## How to use

```shell
git clone https://github.com/dronefreak/Co-DETR
cd Co-DETR
bash tools/setup_codetr_env.sh && conda activate codetr

# download this checkpoint + its config from this HF repo, then:
python tools/inference.py \
    --config <config>.py \
    --checkpoint <this_checkpoint>.pth \
    --input path/to/image.jpg --out-dir outputs/
```

See [`docs/en/tutorials/inference.md`](https://github.com/dronefreak/Co-DETR/blob/main/docs/en/tutorials/inference.md)
for folder/video inference, and
[`docs/en/tutorials/finetune_custom_dataset.md`](https://github.com/dronefreak/Co-DETR/blob/main/docs/en/tutorials/finetune_custom_dataset.md)
if you want to further fine-tune this checkpoint on your own data.

## Limitations

[e.g. domain shift outside training distribution, class imbalance, minimum
object size, known failure modes observed during evaluation.]

## License

Released under the MIT license, matching the
[Co-DETR](https://github.com/Sense-X/Co-DETR) and
[dronefreak/Co-DETR](https://github.com/dronefreak/Co-DETR) licenses.

## Citation

If you use this model, please cite the original Co-DETR paper:

```bibtex
@inproceedings{zong2023detrs,
  title={Detrs with collaborative hybrid assignments training},
  author={Zong, Zhuofan and Song, Guanglu and Liu, Yu},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={6748--6758},
  year={2023}
}
```

[Optionally add a citation/acknowledgement for this fork and/or your
fine-tuning work here.]
