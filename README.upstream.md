# DETRs with Collaborative Hybrid Assignments Training

> This is the **original upstream README** from [Sense-X/Co-DETR](https://github.com/Sense-X/Co-DETR),
> preserved here for reference: the paper introduction, the full SOTA model zoo
> (including the ViT-L / Objects365 / LVIS entries that are **not** mirrored by
> this fork), the original News log, and the original multi-GPU / slurm
> instructions. The maintained fork's own README is [`README.md`](README.md).

[![arXiv](https://img.shields.io/badge/arXiv-2211.12860-b31b1b.svg)](https://arxiv.org/abs/2211.12860)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## News

* ***[07/21/2024]*** Co-DETR detection and segmentation checkpoints, fine-tuned on COCO and LVIS, are available on [Hugging Face](https://huggingface.co/zongzhuofan) (new state-of-the-art in instance segmentation).
* ***[04/22/2024]*** [MoVA](https://github.com/TempleX98/MoVA), an MLLM framework, adopts Co-DETR as the vision backbone and achieves state-of-the-art on multimodal benchmarks.
* ***[10/19/2023]*** SOTA model Co-DETR w/ ViT-L released. See [the huggingface page](https://huggingface.co/zongzhuofan).
* ***[09/10/2023]*** LVIS inference configs and a stronger LVIS detector (**64.5 box AP**) released.
* ***[08/21/2023]*** O365 pre-trained Co-DETR with Swin-L achieves **64.8 AP** on COCO test-dev. Config and weights released.
* ***[07/20/2023]*** Code for Co-DINO released: **55.4 AP** with ResNet-50 and **60.7 AP** with Swin-L.
* ***[07/14/2023]*** Co-DETR accepted to ICCV 2023.
* ***[07/12/2023]*** Co-DETR finetuned on LVIS: **72.0 box AP** / **59.7 mask AP** on LVIS minival, **68.0 box AP** / **56.0 mask AP** on LVIS val (no TTA; mask numbers are the auxiliary mask branch).
* ***[07/03/2023]*** Co-DETR with [ViT-L](https://github.com/baaivision/EVA/tree/master/EVA-02) **(304M parameters)** sets a new record of **66.0 AP** on COCO test-dev, surpassing [InternImage-G](https://github.com/OpenGVLab/InternImage) **(~3000M parameters)**. First model to exceed 66.0 AP on COCO test-dev.
* ***[07/03/2023]*** Code for Co-Deformable-DETR released.
* ***[04/05/2023]*** [HoP](https://github.com/Sense-X/HoP) uses Co-DETR as the backbone for new SOTA on the [nuScenes 3D detection leaderboard](https://www.nuscenes.org/object-detection).
* ***[11/19/2022]*** 64.4 AP on COCO minival / 64.5 AP on COCO test-dev with only ImageNet-1K pre-training.

## Introduction

![teaser](figures/framework.png)

Co-DETR is a collaborative hybrid assignments training scheme to learn more
efficient and effective DETR-based detectors from versatile label assignment
manners.

1. **Encoder optimization**: multiple parallel auxiliary heads supervised by
   one-to-many label assignments enhance the encoder's learning ability in
   end-to-end detectors.
2. **Decoder optimization**: extra customized positive queries, extracted from
   the auxiliary heads' positive coordinates, improve decoder attention
   learning.
3. **State-of-the-art performance**: Co-DETR with
   [ViT-L](https://github.com/baaivision/EVA/tree/master/EVA-02) (304M
   parameters) is the first model to achieve **66.0 AP on COCO test-dev**.

![teaser](figures/performance.png)

## Model Zoo (original, full)

Checkpoints below are the original authors', on Google Drive and
[`zongzhuofan` on Hugging Face](https://huggingface.co/zongzhuofan). The fork
mirrors the non-ViT COCO subset to per-model Hub repos (see
[`README.md`](README.md#model-zoo-mirrored-to-hugging-face)); the ViT-L /
Objects365 / LVIS rows here are **not** mirrored.

### Objects365 pre-trained Co-DETR

| Model  | Backbone | Aug | Dataset | box AP (val) | mask AP (val) | box AP (test) | mask AP (test) | Config | Download |
| ------ | -------- | --- | ------- | ------------ | ------------- | ------ | ------------- | ------ | ----- |
| Co-DINO | Swin-L | DETR | COCO | 64.1 | - | - | - | [config](projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO | Swin-L | LSJ | LVIS | 64.5 | - | - | - | [config (test)](projects/configs/co_dino/co_dino_5scale_lsj_swin_large_16e_o365tolvis.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO | ViT-L | DETR | Objects365 | - | - | - | - | N/A (not released) | [model](https://huggingface.co/zongzhuofan) |
| Co-DINO | ViT-L | DETR | COCO | 65.9 | - | 66.0 | - | [config](projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py) | [model](https://huggingface.co/zongzhuofan) |
| Co-DINO-Inst | ViT-L | DETR | COCO | 65.8 | 56.6 | 66.0 | 57.1 | [config](projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py) | [model](https://huggingface.co/zongzhuofan) |
| Co-DINO | ViT-L | LSJ | LVIS | 68.0 | - | - | - | [config (test)](projects/configs/co_dino_vit/co_dino_5scale_lsj_vit_large_lvis.py) | [model](https://huggingface.co/zongzhuofan) |
| Co-DINO-Inst | ViT-L | LSJ | LVIS | 67.3 | 60.7 | - | - | [config (test)](projects/configs/co_dino_vit/co_dino_5scale_lsj_vit_large_lvis_instance.py) | [model](https://huggingface.co/zongzhuofan) |

### Co-DINO with ResNet-50

| Model  | Backbone | Epochs | Aug | Dataset | box AP | Config | Download |
| ------ | -------- | ------ | --- | ------- | ------ | ------ | ----- |
| Co-DINO | R50 | 12 | DETR | COCO | 52.1 | [config](projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO | R50 | 12 | LSJ | COCO | 52.1 | [config](projects/configs/co_dino/co_dino_5scale_lsj_r50_1x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO-9enc | R50 | 12 | LSJ | COCO | 52.6 | [config](projects/configs/co_dino/co_dino_5scale_9encoder_lsj_r50_1x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO | R50 | 36 | LSJ | COCO | 54.8 | [config](projects/configs/co_dino/co_dino_5scale_lsj_r50_3x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO-9enc | R50 | 36 | LSJ | COCO | 55.4 | [config](projects/configs/co_dino/co_dino_5scale_9encoder_lsj_r50_3x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |

### Co-DINO with Swin-L

| Model  | Backbone | Epochs | Aug | Dataset | box AP | Config | Download |
| ------ | -------- | ------ | --- | ------- | ------ | ------ | ----- |
| Co-DINO | Swin-L | 12 | DETR | COCO | 58.9 | [config](projects/configs/co_dino/co_dino_5scale_swin_large_1x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO | Swin-L | 24 | DETR | COCO | 59.8 | [config](projects/configs/co_dino/co_dino_5scale_swin_large_2x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO | Swin-L | 36 | DETR | COCO | 60.0 | [config](projects/configs/co_dino/co_dino_5scale_swin_large_3x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO | Swin-L | 12 | LSJ | COCO | 59.3 | [config](projects/configs/co_dino/co_dino_5scale_lsj_swin_large_1x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO | Swin-L | 24 | LSJ | COCO | 60.4 | [config](projects/configs/co_dino/co_dino_5scale_lsj_swin_large_2x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO | Swin-L | 36 | LSJ | COCO | 60.7 | [config](projects/configs/co_dino/co_dino_5scale_lsj_swin_large_3x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO | Swin-L | 36 | LSJ | LVIS | 56.9 | [config (test)](projects/configs/co_dino/co_dino_5scale_lsj_swin_large_3x_lvis.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |

### Co-Deformable-DETR

| Model  | Backbone | Epochs | Queries | box AP | Config | Download |
| ------ | -------- | ------ | ------- | ------ | ---- | --- |
| Co-Deformable-DETR | R50 | 12 | 300 | 49.5 | [config](projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py) | [model](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) \| [log](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-T | 12 | 300 | 51.7 | [config](projects/configs/co_deformable_detr/co_deformable_detr_swin_tiny_1x_coco.py) | [model](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) \| [log](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-T | 36 | 300 | 54.1 | [config](projects/configs/co_deformable_detr/co_deformable_detr_swin_tiny_3x_coco.py) | [model](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) \| [log](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-S | 12 | 300 | 53.4 | [config](projects/configs/co_deformable_detr/co_deformable_detr_swin_small_1x_coco.py) | [model](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) \| [log](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-S | 36 | 300 | 55.3 | [config](projects/configs/co_deformable_detr/co_deformable_detr_swin_small_3x_coco.py) | [model](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) \| [log](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-B | 12 | 300 | 55.5 | [config](projects/configs/co_deformable_detr/co_deformable_detr_swin_base_1x_coco.py) | [model](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) \| [log](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-B | 36 | 300 | 57.5 | [config](projects/configs/co_deformable_detr/co_deformable_detr_swin_base_3x_coco.py) | [model](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) \| [log](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-L | 12 | 300 | 56.9 | [config](projects/configs/co_deformable_detr/co_deformable_detr_swin_large_1x_coco.py) | [model](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) \| [log](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-L | 36 | 900 | 58.5 | [config](projects/configs/co_deformable_detr/co_deformable_detr_swin_large_900q_3x_coco.py) | [model](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) \| [log](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |

## Running (original instructions)

### Install

Co-DETR is implemented with [MMDetection V2.25.3](https://github.com/open-mmlab/mmdetection/releases/tag/v2.25.3)
and [MMCV V1.5.0](https://github.com/open-mmlab/mmcv/releases/tag/v1.5.0). The
MMDetection source is vendored in this repo; build MMCV per the
[official instructions](https://github.com/open-mmlab/mmcv/tree/v1.5.0#installation).
The original authors tested under `python=3.7.11, pytorch=1.11.0, cuda=11.3`.
(The fork's validated setup uses Python 3.8; see [`README.md`](README.md).)

### Data

Organize COCO and LVIS as:

```
Co-DETR
└── data
    ├── coco
    │   ├── annotations
    │   │      ├── instances_train2017.json
    │   │      └── instances_val2017.json
    │   ├── train2017
    │   └── val2017
    └── lvis_v1
        ├── annotations
        │      ├── lvis_v1_train.json
        │      └── lvis_v1_val.json
        ├── train2017
        └── val2017
```

### Training

Train Co-Deformable-DETR + ResNet-50 with 8 GPUs:

```shell
sh tools/dist_train.sh projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py 8 path_to_exp
```

Train using slurm:

```shell
sh tools/slurm_train.sh partition job_name projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_exp
```

### Testing

Test Co-Deformable-DETR + ResNet-50 with 8 GPUs and evaluate:

```shell
sh tools/dist_test.sh projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_checkpoint 8 --eval bbox
```

Test using slurm:

```shell
sh tools/slurm_test.sh partition job_name projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_checkpoint --eval bbox
```

## Cite Co-DETR

```bibtex
@inproceedings{zong2023detrs,
  title={DETRs with Collaborative Hybrid Assignments Training},
  author={Zong, Zhuofan and Song, Guanglu and Liu, Yu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={6748--6758},
  year={2023}
}
```
