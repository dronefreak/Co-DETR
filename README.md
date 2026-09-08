# DETRs with Collaborative Hybrid Assignments Training

[![arXiv](https://img.shields.io/badge/arXiv-2211.12860-b31b1b.svg)](https://arxiv.org/abs/2211.12860)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](tools/setup_codetr_env.sh)
[![PyTorch 1.11](https://img.shields.io/badge/PyTorch-1.11-EE4C2C.svg)](tools/setup_codetr_env.sh)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/zongzhuofan)

> **Note:** The original README badges linked to [paperswithcode.com](https://paperswithcode.com)
> leaderboards, but that site has since been shut down/redirected and no longer serves those
> pages or badge endpoints — they were replaced above with working, self-hosted badges. Co-DETR's
> reported SOTA results (COCO, LVIS) are unaffected; see the [Model Zoo](#model-zoo) below and the
> [paper](https://arxiv.org/abs/2211.12860) for benchmark numbers.

> **This is [dronefreak/Co-DETR](https://github.com/dronefreak/Co-DETR)**, a community-maintained fork of the original [Sense-X/Co-DETR](https://github.com/Sense-X/Co-DETR) research repo. The upstream project has been unmaintained for years and no longer installs cleanly on modern systems. This fork keeps the original model code and results intact while adding: compatibility fixes for modern Python/NumPy, a validated one-command environment setup script, an end-to-end pipeline for fine-tuning/evaluating/running inference on **your own custom datasets**, and refreshed documentation. See [Acknowledgements & Attribution](#acknowledgements--attribution) for details on what changed and why.

## News

* ***[07/21/2024]*** Check out our Co-DETR detection and segmentation checkpoints, fine-tuned on COCO and LVIS, now available on [Hugging Face](https://huggingface.co/zongzhuofan). We've achieved new state-of-the-art performance in instance segmentation!
* ***[04/22/2024]*** We release a new MLLM framework [MoVA](https://github.com/TempleX98/MoVA), which adopts Co-DETR as the vision and achieves state-of-the-art performance on multimodal benchmarks.
* ***[10/19/2023]*** Our SOTA model Co-DETR w/ ViT-L is released now. Please refer to [our huggingface page](https://huggingface.co/zongzhuofan) for more details.
* ***[09/10/2023]*** We release LVIS inference configs and a stronger LVIS detector that achieves **64.5 box AP**.
* ***[08/21/2023]*** Our O365 pre-trained Co-DETR with Swin-L achieves **64.8 AP** on COCO test-dev. The config and weights are released.
* ***[07/20/2023]*** Code for Co-DINO is released: **55.4 AP** with ResNet-50 and **60.7 AP** with Swin-L.
* ***[07/14/2023]*** Co-DETR is accepted to ICCV 2023!
* ***[07/12/2023]*** We finetune Co-DETR on LVIS and achieve the best results **without TTA**: **72.0 box AP** and **59.7 mask AP** on LVIS minival, **68.0 box AP** and **56.0 mask AP** on LVIS val. For instance segmentation, we report the performance of the auxiliary mask branch.
* ***[07/03/2023]*** Co-DETR with [ViT-L](https://github.com/baaivision/EVA/tree/master/EVA-02) **(304M parameters)** sets a new record of <strike>65.6</strike> **66.0 AP** on COCO test-dev, surpassing the previous best model [InternImage-G](https://github.com/OpenGVLab/InternImage) **(~3000M parameters)**. It is the **first model to exceed 66.0 AP on COCO test-dev**.
* ***[07/03/2023]*** Code for Co-Deformable-DETR is released.
* ***[04/05/2023]*** [HoP](https://github.com/Sense-X/HoP) leverages Co-DETR as the backbone and achieves new SOTA performance on [nuScenes 3D detection leaderboard](https://www.nuscenes.org/object-detection?externalData=all&mapData=all&modalities=Camera).
* ***[11/19/2022]*** We achieved 64.4 AP on COCO minival and 64.5 AP on COCO test-dev with only ImageNet-1K as pre-training data. Codes will be available soon.


## Introduction

![teaser](figures/framework.png)

In this paper, we present a novel collaborative hybrid assignments training scheme, namely Co-DETR, to learn more efficient and effective DETR-based detectors from versatile label assignment manners.
1. **Encoder optimization**: The proposed training scheme can easily enhance the encoder's learning ability in end-to-end detectors by training multiple parallel auxiliary heads supervised by one-to-many label assignments.
2. **Decoder optimization**: We conduct extra customized positive queries by extracting the positive coordinates from these auxiliary heads to improve attention learning of the decoder.
3. **State-of-the-art performance**: Co-DETR with [ViT-L](https://github.com/baaivision/EVA/tree/master/EVA-02) (304M parameters) is **the first model to achieve 66.0 AP on COCO test-dev.**

![teaser](figures/performance.png)

## Model Zoo

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


### Co-DETR with ResNet-50

| Model  | Backbone | Epochs | Aug | Dataset | box AP | Config | Download |
| ------ | -------- | ------ | --- | ------- | ------ | ------ | ----- |
| Co-DINO | R50 | 12 | DETR | COCO | 52.1 | [config](projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO | R50 | 12 | LSJ | COCO | 52.1 | [config](projects/configs/co_dino/co_dino_5scale_lsj_r50_1x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO-9enc | R50 | 12 | LSJ | COCO | 52.6 | [config](projects/configs/co_dino/co_dino_5scale_9encoder_lsj_r50_1x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO | R50 | 36 | LSJ | COCO | 54.8 | [config](projects/configs/co_dino/co_dino_5scale_lsj_r50_3x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |
| Co-DINO-9enc | R50 | 36 | LSJ | COCO | 55.4 | [config](projects/configs/co_dino/co_dino_5scale_9encoder_lsj_r50_3x_coco.py) | [model](https://drive.google.com/drive/folders/1nAXOkzqrEgz-YnXxIEs4d5j9li_kmrnv?usp=sharing) |


### Co-DETR with Swin-L

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

## Running

### Install
We implement Co-DETR using [MMDetection V2.25.3](https://github.com/open-mmlab/mmdetection/releases/tag/v2.25.3) and [MMCV V1.5.0](https://github.com/open-mmlab/mmcv/releases/tag/v1.5.0).
The source code of MMdetection has been included in this repo and you only need to build MMCV following [official instructions](https://github.com/open-mmlab/mmcv/tree/v1.5.0#installation).
The original authors tested their models under ```python=3.7.11,pytorch=1.11.0,cuda=11.3```.
This fork's validated setup (see below) uses ```python=3.8,pytorch=1.11.0,cuda=11.3```
instead — both combinations are known to work; other versions may not be compatible.

**Recommended: one-command setup.** Run the validated setup script, which creates a
conda env (`codetr`, Python 3.8) with a combination of `torch==1.11.0+cu113`,
`mmcv-full==1.5.0` and this repo's pinned dependencies that has been tested
end-to-end for training, evaluation, and inference:
```shell
bash tools/setup_codetr_env.sh
conda activate codetr
```
See `requirements/codetr.txt` (top-level pins) and `requirements/codetr_freeze.txt`
(exact freeze, for byte-for-byte reproduction) for details.

> **CUDA 12.x / PyTorch 2.x status:** `mmcv-full` 1.x (required by this mmdet
> 2.x-based codebase) has no official prebuilt wheels for CUDA 12, and building
> it from source against CUDA 12 / PyTorch 2 is currently **unverified** for
> this repo. If you need CUDA 12 support, please open an issue describing your
> setup so we can track and validate it — until then, CUDA 11.3 (above) is the
> only end-to-end tested path.

### Data
The COCO dataset and LVIS dataset should be organized as:
```
Co-DETR
└── data
    ├── coco
    │   ├── annotations
    │   │      ├── instances_train2017.json
    │   │      └── instances_val2017.json
    │   ├── train2017
    │   └── val2017
    │
    └── lvis_v1
        ├── annotations
        │      ├── lvis_v1_train.json
        │      └── lvis_v1_val.json
        ├── train2017
        └── val2017
```

Training on your **own** dataset? See
[`docs/en/tutorials/finetune_custom_dataset.md`](docs/en/tutorials/finetune_custom_dataset.md)
for a full COCO-format walkthrough (dataset prep, config templates, fine-tuning
schedule).

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
Test Co-Deformable-DETR + ResNet-50 with 8 GPUs, and evaluate:
```shell
sh tools/dist_test.sh projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_checkpoint 8 --eval bbox
```
Test using slurm:
```shell
sh tools/slurm_test.sh partition job_name projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_checkpoint --eval bbox
```
For a single-GPU run with a clean, human-readable metrics summary (handy when
evaluating on a custom dataset), use the `tools/eval.py` wrapper instead — see
[`docs/en/tutorials/evaluate_custom_dataset.md`](docs/en/tutorials/evaluate_custom_dataset.md):
```shell
python tools/eval.py projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_checkpoint --eval bbox
```

### Inference (images & video)
Run a trained/pretrained checkpoint on a single image, a folder of images, or
a video with one consistent CLI (auto-detects the input type, optionally
saves raw detections as JSON):
```shell
python tools/inference.py \
    --config projects/configs/co_dino/co_dino_5scale_r50_1x_coco.py \
    --checkpoint checkpoints/co_dino_5scale_r50_1x_coco.pth \
    --input demo/demo.jpg --out-dir outputs/
```
See [`docs/en/tutorials/inference.md`](docs/en/tutorials/inference.md) for
folder/video examples and all flags.

## Cite Co-DETR

If you find this repository useful, please use the following BibTeX entry for citation.

```latex
@inproceedings{zong2023detrs,
  title={Detrs with collaborative hybrid assignments training},
  author={Zong, Zhuofan and Song, Guanglu and Liu, Yu},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={6748--6758},
  year={2023}
}
```

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Acknowledgements & Attribution

This repository builds directly on the work of others and would not exist
without it:

- **[Co-DETR](https://github.com/Sense-X/Co-DETR)** (Sense-X / SenseTime
  X-Lab) — the original implementation and paper this repo is forked from
  and tracks. All credit for the Co-DETR method itself belongs to the
  original authors (see [Cite Co-DETR](#cite-co-detr) above).
- **[MMDetection](https://github.com/open-mmlab/mmdetection)** and
  **[MMCV](https://github.com/open-mmlab/mmcv)** (OpenMMLab) — the detection
  framework and CUDA/CPU op library this codebase is built on top of.
- The pretrained checkpoints linked from the Model Zoo above are hosted by
  the original authors on [Hugging Face](https://huggingface.co/zongzhuofan)
  and Google Drive; this fork does not claim authorship of those weights.

This fork (dronefreak/Co-DETR) additionally adds: Python/NumPy compatibility
fixes for modern environments, a validated reproducible environment setup
script, and end-to-end training/evaluation/inference tooling and docs for
fine-tuning on custom datasets (see the `## Running` section above and
`docs/en/tutorials/`). These additions are released under the same MIT
license as the rest of the repository.
