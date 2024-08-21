<h3><a href="">DNTextSpotter: Arbitrary-Shaped Scene Text Spotting via Improved Denoising Training</a></h3>

<a href="https://qianqiaoai.github.io/projects/dntextspotter/"><img src="https://img.shields.io/badge/Project-Page-Green"></a>
<a href="https://arxiv.org/abs/2408.00355"><img src="https://img.shields.io/badge/Paper-PDF-orange"></a> 

 [Yu Xie*](https://arxiv.org/search/cs?searchtype=author&query=Xie,+Y), [Qian Qiao*](https://arxiv.org/search/cs?searchtype=author&query=Qiao,+Q), Tianxiang Wu, Shaoyao Huang, Jiaqing Fan, Ziqiang Cao, Zili Wang, Yue Zhang, Jielei Zhang, Huyang Sun

## Release

- [2024/7/16] 🎉🎉🎉 DNTextSpotter is accepted by ACM'MM 2024!

## Main Results

**Pre-trained Models for Total-Text & Inverse-Text & IC15**

| Backbone  |                Training Data                 |                           Weights                            |
| :-------: | :------------------------------------------: | :----------------------------------------------------------: |
|  Res-50   | Synth150K+Total-Text+MLT17+IC13+IC15+TextOCR | [Drive](https://drive.google.com/file/d/1ya5N4gE_Sfl8yMRMYBAmjScRZrSJP7Wk/view?usp=drive_link) |
| ViTAEv2-S | Synth150K+Total-Text+MLT17+IC13+IC15+TextOCR | [Drive](https://drive.google.com/file/d/19O3xB2r7Dmren2rjg0aVPCk-wFc9QJi9/view?usp=drive_link) |

**Total-Text**

| Backbone  |           External Data           | Det-P | Det-R |          Det-F1           |         E2E-None          |         E2E-Full          |                           Weights                            |
| :-------: | :-------------------------------: | :---: | :---: | :-----------------------: | :-----------------------: | :-----------------------: | :----------------------------------------------------------: |
|  Res-50   | Synth150K+MLT17+IC13+IC15+TextOCR | 91.5  | 87.0  | $\underline{\text{89.2}}$ | $\underline{\text{84.5}}$ | $\underline{\text{89.8}}$ | [Drive](https://drive.google.com/file/d/1eKZvjkrqJ4ABKLGHs_4Uj2weyIc6zBm4/view?usp=drive_link) |
| ViTAEv2-S | Synth150K+MLT17+IC13+IC15+TextOCR | 92.9  | 88.6  |         **90.7**          |         **85.0**          |         **90.5**          | [Drive](https://drive.google.com/file/d/19O3xB2r7Dmren2rjg0aVPCk-wFc9QJi9/view?usp=drive_link) |

**Inverse-Text (using the same weights as Total-Text)**

| Backbone  |           External Data           | Det-P | Det-R |          Det-F1           |         E2E-None          |         E2E-Full          |                           Weights                            |
| :-------: | :-------------------------------: | :---: | :---: | :-----------------------: | :-----------------------: | :-----------------------: | :----------------------------------------------------------: |
|  Res-50   | Synth150K+MLT17+IC13+IC15+TextOCR | 94.3  | 77.2  | $\underline{\text{84.9}}$ | $\underline{\text{75.9}}$ | $\underline{\text{81.6}}$ | [Drive](https://drive.google.com/file/d/1eKZvjkrqJ4ABKLGHs_4Uj2weyIc6zBm4/view?usp=drive_link) |
| ViTAEv2-S | Synth150K+MLT17+IC13+IC15+TextOCR | 95.4  | 79.2  |         **86.4**          |         **78.1**          |         **83.8**          | [Drive](https://drive.google.com/file/d/19O3xB2r7Dmren2rjg0aVPCk-wFc9QJi9/view?usp=drive_link) |

**ICDAR 2015 (IC15)**

| Backbone  |              External Data              | Det-P | Det-R |          Det-F1           |           E2E-S           |           E2E-W           |           E2E-G           |                           Weights                            |
| :-------: | :-------------------------------------: | :---: | :---: | :-----------------------: | :-----------------------: | :-----------------------: | :-----------------------: | :----------------------------------------------------------: |
|  Res-50   | Synth150K+Total-Text+MLT17+IC13+TextOCR | 92.5  | 87.2  |           89.8            | $\underline{\text{88.7}}$ | $\underline{\text{84.3}}$ | $\underline{\text{79.9}}$ | [OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcdonZXu6_JtW2QMuA?e=8BTzmi) |
| ViTAEv2-S | Synth150K+Total-Text+MLT17+IC13+TextOCR | 92.4  | 87.9  | $\underline{\text{90.1}}$ |         **89.4**          |         **85.2**          |         **80.6**          | [OneDrive](https://1drv.ms/u/s!AimBgYV7JjTlgcdqw1UUnbSAG4qoWA?e=Co1prY) |

**Pre-trained Model for CTW1500**

| Backbone |                Training Data                 |                           Weights                            |
| :------: | :------------------------------------------: | :----------------------------------------------------------: |
|  Res-50  | Synth150K+Total-Text+MLT17+IC13+IC15+TextOCR | [Drive](https://drive.google.com/file/d/1khGllJJeGzVxHUrnjodhNZF2bMew25XR/view?usp=drive_link) |

**CTW1500**

| Backbone |                External Data                 | Det-P | Det-R | Det-F1 | E2E-None | E2E-Full |                           Weights                            |
| :------: | :------------------------------------------: | :---: | :---: | :----: | :------: | :------: | :----------------------------------------------------------: |
|  Res-50  | Synth150K+Total-Text+MLT17+IC13+IC15+TextOCR | 93.2  | 85.0  |  88.9  |   64.2   |   81.4   | [Drive](https://drive.google.com/file/d/1ODBueatGswUcD24M48GQL-6ZBCTdwH0D/view?usp=drive_link) |

## Usage

- ### Installation

Python 3.8 + PyTorch 2.0.1 + CUDA 11.7 + Detectron2

```
conda create -n dnts python=3.8 -y
conda activate dnts
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
cd detectron2
pip install -e .
cd ..
pip install -r requirements.txt
python setup.py build develop
```

- ### Preparation

You can find the datasets [here](https://github.com/ViTAE-Transformer/DeepSolo/tree/main/DeepSolo).

```
|- ./datasets
   |- syntext1
   |  |- train_images
   |  └  annotations
   |       |- train_37voc.json
   |       └  train_96voc.json
   |- syntext2
   |  |- train_images
   |  └  annotations
   |       |- train_37voc.json
   |       └  train_96voc.json
   |- mlt2017
   |  |- train_images
   |  └  annotations
   |       |- train_37voc.json
   |       └  train_96voc.json
   |- totaltext
   |  |- train_images
   |  |- test_images
   |  |- train_37voc.json
   |  |- train_96voc.json
   |  |- weak_voc_new.txt
   |  |- weak_voc_pair_list.txt
   |  └  test.json
   |- ic13
   |  |- train_images
   |  |- train_37voc.json
   |  └  train_96voc.json
   |- ic15
   |  |- train_images
   |  |- test_images
   |  |- train_37voc.json
   |  |- train_96voc.json
   |  └  test.json
   |- CTW1500
   |  |- train_images
   |  |- test_images
   |  └  annotations
   |       |- train_96voc.json
   |       └  test.json
   |- textocr
   |  |- train_images
   |  |- train_37voc_1.json
   |  |- train_37voc_2.json
   |  |- train_96voc_1.json
   |  └  train_96voc_2.json
   |- inversetext
   |  |- test_images
   |  |- inversetext_lexicon.txt
   |  |- inversetext_pair_list.txt
   |- evaluation
   |  |- gt_*.zip
```

- ### Training

<details>
<summary>Total-Text & ICDAR2015</summary>

**1. Pre-train**

For example, pre-train DNTextSpotter: 

```
python tools/train_net.py --config-file configs/R_50/pretrain/150k_tt_mlt_13_15.yaml --num-gpus 8
```

**2. Fine-tune**

Fine-tune on Total-Text or ICDAR2015:

```
python tools/train_net.py --config-file configs/R_50/TotalText/finetune_150k_tt_mlt_13_15_textocr.yaml --num-gpus 8
python tools/train_net.py --config-file configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml --num-gpus 8
```

</details>
<details>
<summary>CTW1500</summary>

**1. Pre-train**


```
python tools/train_net.py --config-file configs/R_50/CTW1500/pretrain_96voc_50maxlen.yaml --num-gpus 8
```

**2. Fine-tune**

```
python tools/train_net.py --config-file configs/R_50/CTW1500/finetune_96voc_50maxlen.yaml --num-gpus 8
```
</details>

- ### Evaluation

```
python tools/train_net.py --config-file ${CONFIG_FILE} --eval-only MODEL.WEIGHTS ${MODEL_PATH}
```

- ### Visualization Demo

```
python demo/demo.py --config-file ${CONFIG_FILE} --input ${IMAGES_FOLDER_OR_ONE_IMAGE_PATH} --output ${OUTPUT_PATH} --opts MODEL.WEIGHTS <MODEL_PATH>
```

## Citation

If you find DNTextSpotter helpful, please consider giving this repo a star ⭐ and citing:

```
@article{xie2024dntextspotter,
  title={DNTextSpotter: Arbitrary-Shaped Scene Text Spotting via Improved Denoising Training},
  author={Xie, Yu and Qiao, Qian and Gao, Jun and Wu, Tianxiang and Huang, Shaoyao and Fan, Jiaqing and Cao, Ziqiang and Wang, Zili and Zhang, Yue and Zhang, Jielei and others},
  journal={arXiv preprint arXiv:2408.00355},
  year={2024}
}
```

## Acknowledgement

This project is based on [Adelaidet](https://github.com/aim-uofa/AdelaiDet) and [DeepSolo](https://github.com/ViTAE-Transformer/DeepSolo). For academic use, this project is licensed under the 2-clause BSD License.