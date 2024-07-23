<h1 align="center">DNTextSpotter: Arbitrary-Shaped Scene Text Spotting via
Improved Denoising Training</h1> 

## The modified version will be released later.



## Main Results

**Total-Text**

You can use the finetuned weights from Total-Text for inference on the Inverse-Text dataset.

| Backbone  |                Training Data                 | Det-P | Det-R | Det-F1 |         E2E-None          |         E2E-Full          | Weights |
| :-------: | :------------------------------------------: | :---: | :---: | :----: | :-----------------------: | :-----------------------: | :-----: |
|  Res-50   | Synth150K+MLT17+IC13+IC15+TextOCR (pretrain) | 91.5  | 87.0  |  89.2  | $\underline{\text{84.5}}$ | $\underline{\text{89.8}}$ |         |
|  Res-50   |            Total-Text  (finetune)            | 91.5  | 87.0  |  89.2  | $\underline{\text{84.5}}$ | $\underline{\text{89.8}}$ |         |
|  Res-50   |               IC15 (finetune)                | 91.5  | 87.0  |  89.2  | $\underline{\text{84.5}}$ | $\underline{\text{89.8}}$ |         |
| ViTAEv2-S | Synth150K+MLT17+IC13+IC15+TextOCR (pretrain) | 91.5  | 87.0  |  89.2  | $\underline{\text{84.5}}$ | $\underline{\text{89.8}}$ |         |
| ViTAEv2-S |             Total-Text(finetune)             | 91.5  | 87.0  |  89.2  | $\underline{\text{84.5}}$ | $\underline{\text{89.8}}$ |         |



## Usage

- ### Installation

Python 3.8 + PyTorch 2.0.1 + CUDA 11.7 + Detectron2

```
conda create -n dnts python=3.8 -y
conda activate dnts
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
cd detectron2
pip install -e .
pip install -r requirements.txt
cd ..
python setup.py build develop
```

- ### Preparation

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
   |- evaluation
   |  |- gt_*.zip
```

- ### Training

<details>
<summary>Total-Text & ICDAR2015</summary>


**1. Pre-train**

For example, pre-train DNTextSpotter with Synth150K+Total-Text+MLT17+IC13+IC15+TextOCR:

```
python tools/train_net.py --config-file configs/R_50/pretrain/150k_tt_mlt_13_15.yaml --num-gpus 8
```

**2. Fine-tune**

Fine-tune on Total-Text or ICDAR2015:

```
python tools/train_net.py --config-file configs/R_50/TotalText/finetune_150k_tt_mlt_13_15_textocr.yaml --num-gpus 8
python tools/train_net.py --config-file configs/R_50/IC15/finetune_150k_tt_mlt_13_15_textocr.yaml --num-gpus 8
```

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

- ### Evaluation

```
python tools/train_net.py --config-file ${CONFIG_FILE} --eval-only MODEL.WEIGHTS ${MODEL_PATH}
```

- ### Visualization Demo

```
python demo/demo.py --config-file ${CONFIG_FILE} --input ${IMAGES_FOLDER_OR_ONE_IMAGE_PATH} --output ${OUTPUT_PATH} --opts MODEL.WEIGHTS <MODEL_PATH>
```



## Acknowledgement

This project is based on [Adelaidet](https://github.com/aim-uofa/AdelaiDet) and [DeepSolo](https://github.com/ViTAE-Transformer/DeepSolo). For academic use, this project is licensed under the 2-clause BSD License.