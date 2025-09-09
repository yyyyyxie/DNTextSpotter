# Tread_Detect Module

## How to Prepare

### Environment Settings

CUDA 11.8
Visual Studio Build Tools C++ 2019 installed
Prepare model checkpoints
```
|- ./models
    |-res50_ctw_model_pretrain.pth
    |-vitaev2_pretrain_tt_model_final.pth
```
model links
- [ResNet50 Backbone](https://drive.google.com/file/d/1khGllJJeGzVxHUrnjodhNZF2bMew25XR/view)
- [ViTAE2_S Backbone](https://drive.google.com/file/d/19O3xB2r7Dmren2rjg0aVPCk-wFc9QJi9/view)

### Install

```
conda create -n tread python=3.8
conda activate tread
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
cd detectron2
pip install -e .
cd ..
pip install -r requirements.txt
python setup.py build develop
```

## How to Use
ResNet50 Backbone model use **96voc(include '/')**, however ViT Backbone model us **37voc(not include '/'**)
```
from tread_detect import predictor

img = im.read()
pred = predictor(backbone = "R_50" or backbone = "ViT")
texts, scores = pred(img)
```