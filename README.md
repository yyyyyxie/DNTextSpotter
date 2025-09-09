# Tread_Detect Module

## How to Prepare

### Environment Settings

This project is optimized for GPU (CUDA) environments but also supports CPU-only execution. If a CUDA-enabled GPU is not available, the model will automatically run on the CPU without any additional setup.

**GPU (CUDA) Environment:**
- CUDA 11.8
- Visual Studio Build Tools C++ 2019 installed

**CPU-Only Environment:**
- No special requirements needed beyond the standard Python packages.

Prepare model checkpoints
```
|- ./models
    |-res50_ctw_model_pretrain.pth
    |-vitaev2_pretrain_tt_model_final.pth
```
model links
- [ResNet50 Backbone](https://drive.google.com/file/d/1khGllJJeGzVxHUrnjodhNZF2bMew25XR/view)
- [ViTAE2_S Backbone](https://drive.google.com/file/d/19O3xB2r7Dmren2rjg0aVPCk-wFc9QJi9/view)

config settings
- change model path to absolute path in configs/R_50/tread_detect_R50.yaml and configs/ViTAEv2_S/tread_detect_ViT.yaml

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

This project has been modularized into the `tread_detector` package for easy use.

### Initialization

You can initialize the `TreadPredictor` by specifying the backbone model. Supported backbones are `'R_50'` and `'ViT'`.

```python
from tread_detector import TreadPredictor

# Initialize the predictor with the desired backbone
# Default is 'R_50'
predictor_r50 = TreadPredictor(backbone="R_50") 
predictor_vit = TreadPredictor(backbone="ViT")
```

**Note**: The ResNet50 backbone model uses a 96-character vocabulary (including '/'), while the ViT backbone model uses a 37-character vocabulary (alphanumeric only).

### Prediction

Call the predictor instance with an image loaded in BGR format (e.g., using `cv2.imread` or `read_image`). It returns the recognized texts and the confidence scores for each character.

```python
import cv2
from detectron2.data.detection_utils import read_image

# Load image
image_path = "path/to/your/image.jpg"
img = read_image(image_path, format="BGR") # or cv2.imread(image_path)

# Perform prediction
texts, scores = predictor_r50(img)

# Print results
print("Recognized Texts:", texts)
print("Confidence Scores per character:", scores)
```

### Using as a Module in Another Project

You can integrate the `Tread_Detect` project into your own project. Simply place the entire `Tread_Detect` folder inside your project directory.

```
Your_Project/
├── your_main_script.py
└── Tread_Detect/
    ├── tread_detector/
    │   ├── __init__.py
    │   ├── predictor.py
    │   └── ...
    └── ...
```

Then, you can import and use the `TreadPredictor` as follows:

```python
# your_main_script.py
from Tread_Detect.tread_detector import TreadPredictor
import cv2
from detectron2.data.detection_utils import read_image

# Initialize the predictor
predictor = TreadPredictor(backbone="R_50")

# Load and process an image
image_path = "path/to/your/image.jpg"
img = read_image(image_path, format="BGR")
texts, scores = predictor(img)

print("Recognized Texts:", texts)
print("Confidence Scores per character:", scores)
```

### Execute Demo

```
conda activate tread && python demo\demo.py --input [input_dir]
```
