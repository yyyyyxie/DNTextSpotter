import torch
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from tread_detector.config import setup_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

class ONNXExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, image):
        # The model expects a list of dictionaries as input.
        height, width = image.shape[2], image.shape[3]
        inputs = [{"image": image[0], "height": height, "width": width}]
        
        with torch.no_grad():
            predictions = self.model(inputs)
        
        # Extract the 'instances' object and return tensors as a tuple.
        instances = predictions[0]["instances"]
        return instances.recs, instances.rec_scores, instances.bd

def export_onnx_model(backbone, dummy_input, output_path):
    """
    Loads a model based on the backbone and exports it to ONNX format.
    """
    print(f"Setting up configuration for {backbone} backbone...")
    cfg = setup_cfg(backbone)
    
    cfg.defrost()
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = cfg.MODEL.WEIGHTS.replace('\\', '/')
    
    print(f"Building model for {backbone} backbone...")
    model = build_model(cfg)
    model.eval()

    print(f"Loading checkpoint from {cfg.MODEL.WEIGHTS}...")
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    
    # Wrap the model for ONNX export
    wrapped_model = ONNXExportWrapper(model)

    print(f"Exporting {backbone} model to ONNX format at {output_path}...")
    
    output_names = ['recs', 'rec_scores', 'bd']

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        output_path,
        opset_version=16,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'recs': {0: 'num_instances'},
            'rec_scores': {0: 'num_instances'},
            'bd': {0: 'num_instances'}
        }
    )
    print(f"Successfully exported {backbone} model to {output_path}")

if __name__ == "__main__":
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    # --- Export R_50 Model ---
    cfg_r50 = setup_cfg("R_50")
    r50_dummy_input = torch.randn(1, 3, cfg_r50.INPUT.MIN_SIZE_TEST, cfg_r50.INPUT.MAX_SIZE_TEST)
    r50_output_path = os.path.join(models_dir, "r50_model.onnx")
    export_onnx_model("R_50", r50_dummy_input, r50_output_path)

    # --- Export ViT Model ---
    cfg_vit = setup_cfg("ViT")
    # Use dimensions divisible by 32 to avoid shape errors during tracing
    vit_dummy_input = torch.randn(1, 3, 1024, 1920)
    vit_output_path = os.path.join(models_dir, "vit_model.onnx")
    export_onnx_model("ViT", vit_dummy_input, vit_output_path)
