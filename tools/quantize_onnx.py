import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model(input_model_path, output_model_path):
    """
    Applies dynamic quantization to an ONNX model.

    Args:
        input_model_path (str): Path to the input ONNX model.
        output_model_path (str): Path to save the quantized ONNX model.
    """
    print(f"Applying dynamic quantization to {input_model_path}...")
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        op_types_to_quantize=['Conv'],
        weight_type=QuantType.QUInt8
    )
    print(f"Successfully saved quantized model to {output_model_path}")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(project_root, "models")

    # --- Quantize R_50 Model ---
    r50_input_path = os.path.join(models_dir, "r50_model.onnx")
    r50_output_path = os.path.join(models_dir, "r50_model.quant.onnx")
    if os.path.exists(r50_input_path):
        quantize_onnx_model(r50_input_path, r50_output_path)
    else:
        print(f"Error: {r50_input_path} not found. Please run export_onnx.py first.")

    # --- Quantize ViT Model ---
    vit_input_path = os.path.join(models_dir, "vit_model.onnx")
    vit_output_path = os.path.join(models_dir, "vit_model.quant.onnx")
    if os.path.exists(vit_input_path):
        quantize_onnx_model(vit_input_path, vit_output_path)
    else:
        print(f"Error: {vit_input_path} not found. Please run export_onnx.py first.")
