import argparse
import os
import cv2
import time
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from tread_detector.predictor_onnx import TreadPredictorONNX

def main(args):
    # --- Create ONNX Predictor ---
    print(f"Loading ONNX model with backbone: {args.backbone} (Quantized: {args.quantized})")
    predictor = TreadPredictorONNX(
        backbone=args.backbone,
        quantized=args.quantized,
        cpu=args.cpu
    )

    # --- Load Image ---
    if not os.path.exists(args.input):
        print(f"Error: Image not found at {args.input}")
        return
    
    image = cv2.imread(args.input)
    if image is None:
        print(f"Error: Could not read image from {args.input}")
        return

    # --- Run Inference ---
    print("Running inference...")
    start_time = time.time()
    
    texts, scores = predictor(image)
    
    end_time = time.time()
    print(f"Inference took {end_time - start_time:.4f} seconds.")

    # --- Print Results ---
    print("\n--- Detection Results ---")
    if not texts:
        print("No text detected.")
    else:
        for i, (text, score) in enumerate(zip(texts, scores)):
            char_scores = ", ".join([f"{s:.2f}" for s in score])
            print(f"[{i+1}] Text: {text} \n    Char Scores: [{char_scores}]")
    print("-----------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for Tread_Detect ONNX Predictor.")
    parser.add_argument(
        "--backbone",
        type=str,
        default="R_50",
        choices=["R_50", "ViT"],
        help="The model backbone to use.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        '--quantized', 
        action='store_true', 
        help='Use the quantized model.'
    )
    parser.add_argument(
        '--no-quantized', 
        dest='quantized', 
        action='store_false',
        help='Do not use the quantized model.'
    )
    parser.set_defaults(quantized=True)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run inference on CPU.",
    )
    args = parser.parse_args()
    main(args)
