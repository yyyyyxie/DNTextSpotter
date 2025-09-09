# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from tread_detector import TreadPredictor

# constants
WINDOW_NAME = "COCO detections"


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--backbone",
        default="R_50",
        help="Backbone model to use for prediction. Available options are 'R_50' and 'ViT'.",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
 
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    # Initialize the predictor with the specified backbone
    predictor = TreadPredictor(backbone=args.backbone)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            texts, scores = predictor(img)
            logger.info(
                "{}: detected {} texts in {:.2f}s".format(
                    path, len(texts), time.time() - start_time
                )
            )
            logger.info(f"Recognized Texts: {texts}")
            logger.info(f"Confidence Scores: {scores}")

            # Visualization is removed for simplicity as the new API does not return visualized output.
            # If visualization is needed, it should be implemented separately.
            if not args.output:
                # cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                # if cv2.waitKey(0) == 27:
                #     break  # esc to quit
                pass
    # Webcam and video input processing are removed for simplicity as they require visualization logic.
    elif args.webcam:
        raise NotImplementedError("Webcam input is not supported in this simplified demo.")
    elif args.video_input:
        raise NotImplementedError("Video input is not supported in this simplified demo.")
