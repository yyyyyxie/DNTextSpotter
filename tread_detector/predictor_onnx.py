import os
import numpy as np
import onnxruntime
import torch
import cv2
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode
import detectron2.data.transforms as T
from adet.data.augmentation import Pad
from .config import setup_cfg

class TreadPredictorONNX:
    def __init__(self, backbone="R_50", quantized=True, instance_mode=ColorMode.IMAGE, cpu=False):
        self.cfg = setup_cfg(backbone)
        self.metadata = MetadataCatalog.get(
            self.cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.vis_text = self.cfg.MODEL.TRANSFORMER.ENABLED

        self.voc_size = self.cfg.MODEL.TRANSFORMER.VOC_SIZE
        if self.voc_size == 96:
            self.CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        elif self.voc_size == 37:
            self.CTLABELS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']
        else:
            raise NotImplementedError

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_suffix = ".quant.onnx" if quantized else ".onnx"
        model_name = f"{'r50' if backbone == 'R_50' else 'vit'}_model{model_suffix}"
        model_path = os.path.join(project_root, "models", model_name)

        providers = ['CPUExecutionProvider'] if cpu else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)

        self.aug = T.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
        )
        self.pad = Pad(divisible_size=32) if backbone == "ViT" else (lambda x: x)

        self.input_format = self.cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def _ctc_decode_recognition(self, rec):
        if self.voc_size == 37:
            last_char = '-'
            s = ''
            for c in rec:
                c = int(c)
                if c < self.voc_size - 1:
                    if last_char != c:
                        s += self.CTLABELS[c]
                        last_char = c
                else:
                    last_char = '-'
            s = s.replace('-', '')
        elif self.voc_size == 96:
            last_char = '###'
            s = ''
            for c in rec:
                c = int(c)
                if c < self.voc_size - 1:
                    if last_char != c:
                        s += self.CTLABELS[c]
                        last_char = c
                else:
                    last_char = '###'
        else:
            raise NotImplementedError
        return s

    def __call__(self, original_image):
        if self.input_format == "RGB":
            original_image = original_image[:, :, ::-1]
        
        height, width = original_image.shape[:2]
        image = self.aug.get_transform(original_image).apply_image(original_image)
        
        if isinstance(self.pad, Pad):
            image = self.pad.get_transform(image).apply_image(image)
        else:
            image = self.pad(image)

        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).unsqueeze(0)
        
        ort_inputs = {self.session.get_inputs()[0].name: image.numpy()}
        recs, scores, boxes = self.session.run(None, ort_inputs)

        decoded_texts = [self._ctc_decode_recognition(rec).upper() for rec in recs]
        
        scores = torch.from_numpy(scores)
        final_scores = [torch.max(score[:len(text)], dim=1)[0].tolist() for text, score in zip(decoded_texts, scores)]

        if not boxes.any():
            return decoded_texts, final_scores

        combined_results = []
        for i in range(len(recs)):
            x_coordinate = boxes[i][:, 0].min().item()
            combined_results.append((x_coordinate, decoded_texts[i], final_scores[i]))

        combined_results.sort(key=lambda x: x[0])

        if not combined_results:
            return [], []
        
        sorted_texts, sorted_scores = zip(*[(text, score) for _, text, score in combined_results])

        return list(sorted_texts), list(sorted_scores)
