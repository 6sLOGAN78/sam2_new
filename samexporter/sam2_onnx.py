import time
from typing import Any, List, Tuple

import cv2
import numpy as np
import onnxruntime


class SegmentAnything2ONNX:
    """Segmentation model using Segment Anything 2 (SAM2)"""

    def __init__(self, encoder_model_path, decoder_model_path) -> None:
        self.encoder = SAM2ImageEncoder(encoder_model_path)
        self.decoder = SAM2ImageDecoder(
            decoder_model_path, self.encoder.input_shape[2:]
        )

    def encode(self, cv_image: np.ndarray) -> dict:
        original_size = cv_image.shape[:2]
        high_res_feats_0, high_res_feats_1, image_embed = self.encoder(cv_image)
        return {
            "high_res_feats_0": high_res_feats_0,
            "high_res_feats_1": high_res_feats_1,
            "image_embedding": image_embed,
            "original_size": original_size,
        }

    def predict_masks(self, embedding, prompt) -> np.ndarray:
        points = []
        labels = []

        for mark in prompt:
            if mark["type"] == "point":
                points.append(mark["data"])
                labels.append(mark["label"])
            elif mark["type"] == "rectangle":
                points.append([mark["data"][0], mark["data"][1]])
                points.append([mark["data"][2], mark["data"][3]])
                labels.append(2)
                labels.append(3)

        points = np.array(points, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        image_embedding = embedding["image_embedding"]
        high_res_feats_0 = embedding["high_res_feats_0"]
        high_res_feats_1 = embedding["high_res_feats_1"]
        original_size = embedding["original_size"]

        self.decoder.set_image_size(original_size)

        masks, _ = self.decoder(
            image_embedding,
            high_res_feats_0,
            high_res_feats_1,
            points,
            labels,
        )

        return masks


class SAM2ImageEncoder:
    def __init__(self, path: str) -> None:
        self.session = onnxruntime.InferenceSession(
            path, providers=onnxruntime.get_available_providers()
        )
        self.get_input_details()
        self.get_output_details()

    def __call__(self, image: np.ndarray):
        return self.encode_image(image)

    def encode_image(self, image: np.ndarray):
        input_tensor = self.prepare_input(image)
        outputs = self.infer(input_tensor)
        return outputs[0], outputs[1], outputs[2]

    def prepare_input(self, image: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = (img / 255.0 - mean) / std
        img = img.transpose(2, 0, 1)

        return img[np.newaxis, :, :, :].astype(np.float32)

    def infer(self, input_tensor):
        return self.session.run(
            self.output_names, {self.input_names[0]: input_tensor}
        )

    def get_input_details(self):
        m = self.session.get_inputs()[0]
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.input_shape = m.shape
        self.input_height = m.shape[2]
        self.input_width = m.shape[3]

    def get_output_details(self):
        self.output_names = [x.name for x in self.session.get_outputs()]


class SAM2ImageDecoder:
    def __init__(
        self,
        path: str,
        encoder_input_size: Tuple[int, int],
        orig_im_size: Tuple[int, int] = None,
        mask_threshold: float = 0.0,
    ) -> None:

        self.session = onnxruntime.InferenceSession(
            path, providers=onnxruntime.get_available_providers()
        )

        self.encoder_input_size = encoder_input_size
        self.orig_im_size = orig_im_size if orig_im_size else encoder_input_size
        self.scale_factor = 4

        self.get_input_details()
        self.get_output_details()

    def __call__(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ):

        inputs = self.prepare_inputs(
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            point_coords,
            point_labels,
        )

        outputs = self.infer(inputs)
        return self.process_output(outputs)

    # --------------------------
    # FIXED prepare_inputs()
    # --------------------------
    def prepare_inputs(
        self,
        image_embed: np.ndarray,
        high_res_feats_0: np.ndarray,
        high_res_feats_1: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
    ):

        pc, pl = self.prepare_points(point_coords, point_labels)

        mask_input = np.zeros(
            (
                pl.shape[0],
                1,
                self.encoder_input_size[0] // self.scale_factor,
                self.encoder_input_size[1] // self.scale_factor,
            ),
            dtype=np.float32,
        )

        has_mask_input = np.array([0], dtype=np.float32)

        orig_im_size = np.array(self.orig_im_size, dtype=np.int32)

        return (
            image_embed,
            high_res_feats_0,
            high_res_feats_1,
            pc,
            pl,
            mask_input,
            has_mask_input,
            orig_im_size,
        )

    def prepare_points(self, point_coords, point_labels):
        pc = point_coords[np.newaxis, ...]
        pl = point_labels[np.newaxis, ...]

        pc[..., 0] = pc[..., 0] / self.orig_im_size[1] * self.encoder_input_size[1]
        pc[..., 1] = pc[..., 1] / self.orig_im_size[0] * self.encoder_input_size[0]

        return pc.astype(np.float32), pl.astype(np.float32)

    def infer(self, inputs):
        return self.session.run(
            self.output_names,
            {self.input_names[i]: inputs[i] for i in range(len(self.input_names))}
        )

    def process_output(self, outputs):
        masks = outputs[0][0]
        scores = outputs[1].squeeze()

        best = masks[np.argmax(scores)]
        best = cv2.resize(best, (self.orig_im_size[1], self.orig_im_size[0]))

        return np.array([[best]]), scores

    def set_image_size(self, orig_im_size):
        self.orig_im_size = orig_im_size

    def get_input_details(self):
        self.input_names = [x.name for x in self.session.get_inputs()]

    def get_output_details(self):
        self.output_names = [x.name for x in self.session.get_outputs()]
