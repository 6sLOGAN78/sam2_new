import argparse
import sys
import json
import pathlib

sys.path.append(".")

import cv2
import numpy as np
from samexporter.sam2_onnx import SegmentAnything2ONNX


# ----------------------------
# Resize only for display
# ----------------------------
def resize_for_display(image, max_width=1000, max_height=700):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale >= 1:
        return image.copy()
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h))


# ----------------------------
# Argparse setup
# ----------------------------
def str2bool(v):
    return v.lower() in ("true", "1")


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--encoder_model",
    type=str,
    default="output_models/sam_vit_h_4b8939.encoder.onnx",
    help="Path to the ONNX encoder model",
)
argparser.add_argument(
    "--decoder_model",
    type=str,
    default="output_models/sam_vit_h_4b8939.decoder.onnx",
    help="Path to the ONNX decoder model",
)
argparser.add_argument(
    "--image",
    type=str,
    default="images/truck.jpg",
    help="Path to the image",
)
argparser.add_argument(
    "--prompt",
    type=str,
    default="images/truck.json",
    help="Path to the prompt JSON",
)
argparser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Path to the output image",
)
argparser.add_argument(
    "--show",
    action="store_true",
    help="Show the result",
)
argparser.add_argument(
    "--sam_variant",
    type=str,
    default="sam",
    help="Variant of SAM model. Options: sam, sam2",
)
args = argparser.parse_args()


# ----------------------------
# Model selection
# ----------------------------
model = None
if args.sam_variant == "sam2":
    model = SegmentAnything2ONNX(
        args.encoder_model,
        args.decoder_model,
    )

# ----------------------------
# Load Image & Prompt
# ----------------------------
image = cv2.imread(args.image)
if image is None:
    print("Error: Could not load input image.")
    sys.exit(1)

prompt = json.load(open(args.prompt))

# ----------------------------
# Run SAM2
# ----------------------------
embedding = model.encode(image)
masks = model.predict_masks(embedding, prompt)

# ----------------------------
# Merge masks into RGB mask
# ----------------------------
mask = np.zeros((masks.shape[2], masks.shape[3], 3), dtype=np.uint8)
for m in masks[0, :, :, :]:
    mask[m > 0.5] = [255, 0, 0]   # red mask

# ----------------------------
# Overlay mask on image
# ----------------------------
visualized = cv2.addWeighted(image, 0.5, mask, 0.5, 0)

# ----------------------------
# Draw prompt points or rectangles
# ----------------------------
for p in prompt:
    if p["type"] == "point":
        color = (0, 255, 0) if p["label"] == 1 else (0, 0, 255)
        cv2.circle(visualized, (p["data"][0], p["data"][1]), 10, color, -1)
    elif p["type"] == "rectangle":
        cv2.rectangle(
            visualized,
            (p["data"][0], p["data"][1]),
            (p["data"][2], p["data"][3]),
            (0, 255, 0),
            2,
        )

# ----------------------------
# Save output
# ----------------------------
if args.output is not None:
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, visualized)

# ----------------------------
# Display resized window safely
# ----------------------------
if args.show:
    display_img = resize_for_display(visualized)
    cv2.imshow("Result", display_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
