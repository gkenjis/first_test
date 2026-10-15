#!/usr/bin/env python3
"""Pure PyTorch inference/fine-tuning wrapper for PP-OCRv6 medium recognition.

Architecture loaded from Hugging Face Transformers:
    LCNetV4 + EncoderWithLightSVTR + CTC head
Weights:
    PaddlePaddle/PP-OCRv6_medium_rec_safetensors

No PaddlePaddle runtime is used. All model operations are torch.nn modules.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
from sympy import im
import torch
from PIL import Image
from ppocrv6_model import PPOCRV6Small
from transformers import PPOCRV6SmallRecForTextRecognition
from paddle_line import PPOCRv6Detector

def load_model(model_id_or_path: str,local_files_only: bool = False,):
    """Load the official safetensors into the official pure-PyTorch graph."""
    dtype = torch.float32
    model = PPOCRV6SmallRecForTextRecognition.from_pretrained( model_id_or_path,use_safetensors=True,local_files_only=local_files_only,torch_dtype=dtype)
    return model

def convert_safetensors_to_pt(model_id_or_path: str,local_files_only: bool = False):
    """Save pretrained safetensors weights as a PyTorch state_dict file."""
    model = PPOCRV6SmallRecForTextRecognition.from_pretrained(
        model_id_or_path,
        use_safetensors=True,
        local_files_only=local_files_only,
    )
    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }

    torch.save(state_dict, "ppocrv6_small.pt")
    return


def _preprocess_images(images: list[Image.Image], preprocessor_config: dict) -> torch.Tensor:
    """Resize, right-pad, and normalize RGB recognition images."""
    size = preprocessor_config["size"]
    pad_size = preprocessor_config.get("pad_size", size)
    target_height = int(size["height"])
    target_width = int(pad_size["width"])
    max_width = int(preprocessor_config.get("max_image_width", target_width))

    resized_images = []
    for image in images:
        image = image.convert("RGB")
        resized_width = max(1, math.ceil(image.width * target_height / image.height))
        resized_width = min(resized_width, max_width)
        resized_images.append(image.resize((resized_width, target_height), Image.Resampling.BILINEAR))

    batch_width = min(max_width, max(target_width, max(image.width for image in resized_images)))
    pixel_values = torch.zeros((len(resized_images), 3, target_height, batch_width), dtype=torch.float32)
    for index, image in enumerate(resized_images):
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32))
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0
        pixel_values[index, :, :, : image.width] = (image_tensor - 0.5) / 0.5

    return pixel_values


def _decode_ctc(probabilities: torch.Tensor, preprocessor_config: dict) -> list[dict[str, str | float]]:
    """Greedy CTC decode of model probabilities using the configured character list."""
    characters = preprocessor_config["character_list"]
    confidences, token_ids = probabilities.max(dim=-1)
    results = []

    for sequence, sequence_confidences in zip(token_ids, confidences):
        decoded = []
        selected_confidences = []
        previous_token = 0
        for token, confidence in zip(sequence.tolist(), sequence_confidences.tolist()):
            if token != 0 and token != previous_token:
                decoded.append(characters[token])
                selected_confidences.append(confidence)
            previous_token = token
        score = sum(selected_confidences) / len(selected_confidences) if selected_confidences else 0.0
        results.append({"text": "".join(decoded), "score": score})

    return results


@torch.inference_mode()
def recognize(line_image: Image.Image, model, preprocessor_config, device_obj):
    pixel_values = _preprocess_images([line_image], preprocessor_config).to(device_obj)
    pixel_values = pixel_values.to(dtype=next(model.parameters()).dtype)

    outputs = model(pixel_values=pixel_values)
    logits = outputs[0]
    decoded = _decode_ctc(logits, preprocessor_config)
    return decoded, logits

@torch.inference_mode()
def recognize_batch(line_images: list[Image.Image],model,preprocessor_config,device: torch.device):
    if not line_images:
        return [], None

    model_dtype = next(model.parameters()).dtype
    pixel_values = _preprocess_images(line_images, preprocessor_config)
    pixel_values = pixel_values.to(device=device, dtype=model_dtype)

    outputs = model(pixel_values=pixel_values)
    logits = outputs[0]

    results = _decode_ctc(logits, preprocessor_config)
    return results, logits

if __name__ == "__main__":
    detector = PPOCRv6Detector("inference.onnx", score_thresh=0.30, min_area=30)
    image, boxes = detector.predict("test.png")
    # convert_safetensors_to_pt(R"C:\projects\ppocr\model", local_files_only=True)
    model = PPOCRV6Small.from_files(R"C:\projects\ppocr\model\config.json",R"ppocrv6_small.pt")
    # model = load_model(R"C:\projects\ppocr\model", local_files_only=True)
    model.to(torch.device("cpu"))
    model.eval()
    
    with open(R"C:\projects\ppocr\model\preprocessor_config.json", encoding="utf-8") as config_file:
        preprocessor_config = json.load(config_file)

    line_images = [image.crop(box) for box in boxes]
    print(f"Total line images: {len(line_images)}")
    # for i in range(0, len(line_images), 5):
    #     batch = line_images[i:i+5]
    #     results, logits = recognize_batch(batch,model,preprocessor_config,device=torch.device("cpu"))
    #     for j,result in enumerate(results):
    #         print(f"{i+1+j}:{result['text']}")

    for i in range(len(line_images)):
        img = line_images[i]
        results, logits = recognize(img,model,preprocessor_config,device_obj=torch.device("cpu"))
        for j,result in enumerate(results):
            print(f"{i+1+j}:{result['text']}")
