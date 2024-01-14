# credit:
# https://github.com/patrickjohncyh/fashion-clip/tree/master

import numpy as np
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

def classify(image_path):
    image = Image.open(image_path)
    model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    prepocess = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    inputs = prepocess(text=["t-shirt, shirt, dress, jacket, tank, coat", "skirt, pant, shorts, jeans","shoes","bag, watch, purse, jewelry, belt"],
                    images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  
    index =torch.argmax(probs)
    return(["top","bottom","shoes","accessory"][index])