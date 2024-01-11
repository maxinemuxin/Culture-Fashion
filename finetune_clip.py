# credit: https://www.kaggle.com/code/zacchaeus/clip-finetune

from PIL import Image
import torch
from torch import nn, optim
import glob
import os
import pandas as pd
import json
import numpy as np
import clip
from torch.utils.data import Dataset, DataLoader, BatchSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from matplotlib.pyplot import imshow
import torchtext
import nltk, re, string, collections
from nltk.util import ngrams
import collections
import csv
BATCH_SIZE = 128
EPOCH = 5

# get african data
IMG_ROOT = "AFRIFASHION1600"
CSV_PATH = "image_label.csv"
img_paths_african = glob.glob(os.path.join(IMG_ROOT, "*.png"))
d = {}
label_dict = {}
with open(CSV_PATH, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        key = row[0]
        value = row[1]
        label_dict[key] = value
for i, img_path in enumerate(img_paths_african):
    name = img_path.split("/")[-1]
    caption = label_dict[name]
    d[img_path] = caption


# get indian data
IMG_ROOT = os.path.join("archive", "test")
JSON_PATH = os.path.join("archive","test_data.json")
img_paths_indo = glob.glob(os.path.join(IMG_ROOT,"*.jpeg"))
with open(JSON_PATH,"r") as f:
    captions = json.load(f)
    import pdb
    pdb.trace()
    # for i, img_path in enumerate(img_paths_indo):
    #     caption = captions.

# split validation
train_img_paths, test_img_paths = train_test_split(img_paths, test_size=0.2, random_state=42)
d_train = {k: d[k] for k in train_img_paths}
d_test = {k: d[k] for k in test_img_paths}
print(len(d_train), len(d_test))

# load pretrained model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# define dataset
class CultureDataset(Dataset):
    def __init__(self, data, preprocess):
        self.preprocess = preprocess
        self.img_paths = []
        self.captions = []
        for img_path, caption in data.items():
            self.img_paths.append(img_path)
            self.captions.append(caption)
        self.processed_cache = {}
        for img_path in data:
            self.processed_cache[img_path] = self.preprocess(Image.open(img_path))
        self.img_paths_set = list(data.keys())
        self.path2label = {path: self.img_paths_set.index(path) for path in self.img_paths_set}
        
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self.processed_cache[img_path]
        caption = self.captions[idx]
        label = self.path2label[img_path]
        return image, caption, label

train_dataset = CultureDataset(d_train, preprocess)
test_dataset = CultureDataset(d_test, preprocess)


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

if device == "cpu":
    model.float()

# train CLIP
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader)*EPOCH)

best_te_loss = 1e5
best_ep = -1
for epoch in range(EPOCH):
    print(f"running epoch {epoch}, best test loss {best_te_loss} after epoch {best_ep}")
    step = 0
    tr_loss = 0
    model.train()
    pbar = tqdm(train_dataloader, leave=False)
    for batch in pbar:
        step += 1
        optimizer.zero_grad()

        images, texts, _ = batch
        images = images.to(device)
        texts = clip.tokenize(texts).to(device)
        logits_per_image, logits_per_text = model(images, texts)
        ground_truth = torch.arange(BATCH_SIZE).to(device)

        total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
        total_loss.backward()
        tr_loss += total_loss.item()
        if device == "cpu":
            optimizer.step()
            scheduler.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            scheduler.step()
            clip.model.convert_weights(model)
        pbar.set_description(f"train batchCE: {total_loss.item()}", refresh=True)
    tr_loss /= step
    
    step = 0
    te_loss = 0
    with torch.no_grad():
        model.eval()
        test_pbar = tqdm(test_dataloader, leave=False)
        for batch in test_pbar:
            step += 1
            images, texts, _ = batch
            images = images.to(device)
            texts = clip.tokenize(texts).to(device)
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(images.size(0)).to(device)


            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            te_loss += total_loss.item()
            test_pbar.set_description(f"test batchCE: {total_loss.item()}", refresh=True)
        te_loss /= step
        
    if te_loss < best_te_loss:
        best_te_loss = te_loss
        best_ep = epoch
        torch.save(model.state_dict(), "best_model.pt")
    print(f"epoch {epoch}, tr_loss {tr_loss}, te_loss {te_loss}")
torch.save(model.state_dict(), "last_model.pt")