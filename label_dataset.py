import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
import csv
import json
import re

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def african_data():
    folder_dir = "AFRIFASHION1600"
    count = 0
    data = []
    for image in os.listdir(folder_dir):
        if (image.endswith(".png")):
            count += 1
            raw_image = Image.open(os.path.join(folder_dir,image))
            name_list = image.split("_")
            del name_list[-1]
            if "African" in name_list:
                name_list.remove("African")
            
            type = " ".join(name_list)
            text = f"an african {type.lower()} of"

            inputs = processor(raw_image, text, return_tensors="pt")

            out = model.generate(**inputs,max_new_tokens = 30)
            label = processor.decode(out[0], skip_special_tokens=True)
            data.append([image, label])
            print(f"{count}/1600")

    with open('image_label.csv', 'w', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
        writer = csv.writer(file)
        writer.writerows(data) # Use writerows for nested list

def indo_process_json():
    data = {}
    JSON_PATH = os.path.join("archive","test_data.json")
    with open(JSON_PATH,"r") as f:
        for line_number, line in enumerate(f, 1):
            try:
                json_object = json.loads(line)
                class_ = json_object["class_label"]
                class_ = class_.replace("_", " ")
                class_ = re.sub(r'\bmen\b', '', class_, flags=re.IGNORECASE)
                class_ = re.sub(r'\bwomen\b', '', class_, flags=re.IGNORECASE)
                data[json_object["image_path"]] = class_
            except json.JSONDecodeError as e:
                print(f"Error processing line {line_number}: {e}")

    with open("indo_type", 'w') as file:
        json.dump(data, file, indent=4)

def indo_data():
    folder_dir = os.path.join("archive","images","test")
    count = 0
    data = []
    total = len(os.listdir(folder_dir))
    with open("indo_type","r") as f:
        type_json = json.load(f)
        for image in os.listdir(folder_dir):
            if (image.endswith(".jpeg")):
                count += 1
                raw_image = Image.open(os.path.join(folder_dir,image))
                type = type_json[os.path.join("images","test",image)]
                text = f"an indian {type} of"

                inputs = processor(raw_image, text, return_tensors="pt")

                out = model.generate(**inputs,max_new_tokens = 30)
                label = processor.decode(out[0], skip_special_tokens=True)
                data.append([image, label])
                print(label)
                print(f"{count}/{total}")

        with open('indo_image_label.csv', 'w', newline='') as file:
            # Step 4: Using csv.writer to write the list to the CSV file
            writer = csv.writer(file)
            writer.writerows(data) # Use writerows for nested list

indo_data()