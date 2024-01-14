# credit:
# https://huggingface.co/jinaai/jina-embeddings-v2-base-en
# https://github.com/patrickjohncyh/fashion-clip/tree/master

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
from fashion_project import finetune_clip
import torch
import clip
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoModel,CLIPProcessor, CLIPModel
from numpy.linalg import norm
from fashion_project import classifier
from collections import defaultdict
import shutil


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///closet.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads/'  # Relative path to the uploads folder
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



db = SQLAlchemy(app)


# Model definition
class ClothingItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(80), nullable=False)
    image_filename = db.Column(db.String(120), nullable=False)

# Create the database tables
with app.app_context():
    db.drop_all()
    db.create_all()

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def clear_upload_folder():
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

clear_upload_folder()

# Route to upload files
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'No selected file or invalid file type'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    abs_file_path = os.path.abspath(file_path)
    file.save(abs_file_path)

    # Here you would implement your logic to process the image and determine its type
    # For this example, let's assume a function 'process_image' exists and returns the type
    image_type = classifier.classify(abs_file_path)  # This should be replaced with the actual result of process_image(file_path)
    
    # Save the new clothing item to the database
    new_item = ClothingItem(category=image_type, image_filename=filename)
    db.session.add(new_item)
    db.session.commit()

    # Return the URL path for the image and the determined type
    url_path = os.path.join('/uploads', filename)
    return jsonify({'extracted_image_url': url_path, 'type': image_type}), 200

# Route to retrieve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/get-outfit', methods=['POST'])
def get_outfit():
    description = request.form.get('description')
    
    # filter description
    cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method
    embeddings = model.encode([description, 'cultural and ethnic'])
    cultural_related = (cos_sim(embeddings[0], embeddings[1]))
    if cultural_related >= 0.73:
        model, prepocess = finetune_clip.load_model("../best_model.pt")
    else:
        model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
        prepocess = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    use_default = request.form.get('useDefault') == 'true'
    model, prepocess = finetune_clip.load_model("../best_model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = clip.tokenize([description]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text)
    if use_default:
        highest_similarity = -1  # Initialize with the lowest possible similarity
        best_combination = None
        for i in range(1,4):
            top = os.path.join("static","default_closet",f"top_{i}.jpeg")
            for j in range(1,4):
                bottom = os.path.join("static","default_closet",f"bottom_{j}.jpeg")
                for k in range(1,4):
                    shoe = os.path.join("static","default_closet",f"shoe_{k}.jpeg")
                    for l in range(1,4):
                        accessory = os.path.join("static","default_closet",f"accessory_{l}.jpeg")
                        image_pathes = [top, bottom, shoe, accessory]
                        images = [Image.open(path) for path in image_pathes]

                        # Define a new size (you can choose the size that best fits your needs)
                        new_size = (200, 200)  # Example: 200x200 pixels

                        # Resize all images to the new size
                        resized_images = [img.resize(new_size, Image.Resampling.LANCZOS) for img in images]


                        # Create a new image with enough space to hold all four images (2x2 layout)
                        combined_image = Image.new('RGB', (new_size[0] * 2, new_size[1] * 2))

                        # Paste each image into the appropriate position
                        combined_image.paste(resized_images[0], (0, 0))
                        combined_image.paste(resized_images[1], (new_size[0], 0))
                        combined_image.paste(resized_images[2], (0, new_size[1]))
                        combined_image.paste(resized_images[3], (new_size[0], new_size[1]))
                        image = prepocess(combined_image).unsqueeze(0).to(device)
                        # Save or display the new image
                        # combined_image.save('combined_image.jpg')
                        # combined_image.show()
                        image_embedding = model.encode_image(image)
                        cosine_similarity = torch.nn.functional.cosine_similarity(image_embedding, text_embedding)
                        if cosine_similarity > highest_similarity:
                            highest_similarity = cosine_similarity
                            best_combination = image_pathes
        outfit_urls = {"top":best_combination[0],"bottom":best_combination[1], "shoes":best_combination[2],"accessory":best_combination[3]}

    else:
        outfit_urls = {}
        all_clothings = ClothingItem.query.all()
        clothings_by_type = defaultdict(list)
        for item in all_clothings:
            clothings_by_type[item.category].append(item)
        for type in clothings_by_type:
            highest_similarity = -1
            best = None
            for item in clothings_by_type[type]:
                img_path = os.path.join("static","uploads",item.image_filename)
                image = Image.open(img_path)
                image = prepocess(image).unsqueeze(0).to(device)
                image_embedding = model.encode_image(image)
                cosine_similarity = torch.nn.functional.cosine_similarity(image_embedding, text_embedding)
                if cosine_similarity > highest_similarity:
                    highest_similarity = cosine_similarity
                    best = img_path
            outfit_urls[type] = best
    # Assuming you have some logic that populates the outfit dictionary
    # with the chosen images

    # outfit_urls = {key: os.path.join('/uploads', getattr(item, 'image_filename', 'default.png'))
    #                for key, item in outfit.items()}
    return jsonify(outfit_urls)



@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
