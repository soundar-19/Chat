from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import io
import os
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Define the path for storing uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file uploaded", 400

        file = request.files["image"]
        if file.filename == "":
            return "No selected file", 400

        if file:
            image_bytes = file.read()
            description = generate_image_description(image_bytes)
            
            # Save the image
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(file_path, 'wb') as f:
                f.write(image_bytes)

            # Serve the image
            uploaded_image_url = f'/uploads/{filename}'
            return render_template("index.html", result=description, uploaded_image=uploaded_image_url)

    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def generate_image_description(image_bytes):
    # Open the image using PIL
    image = Image.open(io.BytesIO(image_bytes))

    # Preprocess the image for BLIP
    inputs = processor(images=image, return_tensors="pt")

    # Generate caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return f"Image Description: {caption}"

if __name__ == "__main__":
    app.run(debug=True)
