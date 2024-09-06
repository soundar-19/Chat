import streamlit as st
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate image description
def generate_image_description(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))

    # Preprocess the image for BLIP
    inputs = processor(images=image, return_tensors="pt")

    # Generate caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return caption

def main():
    st.title("Image Description Generator")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read the image file
        image_bytes = uploaded_file.read()

        # Display the uploaded image
        st.image(image_bytes, caption='Uploaded Image', use_column_width=True)

        # Generate and display image description
        description = generate_image_description(image_bytes)
        st.write(f"Image Description: {description}")

if __name__ == "__main__":
    main()
