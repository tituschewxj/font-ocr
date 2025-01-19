from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from torchvision import transforms
import cv2
import easyocr
import os
import numpy as np
import tensorflow as tf

# Load the model from the .h5 file
with tf.device('/CPU:0'):
    model = tf.keras.models.load_model('path_to_model.h5')

app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your frontend URL for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def enhance_contrast(image):
    """Increase text contrast using adaptive histogram equalization."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def split_characters(image_path, output_folder):
    """Split characters and save them as individual images."""
    reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=True if available
    
    # Load and enhance image contrast
    image = cv2.imread(image_path)
    enhanced_image = enhance_contrast(image)
    
    # Perform OCR with character-level detail
    results = reader.readtext(enhanced_image, detail=1, paragraph=False)
    
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    char_count = 0
    for bbox, text, confidence in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        x_min = int(min(top_left[0], bottom_left[0]))
        y_min = int(min(top_left[1], top_right[1]))
        x_max = int(max(top_right[0], bottom_right[0]))
        y_max = int(max(bottom_left[1], bottom_right[1]))
        
        # Crop the character from the image
        cropped = enhanced_image[y_min:y_max, x_min:x_max]
        
        # Skip if the cropped region is too small (filter noise)
        if cropped.shape[0] < 5 or cropped.shape[1] < 5:
            continue
        
        # Save the cropped character as a separate image
        char_count += 1
        output_path = os.path.join(output_folder, f"char_{char_count}.png")
        cv2.imwrite(output_path, cropped)
        print(f"Saved: {output_path}")
    
    print(f"Processed {char_count} characters.")

@app.post("/classify/")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocessing the image
        image_path = "temp_image.jpg"
        image.save(image_path)

        # Enhance contrast and split characters
        output_folder = "output_characters"
        split_characters(image_path, output_folder)

        # Load all cropped character images and run the model on them
        char_predictions = []
        for char_image_path in os.listdir(output_folder):
            char_image_full_path = os.path.join(output_folder, char_image_path)
            char_image = Image.open(char_image_full_path).convert("RGB")

            # Preprocess character image for classification using TensorFlow's standard resize and normalization
            char_image = char_image.resize((256, 256))  # Resize to model's expected input size
            char_image_array = np.array(char_image) / 255.0  # Normalize to [0, 1]

            # Add batch dimension: TensorFlow expects a 4D input tensor
            # char_image_array = np.expand_dims(char_image_array, axis=0)

            # Run model on character image
            prediction = model.predict(char_image_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            prob = np.max(prediction)

            char_predictions.append((predicted_class, prob))

        # Select the prediction with the highest probability
        if not char_predictions:
            return JSONResponse({"error": "No characters found in image."}, status_code=400)

        best_prediction = max(char_predictions, key=lambda x: x[1])  # Max by probability
        best_class, best_prob = best_prediction

        return JSONResponse({"best_prediction": str(best_class), "probability": float(best_prob)})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
