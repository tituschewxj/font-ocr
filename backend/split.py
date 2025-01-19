import cv2
import easyocr
import os
import numpy as np

def enhance_contrast(image):
    """Increase text contrast using adaptive histogram equalization."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced

def split_characters(image_path, output_folder):
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if available
    
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
        # Extract bounding box coordinates
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

# Example Usage
input_image = "input_image.jpg"  # Path to the input image
output_folder = "output_characters"  # Folder to save the characters
split_characters(input_image, output_folder)
