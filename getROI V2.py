import os
import cv2
import numpy as np
# Author: Morteza Farrokhnejad

# Define paths (update these to your actual directories)
input_dir = 'PATH TO POLYU GOES HERE'
output_dir = 'WHERE YOU WANT TO SAVE THE ROIS'          

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define image dimensions and new ROI size
image_width = 384
image_height = 284
roi_width = 165  # New width after trimming
roi_height = 180  # New height after trimming

# Function to extract and save the final adjusted ROI
def extract_and_save_final_adjusted_roi(image_path, output_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error reading image: {image_path}")
        return
    
    # Threshold to separate palm from background
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours of the palm region
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"No palm region detected in: {image_path}")
        return
    
    # Use the largest contour as the palm
    palm_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle for centroid calculation
    x, y, w, h = cv2.boundingRect(palm_contour)
    centroid_x = x + w // 2
    centroid_y = y + h // 2
    
    # Calculate new ROI starting coordinates
    new_roi_x_start = centroid_x - 65  # Shifted further right by 5 pixels
    new_roi_y_start = centroid_y - 90  # Shifted down by 10 pixels
    
    # Boundary checks to keep ROI within image
    new_roi_x_start = max(0, min(new_roi_x_start, image_width - roi_width))
    new_roi_y_start = max(0, min(new_roi_y_start, image_height - roi_height))
    
    # Extract the adjusted ROI
    roi = img[new_roi_y_start:new_roi_y_start + roi_height, 
              new_roi_x_start:new_roi_x_start + roi_width]
    
    # Save the ROI
    cv2.imwrite(output_path, roi)

# Process all images in the input directory
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.endswith('.bmp'):
            image_path = os.path.join(root, file)
            output_filename = file.replace('.bmp', '_ROI.bmp')
            output_path = os.path.join(output_dir, output_filename)
            extract_and_save_final_adjusted_roi(image_path, output_path)

print(f"Final adjusted ROI extraction complete. ROIs saved to {output_dir}")