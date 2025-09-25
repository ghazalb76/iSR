import os
from PIL import Image

# Define the directories for left and right images
left_dir = '/localhome/gba50/Finalize/dataset/test/cr/resultsOfRTX4000/Scale10/L'
right_dir = '/localhome/gba50/Finalize/dataset/test/cr/resultsOfRTX4000/Scale10/R'
output_dir = '/localhome/gba50/Finalize/dataset/test/cr/resultsOfRTX4000/Scale10/stitched'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of images in left directory
left_images = sorted([img for img in os.listdir(left_dir) if img.endswith('.png')])

# Process each image
for img_name in left_images:
    # Define paths for left and right images
    left_img_path = os.path.join(left_dir, img_name)
    right_img_path = os.path.join(right_dir, img_name)

    # Check if the corresponding right image exists
    if not os.path.exists(right_img_path):
        print(f"Warning: {img_name} does not have a corresponding image in the right directory.")
        continue

    # Open images
    left_img = Image.open(left_img_path)
    right_img = Image.open(right_img_path)

    # Ensure both images have the same height
    if left_img.size[1] != right_img.size[1]:
        print(f"Warning: {img_name} has mismatched heights. Skipping.")
        continue

    # Create a new image with the width of both images combined and the height of one of them
    stitched_img = Image.new('RGB', (left_img.width + right_img.width, left_img.height))
    stitched_img.paste(left_img, (0, 0))    
    stitched_img.paste(right_img, (left_img.width, 0))

    # Save the stitched image
    stitched_img.save(os.path.join(output_dir, img_name))

print("Image stitching complete.")
