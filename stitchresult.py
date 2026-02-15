import os
from PIL import Image

def stitch_images(folder_path):
    """
    For each pair:
        XXXXX_out.png (left)
        XXXXX_R.png   (right)

    Create:
        XXXXX.png  (stitched horizontally)
    inside folder: stitched/
    """

    stitched_folder = os.path.join(folder_path, "stitched")
    os.makedirs(stitched_folder, exist_ok=True)

    files = os.listdir(folder_path)

    # Create a set for fast lookup
    file_set = set(files)

    for filename in files:
        if filename.endswith("_out.png"):
            base_name = filename.replace("_out.png", "")
            right_name = base_name + "_R.png"

            if right_name in file_set:
                left_path = os.path.join(folder_path, filename)
                right_path = os.path.join(folder_path, right_name)

                # Open images
                left_img = Image.open(left_path)
                right_img = Image.open(right_path)

                # Ensure same height
                if left_img.height != right_img.height:
                    print(f"Skipping {base_name} (height mismatch)")
                    continue

                # Create new stitched image
                total_width = left_img.width + right_img.width
                height = left_img.height

                stitched_img = Image.new("RGB", (total_width, height))
                stitched_img.paste(left_img, (0, 0))
                stitched_img.paste(right_img, (left_img.width, 0))

                # Save stitched result
                output_path = os.path.join(stitched_folder, base_name + ".png")
                stitched_img.save(output_path)

                print(f"Stitched: {base_name}.png")

    print("\nDone stitching.")


if __name__ == "__main__":
    folder_path = "/path/to/your/folder"  # 🔹 change this
    stitch_images(folder_path)
