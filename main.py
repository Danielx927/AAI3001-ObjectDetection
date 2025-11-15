import os
import shutil
from collections import defaultdict

def collect_images_for_labels(label_root, image_root, output_dir):
    # label_root: directory with folders labelled pear and labelled pomegranate (each with label txt files)
    # image_root: directory containing three subfolders (with actual images)
    # output_dir: where you want the matched images to be moved/copied

    os.makedirs(output_dir, exist_ok=True)
    label_files = []

    # Step 1: Get all label filenames (without extension) from both label folders
    for label_subdir in os.listdir(label_root):
        label_dir = os.path.join(label_root, label_subdir)
        for fname in os.listdir(label_dir):
            if fname.endswith(".txt"):
                label_files.append(os.path.splitext(fname)[0])

    found_images = defaultdict(list)

    # Step 2: Search the three image subdirectories for matching image files
    for img_subdir in os.listdir(image_root):
        img_dir_path = os.path.join(image_root, img_subdir)
        for img_subdir2 in os.listdir(img_dir_path):
            img_dir_path2 = os.path.join(img_dir_path, img_subdir2)
            for fname in os.listdir(img_dir_path2):
                stem, ext = os.path.splitext(fname)
                if stem in label_files and ext.lower() in [".jpg", ".jpeg", ".png"]:
                    # Track duplicates, okay to copy all
                    found_images[stem].append(os.path.join(img_dir_path2, fname))
                    # Copy/move image into output_dir
                    shutil.copy(os.path.join(img_dir_path2, fname), os.path.join(output_dir, fname))

            # Step 3: Report images with duplicated names
            duplicates = [name for name, files in found_images.items() if len(files) > 1]
            print("Duplicate image names (these have more than one matching file):")
            for name in duplicates:
                print(name)

if __name__ == "__main__":
    label_root = "data/unprocessed"
    image_root = "unprocessed"
    output_dir = "output_images"
    collect_images_for_labels(label_root, image_root, output_dir)
