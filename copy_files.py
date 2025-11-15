import os
import shutil

def copy_files(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        if os.path.isfile(src_path):
            shutil.copy(src_path, dst_path)
            print(f"Copied {fname} to {dst_dir}")

if __name__ == "__main__":
    src_dir = "data/unprocessed labels/Labelled Strawberries"    # Replace with your source folder path
    dst_dir = "data/input/label"  # Replace with your target folder path
    copy_files(src_dir, dst_dir)
