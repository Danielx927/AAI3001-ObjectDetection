import os
from collections import defaultdict

def print_duplicate_images(folder):
    stems = defaultdict(list)
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            stem = os.path.splitext(fname)[0]
            stems[stem].append(fname)
    # Print duplicate stems
    count = 0
    for stem, files in stems.items():
        if len(files) > 1:
            count += 1
            print(f"Duplicate image name: {stem} -> {files}")
    print(count)

if __name__ == "__main__":
    folder = "data/unprocessed/Strawbs to label/Strawbs to label"  # Use '.' for current folder
    print_duplicate_images(folder)
