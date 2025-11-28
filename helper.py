import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from PIL import Image
import random
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np

def split_files_by_extension(src_folder, ext_to_targetfolder):
    """
    Moves files from src_folder into target folders according to their extension.
    ext_to_targetfolder: dict, e.g. {'.jpg': 'images', '.txt': 'labels', '.xml': 'annotations'}
    """
    for ext, target_folder in ext_to_targetfolder.items():
        os.makedirs(target_folder, exist_ok=True)

    for fname in os.listdir(src_folder):
        _, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext in ext_to_targetfolder:
            shutil.copy(
                os.path.join(src_folder, fname),
                os.path.join(ext_to_targetfolder[ext], fname)
            )

def rename_images_and_labels(images_folder, labels_folder, prefix='img_'):
    """Renames images and their corresponding label files with a consistent prefix and zero-padded index."""
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()  # Ensure predictable order
    idx = 1  # Counter for new names

    for img_file in image_files:
        label_name = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_folder, label_name)
        if os.path.exists(label_path):
            # Set up new base name
            new_base = f"{prefix}{str(idx).zfill(3)}"
            img_ext = os.path.splitext(img_file)[1]
            new_img_name = f"{new_base}{img_ext}"
            # Rename image
            old_img_path = os.path.join(images_folder, img_file)
            new_img_path = os.path.join(images_folder, new_img_name)
            os.rename(old_img_path, new_img_path)
            # Rename label
            new_label_name = f"{new_base}.txt"
            new_label_path = os.path.join(labels_folder, new_label_name)
            os.rename(label_path, new_label_path)
            idx += 1
        else:
            print(f"Skipped: {img_file} (no matching label)")

def count_files_by_extension(directory):
    ext_counter = Counter()
    for fname in os.listdir(directory):
        _, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext:
            ext_counter[ext] += 1
    for extension, count in ext_counter.items():
        print(f"{extension}: {count}")
    return ext_counter

def voc_to_yolo(xml_file, class_map, output_txt):
    """ Convert VOC XML annotation to YOLO format text file."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    yolo_lines = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        class_idx = class_map.get(name)
        if class_idx is None:
            continue  # Skip if not in map
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        
        yolo_line = f"{class_idx} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}\n"
        yolo_lines.append(yolo_line)

    with open(output_txt, 'w') as f:
        f.writelines(yolo_lines)

def batch_convert_voc_to_yolo(input_folder, output_folder, class_map):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if fname.lower().endswith('.xml'):
            xml_path = os.path.join(input_folder, fname)
            base = os.path.splitext(fname)[0]
            txt_path = os.path.join(output_folder, base + '.txt')
            voc_to_yolo(xml_path, class_map, txt_path)

def remove_verified_xml(folder):
    """ Removes XML files in the folder if a corresponding TXT file exists."""
    for fname in os.listdir(folder):
        if fname.lower().endswith('.xml'):
            base = os.path.splitext(fname)[0]
            txt_name = base + '.txt'
            txt_path = os.path.join(folder, txt_name)
            if os.path.exists(txt_path):
                xml_path = os.path.join(folder, fname)
                os.remove(xml_path)
                print(f"Deleted: {xml_path}")
            else:
                print(f"Skipped (no TXT): {fname}")

def fix_voc_xml_sizes(xml_folder, image_folder, image_exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """Fixes width and height in VOC XML files based on actual image sizes."""
    for fname in os.listdir(xml_folder):
        if fname.lower().endswith(".xml"):
            base = os.path.splitext(fname)[0]
            # Find image file with possible extension
            img_path = None
            for ext in image_exts:
                candidate = os.path.join(image_folder, base + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if not img_path:
                print(f"Image for {fname} not found.")
                continue
            # Read image size
            try:
                with Image.open(img_path) as im:
                    true_w, true_h = im.size
            except Exception as e:
                print(f"Error opening {img_path}: {e}")
                continue
            # Update XML
            xml_path = os.path.join(xml_folder, fname)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            xml_w = int(size.find('width').text)
            xml_h = int(size.find('height').text)
            if xml_w != true_w or xml_h != true_h:
                size.find('width').text = str(true_w)
                size.find('height').text = str(true_h)
                tree.write(xml_path)
                print(f"Fixed {fname}: set width={true_w}, height={true_h}")
            else:
                print(f"No change for {fname}")

def filter_and_extract_images_by_class(
    data_root_dir,
    class_indices_to_keep,
    output_images_dir,
    output_labels_dir,
    target_num_labels,
    image_extensions=(".jpg", ".jpeg", ".png")
):
    """ Filters and extracts images and labels containing specified class indices."""
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    found = 0

    for subset_folder in os.listdir(data_root_dir):
        subset_path = os.path.join(data_root_dir, subset_folder)
        if not os.path.isdir(subset_path):
            continue
        labels_dir = os.path.join(subset_path, "labels")
        images_dir = os.path.join(subset_path, "images")
        if not (os.path.exists(labels_dir) and os.path.exists(images_dir)):
            continue

        for lbl_file in os.listdir(labels_dir):
            if not lbl_file.endswith(".txt"):
                continue
            label_path = os.path.join(labels_dir, lbl_file)
            with open(label_path, "r") as f:
                lines = f.readlines()
            # Filter for relevant class indices only
            filtered_lines = [
                line for line in lines
                if line.strip() and int(line.split()[0]) in class_indices_to_keep
            ]
            # Only proceed if at least one relevant class instance found
            if filtered_lines:
                # Save new filtered label file
                out_lbl_path = os.path.join(output_labels_dir, lbl_file)
                with open(out_lbl_path, "w") as out_f:
                    out_f.writelines(filtered_lines)
                # Copy matching image file
                img_base = os.path.splitext(lbl_file)[0]
                img_copied = False
                for ext in image_extensions:
                    img_path = os.path.join(images_dir, img_base + ext)
                    if os.path.exists(img_path):
                        shutil.copy(img_path, os.path.join(output_images_dir, img_base + ext))
                        img_copied = True
                        break
                if img_copied:
                    found += 1
                if found >= target_num_labels:
                    print(f"Collected {found} labels/images with specified classes.")
                    return
    print(f"Collected {found} labels/images with specified classes (target was {target_num_labels}).")

def count_labels_per_class(labels_folder, images_folder, class_mapping, image_exts=(".jpg", ".jpeg", ".png")):
    """ Counts the number of labeled objects per class in YOLO format label files."""
    # Build index to name mapping for output
    idx_to_name = {idx: name for name, idx in class_mapping.items()}
    class_counter = Counter()
    invalid_files = []
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
    for label_fname in label_files:
        base = os.path.splitext(label_fname)[0]
        found_image = False
        for ext in image_exts:
            image_path = os.path.join(images_folder, base + ext)
            if os.path.exists(image_path):
                found_image = True
                break
        if not found_image:
            invalid_files.append(label_fname)
            continue
        # Count labels in file
        with open(os.path.join(labels_folder, label_fname), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_idx = int(parts[0])
                        class_counter[class_idx] += 1
                    except ValueError:
                        continue
    # Print summary with class names
    print("Total object counts by class:")
    for cls, count in sorted(class_counter.items()):
        name = idx_to_name.get(cls, "Unknown")
        print(f"Class {cls} (\"{name}\"): {count}")
    if invalid_files:
        print("\nLabel files with missing image counterparts:")
        for fname in invalid_files:
            print(fname)


def count_labels_per_class_v2(labels_folder, images_folder, class_mapping, image_exts=(".jpg", ".jpeg", ".png")):
    """
    Enhanced version: Counts labeled objects per class with improved formatting.
    Returns the class counter for programmatic use.
    """
    # Build index to name mapping for output
    idx_to_name = {idx: name for name, idx in class_mapping.items()}
    class_counter = Counter()
    invalid_files = []
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]
    
    for label_fname in label_files:
        base = os.path.splitext(label_fname)[0]
        found_image = False
        for ext in image_exts:
            image_path = os.path.join(images_folder, base + ext)
            if os.path.exists(image_path):
                found_image = True
                break
        if not found_image:
            invalid_files.append(label_fname)
            continue
        # Count labels in file
        with open(os.path.join(labels_folder, label_fname), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_idx = int(parts[0])
                        class_counter[class_idx] += 1
                    except ValueError:
                        continue
    
    # Print enhanced summary with better formatting
    total_objects = sum(class_counter.values())
    print("=" * 60)
    print(f"📊 Object Distribution Summary")
    print(f"   Labels Folder: {labels_folder}")
    print(f"   Images Folder: {images_folder}")
    print("=" * 60)
    print(f"{'Class ID':<10} {'Class Name':<20} {'Count':<10} {'%':<10}")
    print("-" * 60)
    
    for cls, count in sorted(class_counter.items()):
        name = idx_to_name.get(cls, "Unknown")
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"{cls:<10} {name:<20} {count:<10} {percentage:>6.2f}%")
    
    print("-" * 60)
    print(f"{'TOTAL':<10} {'':<20} {total_objects:<10} {'100.00%':>10}")
    print("=" * 60)
    
    if invalid_files:
        print(f"\n⚠️  Warning: {len(invalid_files)} label file(s) without matching images")
        if len(invalid_files) <= 10:
            for fname in invalid_files:
                print(f"   - {fname}")
        else:
            print(f"   Showing first 10 of {len(invalid_files)}:")
            for fname in invalid_files[:10]:
                print(f"   - {fname}")
            print(f"   ... and {len(invalid_files) - 10} more")
    
    return class_counter

def delete_labels_without_images(labels_folder, images_folder, image_extensions=(".jpg", ".jpeg", ".png")):
    """ Deletes label files that do not have a corresponding image file."""
    deleted = []
    for lbl_file in os.listdir(labels_folder):
        if not lbl_file.endswith(".txt"):
            continue
        base = os.path.splitext(lbl_file)[0]
        has_image = False
        for ext in image_extensions:
            img_path = os.path.join(images_folder, base + ext)
            if os.path.exists(img_path):
                has_image = True
                break
        if not has_image:
            lbl_path = os.path.join(labels_folder, lbl_file)
            os.remove(lbl_path)
            deleted.append(lbl_file)
    print(f"Deleted {len(deleted)} label files without matching images:")
    for lbl in deleted:
        print(lbl)

def relabel_yolo_labels(labels_folder, label_map):
    """
    Change class indices in YOLO .txt label files according to label_map.
    label_map: dict, e.g., {0: 5, 1: 5} to convert apples and bananas to watermelon (5).
    """
    changed = 0
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]

    for lbl_file in label_files:
        lbl_path = os.path.join(labels_folder, lbl_file)
        with open(lbl_path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_idx = int(parts[0])
            target_idx = label_map.get(class_idx, class_idx)  # Default: keep original
            parts[0] = str(target_idx)
            new_lines.append(" ".join(parts) + '\n')
        with open(lbl_path, 'w') as f:
            f.writelines(new_lines)
        changed += 1
    print(f"Relabeled {changed} label files.")

def count_image_classes(labels_folder):
    """
    Maps each label file to the Counter of classes present in it.
    Returns: dict(label_filename -> Counter({class_idx: count, ...}))
    """
    image_class_map = {}
    for lbl_file in os.listdir(labels_folder):
        if not lbl_file.endswith(".txt"):
            continue
        full_path = os.path.join(labels_folder, lbl_file)
        class_counter = Counter()
        with open(full_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    try:
                        class_idx = int(parts[0])
                        class_counter[class_idx] += 1
                    except Exception:
                        continue
        image_class_map[lbl_file] = class_counter
    return image_class_map

def balance_classes(image_class_map, target_num_per_class, seed=42):
    """
    Selects images such that the total number of each class is <= target_num_per_class.
    Returns set of label files selected.
    """
    random.seed(seed)
    all_files = list(image_class_map.keys())
    random.shuffle(all_files)
    class_totals = Counter()
    selected_files = set()
    for lbl in all_files:
        c_map = image_class_map[lbl]
        # Check if adding this image will exceed any class's limit
        exceeds = False
        for c, count in c_map.items():
            if class_totals[c] + count > target_num_per_class:
                exceeds = True
                break
        if not exceeds:
            selected_files.add(lbl)
            for c, count in c_map.items():
                class_totals[c] += count
        # Early exit: all classes hit target
        if all(class_totals[c] >= target_num_per_class for c in range(max(class_totals)+1)):
            break
    return selected_files

def iterative_train_val_split(label_files, image_class_map, n_classes, val_ratio=0.2, seed=42):
    """
    Split label files into train/val using iterative multi-label stratification.
    """
    # Build indicator matrix
    label_files = list(label_files)
    y = np.zeros((len(label_files), n_classes), dtype=int)
    for i, lbl in enumerate(label_files):
        classes = image_class_map[lbl]
        for c in classes:
            y[i, c] = 1
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    idx_train, idx_val = next(msss.split(np.zeros(len(label_files)), y))
    train_set = set([label_files[i] for i in idx_train])
    val_set = set([label_files[i] for i in idx_val])
    return train_set, val_set


def save_balanced_split(selected_files, labels_folder, images_folder, out_labels, out_images, image_exts=(".jpg", ".jpeg", ".png")):
    os.makedirs(out_labels, exist_ok=True)
    os.makedirs(out_images, exist_ok=True)
    for lbl_file in selected_files:
        base = os.path.splitext(lbl_file)[0]
        # Copy label
        shutil.copy(os.path.join(labels_folder, lbl_file), os.path.join(out_labels, lbl_file))
        # Copy image (try all common extensions)
        for ext in image_exts:
            img_path = os.path.join(images_folder, base + ext)
            if os.path.exists(img_path):
                shutil.copy(img_path, os.path.join(out_images, base + ext))
                break

def oversample_classes(selected_files, image_class_map, class_target, output_images, output_labels, labels_folder, images_folder, image_exts=(".jpg", ".jpeg", ".png")):
    # Count per-class
    class_counts = Counter()
    class_to_images = defaultdict(list)
    for lbl_file in selected_files:
        class_set = image_class_map[lbl_file]
        for c in class_set:
            class_counts[c] += class_set[c]
            class_to_images[c].append(lbl_file)
    # For each class with < target, duplicate images until target reached
    for c in class_counts:
        if class_counts[c] < class_target:
            imgs = class_to_images[c]
            while class_counts[c] < class_target:
                pick = random.choice(imgs)
                # Copy label and image with a new name
                base, ext = os.path.splitext(pick)
                dup_idx = class_counts[c]
                new_lbl = f"{base}_dup{dup_idx}.txt"
                new_lbl_path = os.path.join(output_labels, new_lbl)
                shutil.copy(os.path.join(labels_folder, pick), new_lbl_path)
                # Copy image
                for e in image_exts:
                    img_file = base + e
                    if os.path.exists(os.path.join(images_folder, img_file)):
                        new_img = f"{base}_dup{dup_idx}{e}"
                        new_img_path = os.path.join(output_images, new_img)
                        shutil.copy(os.path.join(images_folder, img_file), new_img_path)
                        break
                # Count boxes in this file for this class
                with open(os.path.join(labels_folder, pick)) as f:
                    for line in f:
                        if line.strip() and int(line.split()[0]) == c:
                            class_counts[c] += 1
    print("Oversampling complete. New class distribution:", class_counts)