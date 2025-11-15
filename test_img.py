import os

def check_yolo_image_label_pairs(image_dir, label_dir):
    missing_labels = []
    empty_labels = []
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img in images:
        base = os.path.splitext(img)[0]
        label_file = os.path.join(label_dir, base + ".txt")
        if not os.path.exists(label_file):
            missing_labels.append(img)
        elif os.path.getsize(label_file) == 0:
            empty_labels.append(img)

    print(f"Total images checked: {len(images)}")
    print(f"Images with missing label files: {len(missing_labels)}")
    if missing_labels:
        print("Missing labels for images:")
        for img in missing_labels:
            print(img)
    print(f"Images with empty label files: {len(empty_labels)}")
    if empty_labels:
        print("Empty labels for images:")
        for img in empty_labels:
            print(img)

# Example usage
if __name__ == "__main__":
    img_dir = "dataset/split/od/train"      # Update to your train folder
    label_dir = "dataset/split/labels/train"  # Update to your labels folder
    check_yolo_image_label_pairs(img_dir, label_dir)
