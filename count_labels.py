import os
from collections import Counter

def count_labels(label_dir):
    count = 0
    label_counter = Counter()
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            count += 1
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        label_counter[parts[0]] += 1  # parts[0]: class label as string
    for label, count in label_counter.items():
        print(f"Label {label}: {count}")
    print(f"Total label files: {count}")

if __name__ == "__main__":
    label_dir = "data/unprocessed labels/Labelled Watermelons"
    count_labels(label_dir)
