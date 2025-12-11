import os
import shutil
import random

# Input directory (your original dataset)
INPUT_DIR = "fer2013/train"

# Output directory (new split folders)
OUTPUT_DIR = "fer2013_split"

# Create output structure
for split in ["train", "test", "val"]:
    for cls in os.listdir(INPUT_DIR):
        os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

TRAIN_SPLIT = 0.70
TEST_SPLIT = 0.20
VAL_SPLIT = 0.10

for cls in os.listdir(INPUT_DIR):
    src = os.path.join(INPUT_DIR, cls)
    images = os.listdir(src)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_SPLIT)
    test_end = train_end + int(total * TEST_SPLIT)

    train_imgs = images[:train_end]
    test_imgs = images[train_end:test_end]
    val_imgs = images[test_end:]

    print(f"Processing {cls} | Total: {total}")

    for img in train_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(OUTPUT_DIR, "train", cls))

    for img in test_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(OUTPUT_DIR, "test", cls))

    for img in val_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(OUTPUT_DIR, "val", cls))

print("Dataset split complete!")
