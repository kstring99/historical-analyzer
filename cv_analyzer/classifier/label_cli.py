"""Interactive labeling CLI for building the training dataset.

Shows detected crops one at a time and asks the user to classify them.
Builds the labels.json file incrementally.
"""

import json
import shutil
from pathlib import Path

import cv2

from cv_analyzer.classifier.model import TARGET_CLASSES


def label_crops(input_dir: str, output_dir: str):
    """Interactive labeling session.

    Reads crop images from input_dir, shows each one, asks user to label.
    Saves labeled crops + labels.json to output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    crops_dir = output_path / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    labels_path = output_path / "labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            data = json.load(f)
    else:
        data = {"images": []}

    existing_files = {e["file"] for e in data["images"]}
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    crops = sorted(
        p for p in input_path.iterdir()
        if p.suffix.lower() in image_exts and p.name not in existing_files
    )

    if not crops:
        print("No unlabeled crops found.")
        return

    print(f"\nLabeling {len(crops)} crops.")
    print(f"Classes: {', '.join(f'{i}={c}' for i, c in enumerate(TARGET_CLASSES))}")
    print("Enter class number, 's' to skip, 'q' to quit.\n")

    labeled = 0

    for crop_path in crops:
        img = cv2.imread(str(crop_path))
        if img is None:
            continue

        # Show the crop in a window
        display = img.copy()
        h, w = display.shape[:2]
        if max(h, w) < 200:
            scale = 200 / max(h, w)
            display = cv2.resize(display, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Crop - press key in terminal", display)
        cv2.waitKey(1)

        print(f"[{crop_path.name}] ({img.shape[1]}x{img.shape[0]})")
        choice = input("  Class (0-6/s/q): ").strip().lower()

        if choice == "q":
            break
        if choice == "s":
            continue

        try:
            idx = int(choice)
            if idx < 0 or idx >= len(TARGET_CLASSES):
                print("  Invalid class index, skipping.")
                continue
        except ValueError:
            print("  Invalid input, skipping.")
            continue

        cls = TARGET_CLASSES[idx]
        dest = crops_dir / crop_path.name
        shutil.copy2(crop_path, dest)
        data["images"].append({"file": crop_path.name, "class": cls})
        labeled += 1
        print(f"  → {cls}")

    cv2.destroyAllWindows()

    with open(labels_path, "w") as f:
        json.dump(data, f, indent=2)

    total = len(data["images"])
    print(f"\nLabeled {labeled} new crops. Total dataset: {total} images.")
    print(f"Saved to: {labels_path}")

    # Print class distribution
    from collections import Counter
    dist = Counter(e["class"] for e in data["images"])
    print("\nClass distribution:")
    for cls in TARGET_CLASSES:
        print(f"  {cls:<25} {dist.get(cls, 0):>4}")
