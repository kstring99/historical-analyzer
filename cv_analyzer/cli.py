"""CLI entry point for the CV-based historical ESA analyzer."""

import argparse
import sys
import time
from pathlib import Path

from cv_analyzer.aerial.preprocessor import load_images_from_dir, normalize
from cv_analyzer.aerial.change_detect import detect_changes
from cv_analyzer.aerial.object_detect import detect_objects
from cv_analyzer.report import generate_report


def cmd_analyze(args):
    """Analyze aerial photos: change detection + object detection."""
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading images from: {input_dir}")
    entries = load_images_from_dir(input_dir)

    if not entries:
        print("No images with parseable years found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(entries)} images: {', '.join(str(e['year']) for e in entries)}")

    # Normalize all images
    target_size = args.resolution
    for entry in entries:
        entry["image"] = normalize(entry["image"], target_long_edge=target_size)

    # --- Object detection on each image ---
    print("\n--- Object Detection ---")
    image_results = []

    for entry in entries:
        t0 = time.time()
        detections = detect_objects(entry["image"])
        elapsed = time.time() - t0

        image_results.append({
            "filename": entry["filename"],
            "year": entry["year"],
            "path": entry["path"],
            "detections": detections,
        })

        det_summary = ", ".join(
            f"{d['class']}({d['confidence']})" for d in detections[:5]
        )
        print(
            f"  {entry['year']} ({entry['filename']}): "
            f"{len(detections)} detections in {elapsed:.1f}s"
            f"{' — ' + det_summary if detections else ''}"
        )

    # --- Pairwise change detection ---
    print("\n--- Change Detection ---")
    change_results = []

    if len(entries) >= 2:
        for i in range(len(entries) - 1):
            a = entries[i]
            b = entries[i + 1]
            t0 = time.time()

            result = detect_changes(
                a["image"], b["image"],
                a["year"], b["year"],
            )
            elapsed = time.time() - t0

            change_results.append(result)

            n_changes = len(result["changes"])
            print(
                f"  {a['year']} → {b['year']}: "
                f"SSIM={result['ssim_score']}, "
                f"align={result['alignment_quality']}, "
                f"{n_changes} change regions, "
                f"{result['total_changed_pct']}% area "
                f"({elapsed:.1f}s)"
            )
    else:
        print("  (Need 2+ images for change detection)")

    # --- Optional classifier refinement ---
    if args.model:
        _run_classifier(args.model, image_results)

    # --- Optional VLM fallback ---
    if args.vlm:
        _run_vlm_fallback(image_results, args.vlm_model, args.vlm_threshold)

    # --- Generate report ---
    print(f"\n--- Generating Report ---")
    report = generate_report(
        str(input_dir), str(output_dir), image_results, change_results
    )

    n_recs = len(report["rec_candidates"])
    print(f"  Report written to: {output_dir}/")
    print(f"  REC candidates: {n_recs}")

    if n_recs > 0:
        print("\n  Top REC candidates:")
        for r in report["rec_candidates"][:5]:
            print(
                f"    [{r['confidence']}] {r['type'].replace('_', ' ').title()} "
                f"— {r['description']} (year: {r['year']})"
            )


def _run_classifier(model_path: str, image_results: list):
    """Refine detections using a trained classifier."""
    try:
        import torch
        from cv_analyzer.classifier.model import load_model, predict, get_device
        from cv_analyzer.classifier.dataset import prepare_tensor
    except ImportError:
        print("  Warning: PyTorch not available, skipping classifier.", file=sys.stderr)
        return

    print("\n--- Classifier Refinement ---")
    device = get_device()
    model = load_model(model_path, device)
    print(f"  Loaded model from {model_path} (device: {device})")

    for img_res in image_results:
        for det in img_res.get("detections", []):
            if det.get("crop") is None:
                continue
            tensor = prepare_tensor(det["crop"])
            result = predict(model, tensor, device)
            det["classifier_class"] = result["class"]
            det["classifier_confidence"] = result["confidence"]
            # Override CV class if classifier is confident
            if result["confidence"] > det["confidence"]:
                det["class"] = result["class"]
                det["confidence"] = result["confidence"]


def _run_vlm_fallback(image_results: list, model: str, threshold: float):
    """Run VLM on low-confidence detections."""
    try:
        from cv_analyzer.vlm_fallback import check_ollama, classify_with_vlm
    except ImportError:
        print("  Warning: VLM fallback not available.", file=sys.stderr)
        return

    if not check_ollama():
        print("  Warning: ollama not running, skipping VLM fallback.", file=sys.stderr)
        return

    print(f"\n--- VLM Fallback (threshold < {threshold}) ---")
    count = 0

    for img_res in image_results:
        for det in img_res.get("detections", []):
            if det["confidence"] >= threshold:
                continue
            if det.get("crop") is None:
                continue

            result = classify_with_vlm(
                det["crop"], det["class"], det["confidence"], model=model
            )
            if result:
                det["vlm_class"] = result.get("class", det["class"])
                det["vlm_confidence"] = result.get("confidence", 0)
                det["vlm_reasoning"] = result.get("reasoning", "")
                count += 1

    print(f"  Processed {count} low-confidence detections via VLM.")


def cmd_label(args):
    """Interactive labeling CLI."""
    from cv_analyzer.classifier.label_cli import label_crops
    label_crops(args.input, args.output)


def cmd_train(args):
    """Train the classifier."""
    from cv_analyzer.classifier.train import train
    train(
        dataset_dir=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="cv_analyzer",
        description="CV-based historical ESA analyzer — runs entirely local",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- analyze ---
    p_analyze = subparsers.add_parser(
        "analyze",
        help="Analyze aerial photos: change detection + object detection",
    )
    p_analyze.add_argument("input", help="Directory of aerial photos (year in filename)")
    p_analyze.add_argument("-o", "--output", default="./cv_report", help="Output directory")
    p_analyze.add_argument("--resolution", type=int, default=1024, help="Target long-edge resolution")
    p_analyze.add_argument("--model", help="Path to trained classifier checkpoint (.pt)")
    p_analyze.add_argument("--vlm", action="store_true", help="Enable VLM fallback for low-confidence detections")
    p_analyze.add_argument("--vlm-model", default="llava:13b", help="Ollama model for VLM fallback")
    p_analyze.add_argument("--vlm-threshold", type=float, default=0.7, help="Confidence threshold for VLM fallback")

    # --- label ---
    p_label = subparsers.add_parser(
        "label",
        help="Interactively label detection crops for training",
    )
    p_label.add_argument("input", help="Directory of crop images to label")
    p_label.add_argument("-o", "--output", default="./training_data", help="Output dataset directory")

    # --- train ---
    p_train = subparsers.add_parser(
        "train",
        help="Train the structure classifier",
    )
    p_train.add_argument("dataset", help="Labeled dataset directory (with labels.json)")
    p_train.add_argument("-o", "--output", default="./models", help="Output directory for model checkpoints")
    p_train.add_argument("--epochs", type=int, default=30, help="Training epochs")
    p_train.add_argument("--batch-size", type=int, default=16, help="Batch size")
    p_train.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "analyze": cmd_analyze,
        "label": cmd_label,
        "train": cmd_train,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
