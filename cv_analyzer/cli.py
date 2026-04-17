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
    """Analyze aerial photos: change detection + object detection + optional tables."""
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

    # --- Detect property boundary (green rectangle) on each image ---
    from cv_analyzer.aerial.boundary import detect_green_boundary, classify_zone

    boundaries = {}
    for entry in entries:
        boundary = detect_green_boundary(entry["image"])
        boundaries[entry["year"]] = boundary
        if boundary is not None:
            print(f"  {entry['year']}: Green boundary detected")

    # --- Object detection on each image ---
    print("\n--- Object Detection ---")
    image_results = []

    for entry in entries:
        t0 = time.time()
        detections = detect_objects(entry["image"])
        elapsed = time.time() - t0

        # Classify each detection into a zone
        boundary = boundaries.get(entry["year"])
        for det in detections:
            det["zone"] = classify_zone(det["bbox"], boundary, entry["image"].shape)

        image_results.append({
            "filename": entry["filename"],
            "year": entry["year"],
            "path": entry["path"],
            "detections": detections,
        })

        det_summary = ", ".join(
            f"{d['class']}({d['confidence']},{d['zone']})" for d in detections[:5]
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

            # Classify change regions into zones using the later image's boundary
            boundary = boundaries.get(b["year"])
            for ch in result.get("changes", []):
                ch["zone"] = classify_zone(ch["bbox"], boundary, b["image"].shape)

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

    # --- Tables (ESA-formatted output) ---
    if args.tables:
        _generate_tables(image_results, change_results, output_dir, args)


def _generate_tables(image_results, change_results, output_dir, args):
    """Generate ESA-formatted tables using local LLM."""
    from cv_analyzer.ollama_client import check_available, list_models
    from cv_analyzer.table_builder import (
        build_tables, format_tables_text, format_tables_html,
    )

    print("\n--- Generating ESA Tables ---")

    if not check_available():
        print("  Warning: ollama not running. Using fallback (no LLM).", file=sys.stderr)
        print("  Start ollama for better table output: ollama serve", file=sys.stderr)
        llm_model = None
    else:
        available = list_models()
        llm_model = args.llm_model
        if llm_model not in available:
            # Try to find a reasonable default
            for candidate in ("llama3.2:3b", "qwen3:4b", "qwen3:8b", "llama3.2:1b"):
                if candidate in available:
                    llm_model = candidate
                    break
            else:
                print(f"  Warning: model '{llm_model}' not found. Available: {available}", file=sys.stderr)
                print("  Using fallback (no LLM).", file=sys.stderr)
                llm_model = None

    if llm_model:
        print(f"  Using model: {llm_model}")

    tables = build_tables(
        image_results, change_results,
        model=llm_model or "llama3.2:3b",
        verbose=args.verbose,
    )

    output_path = Path(output_dir)

    # Plain text tables
    text = format_tables_text(tables)
    text_path = output_path / "tables.txt"
    with open(text_path, "w") as f:
        f.write(text)
    print(f"\n{text}")

    # HTML tables (paste into Word/reports)
    html = format_tables_html(tables)
    html_path = output_path / "tables.html"
    full_html = f"""<!DOCTYPE html>
<html><head><title>ESA Historical Documentation Tables</title>
<style>
body{{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 20px}}
table{{width:100%;border-collapse:collapse;margin-bottom:20px}}
th,td{{border:1px solid #999;padding:8px;text-align:left}}
th{{background:#e8e8e8;font-size:0.9em}}
</style></head>
<body>
<h1>Historical Documentation Review</h1>
{html}
</body></html>"""
    with open(html_path, "w") as f:
        f.write(full_html)

    print(f"  Text tables: {text_path}")
    print(f"  HTML tables: {html_path}")


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
            if result["confidence"] > det["confidence"]:
                det["class"] = result["class"]
                det["confidence"] = result["confidence"]


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
    p_analyze.add_argument("--tables", action="store_true", help="Generate ESA-formatted tables (uses local LLM)")
    p_analyze.add_argument("--llm-model", default="llama3.2:3b", help="Ollama model for table generation")
    p_analyze.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

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
