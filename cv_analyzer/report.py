"""Generate JSON and Markdown reports from analysis results."""

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2


def generate_report(
    input_dir: str,
    output_dir: str,
    image_results: list[dict],
    change_results: list[dict],
) -> dict:
    """Generate a full analysis report.

    Writes: report.json, report.md, and detection crop images to output_dir/crops/.
    Returns the report dict.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    crops_dir = output_path / "crops"
    crops_dir.mkdir(exist_ok=True)

    report = {
        "site": str(input_dir),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "summary": _build_summary(image_results, change_results),
        "images": [],
        "changes": [],
        "rec_candidates": [],
    }

    crop_idx = 0

    # Process per-image detections
    for img_res in image_results:
        img_entry = {
            "filename": img_res["filename"],
            "year": img_res["year"],
            "detections": [],
        }

        for det in img_res.get("detections", []):
            crop_filename = f"{img_res['year']}_det_{crop_idx:03d}.jpg"
            crop_path = crops_dir / crop_filename
            if det.get("crop") is not None:
                cv2.imwrite(str(crop_path), det["crop"])

            entry = {
                "class": det["class"],
                "confidence": det["confidence"],
                "description": det["description"],
                "bbox": det["bbox"],
                "area_pct": det["area_pct"],
                "crop_file": crop_filename,
            }
            if "metrics" in det:
                entry["metrics"] = det["metrics"]

            img_entry["detections"].append(entry)

            # High-confidence detections are REC candidates
            if det["confidence"] >= 0.5 and det["class"] != "benign":
                report["rec_candidates"].append({
                    "type": det["class"],
                    "year": img_res["year"],
                    "confidence": det["confidence"],
                    "description": det["description"],
                    "source": img_res["filename"],
                    "crop_file": crop_filename,
                })

            crop_idx += 1

        report["images"].append(img_entry)

    # Process change detections
    for change_res in change_results:
        change_entry = {
            "from_year": change_res["from_year"],
            "to_year": change_res["to_year"],
            "ssim_score": change_res["ssim_score"],
            "alignment_quality": change_res["alignment_quality"],
            "total_changed_pct": change_res["total_changed_pct"],
            "changes": [],
        }

        for ch in change_res.get("changes", []):
            crop_filename = f"change_{change_res['from_year']}_{change_res['to_year']}_{crop_idx:03d}.jpg"
            crop_path = crops_dir / crop_filename
            if ch.get("crop") is not None:
                cv2.imwrite(str(crop_path), ch["crop"])

            entry = {
                "type": ch["type"],
                "bbox": ch["bbox"],
                "area_pct": ch["area_pct"],
                "confidence": ch["confidence"],
                "crop_file": crop_filename,
            }
            change_entry["changes"].append(entry)

            if ch["confidence"] >= 0.5 and ch["type"] != "modified":
                report["rec_candidates"].append({
                    "type": ch["type"],
                    "year": change_res["to_year"],
                    "confidence": ch["confidence"],
                    "description": f"{ch['type'].replace('_', ' ').title()} between {change_res['from_year']}-{change_res['to_year']}",
                    "source": "change_detection",
                    "crop_file": crop_filename,
                })

            crop_idx += 1

        report["changes"].append(change_entry)

    # Sort REC candidates by confidence
    report["rec_candidates"].sort(key=lambda r: r["confidence"], reverse=True)

    # Write JSON
    json_path = output_path / "report.json"
    _write_json(report, json_path)

    # Write Markdown
    md_path = output_path / "report.md"
    _write_markdown(report, md_path)

    return report


def _build_summary(image_results: list, change_results: list) -> dict:
    years = sorted(set(r["year"] for r in image_results))
    total_detections = sum(len(r.get("detections", [])) for r in image_results)
    total_changes = sum(len(r.get("changes", [])) for r in change_results)

    return {
        "years_analyzed": years,
        "num_images": len(image_results),
        "total_detections": total_detections,
        "total_change_regions": total_changes,
    }


def _write_json(report: dict, path: Path):
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)


def _write_markdown(report: dict, path: Path):
    lines = []
    lines.append("# Aerial Photo Analysis Report")
    lines.append(f"\n**Site:** {report['site']}")
    lines.append(f"**Analyzed:** {report['analyzed_at']}")

    s = report["summary"]
    lines.append(f"\n**Years:** {', '.join(str(y) for y in s['years_analyzed'])}")
    lines.append(f"**Images analyzed:** {s['num_images']}")
    lines.append(f"**Object detections:** {s['total_detections']}")
    lines.append(f"**Change regions:** {s['total_change_regions']}")

    # REC candidates
    recs = report["rec_candidates"]
    if recs:
        lines.append(f"\n## REC Candidates ({len(recs)})\n")
        for r in recs:
            lines.append(
                f"- **{r['type'].replace('_', ' ').title()}** "
                f"(year: {r['year']}, confidence: {r['confidence']}) — "
                f"{r['description']}  "
            )
            lines.append(f"  Crop: `{r['crop_file']}`")
    else:
        lines.append("\n## REC Candidates\n")
        lines.append("No REC candidates identified.")

    # Change log
    if report["changes"]:
        lines.append("\n## Change Detection Log\n")
        for ch in report["changes"]:
            lines.append(
                f"### {ch['from_year']} → {ch['to_year']} "
                f"(SSIM: {ch['ssim_score']}, alignment: {ch['alignment_quality']})"
            )
            lines.append(f"Total area changed: {ch['total_changed_pct']}%\n")
            if ch["changes"]:
                for c in ch["changes"]:
                    lines.append(
                        f"- {c['type'].replace('_', ' ').title()} "
                        f"at ({c['bbox']['x']}, {c['bbox']['y']}) "
                        f"— {c['area_pct']}% of image, confidence {c['confidence']}"
                    )
            else:
                lines.append("No significant structural changes detected.")
            lines.append("")

    # Per-image detections
    lines.append("\n## Per-Image Detections\n")
    for img in report["images"]:
        lines.append(f"### {img['filename']} ({img['year']})")
        if img["detections"]:
            for d in img["detections"]:
                lines.append(
                    f"- **{d['class'].replace('_', ' ').title()}** "
                    f"(confidence: {d['confidence']}) — {d['description']}  "
                )
                lines.append(f"  Crop: `{d['crop_file']}`")
        else:
            lines.append("No objects of interest detected.")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
