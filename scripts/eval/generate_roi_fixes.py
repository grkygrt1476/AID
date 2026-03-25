#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate canonical roi_area01_v1_fix.json files from same-CCTV groups.

This is a one-shot utility.  It does NOT modify the pipeline or any
existing roi_area01_v1.json files.  It only creates or (when justified)
rewrites _fix.json files under configs/rois/<scene>/.

Usage:
    python3 scripts/eval/generate_roi_fixes.py
    python3 scripts/eval/generate_roi_fixes.py --dry-run
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Same-CCTV groups (manually curated, NOT auto-clustered)
# ---------------------------------------------------------------------------

GROUPS: dict[int, list[str]] = {
    1: [
        "E01_001", "E01_002", "E01_003", "E01_009", "E01_022", "E01_024",
        "E01_025", "E01_027", "E01_029", "E01_034", "E01_040", "E01_041",
        "E01_042", "E01_044",
    ],
    2: ["E01_004"],
    3: ["E01_005", "E01_006", "E01_007", "E01_032"],
    4: ["E01_008"],
    5: [
        "E01_010", "E01_011", "E01_012", "E01_013", "E01_015", "E01_017",
        "E01_019", "E01_023", "E01_033", "E01_035", "E01_036", "E01_037",
        "E01_039", "E01_043", "E01_045",
    ],
    6: [
        "E01_014", "E01_016", "E01_018", "E01_020", "E01_021", "E01_038",
    ],
    7: ["E01_026", "E01_028", "E01_030"],
    8: ["E01_031"],
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ROIS_ROOT = REPO_ROOT / "configs" / "rois"
CLIPS_ROOT = REPO_ROOT / "data" / "clips"

FIX_FILENAME = "roi_area01_v1_fix.json"
V1_FILENAME = "roi_area01_v1.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def get_video_resolution(scene_id: str) -> tuple[int | None, int | None]:
    """Get (width, height) from the first clip under data/clips/<scene>/."""
    clip_dir = CLIPS_ROOT / scene_id
    if not clip_dir.is_dir():
        return None, None
    clips = sorted(clip_dir.glob("*.mp4"))
    if not clips:
        return None, None
    try:
        out = subprocess.check_output(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=p=0",
                str(clips[0]),
            ],
            text=True,
            timeout=10,
        ).strip()
        w_str, h_str = out.split(",")
        return int(w_str), int(h_str)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Donor resolution
# ---------------------------------------------------------------------------

def find_donor(
    group_scenes: list[str],
) -> tuple[str | None, Path | None, dict | None]:
    """Find the best donor scene for a group.

    Priority:
      1. _fix.json with labeled_on="orig"
      2. v1.json (any)
    """
    for scene in group_scenes:
        fix_path = ROIS_ROOT / scene / FIX_FILENAME
        if fix_path.is_file():
            data = load_json(fix_path)
            if data.get("labeled_on") == "orig":
                return scene, fix_path, data

    for scene in group_scenes:
        v1_path = ROIS_ROOT / scene / V1_FILENAME
        if v1_path.is_file():
            return scene, v1_path, load_json(v1_path)

    return None, None, None


def resolve_donor_orig_vertices(
    data: dict,
) -> tuple[list[list[int]], int, int, str]:
    """Return (vertices_orig_px, donor_w, donor_h, method_note).

    Conversion logic mirrors aidlib/intrusion/roi.py:
      - labeled_on="orig"  → use vertices directly
      - labeled_on="disp" with disp_scale_used → attempt x/scale, y/scale
        If that produces coordinates beyond image bounds the vertices are
        already in original space despite the metadata.  This matches the
        existing _fix convention established by E01_001 and E01_007.
    """
    vertices = [list(v) for v in data["vertices_px"]]
    img = data.get("image_size", {})
    w: int = img["width"]
    h: int = img["height"]
    labeled_on = data.get("labeled_on", "")

    if labeled_on == "orig":
        return vertices, w, h, "labeled_on=orig, used directly"

    scale = data.get("disp_scale_used")
    if labeled_on == "disp" and scale is not None:
        scale = float(scale)
        if scale == 1.0:
            return vertices, w, h, "disp_scale=1.0, identity (no conversion)"

        # Attempt the roi.py conversion
        converted = [[x / scale, y / scale] for x, y in vertices]
        all_in_bounds = all(
            0 <= cx <= w * 1.01 and 0 <= cy <= h * 1.01
            for cx, cy in converted
        )
        if all_in_bounds:
            rounded = [[round(cx), round(cy)] for cx, cy in converted]
            return rounded, w, h, f"converted from disp (scale={scale})"

        # Vertices are already in original space despite labeled_on=disp.
        # Consistent with E01_001 / E01_007 _fix convention.
        return (
            vertices, w, h,
            f"kept as-is: dividing by disp_scale={scale} produces "
            f"out-of-bounds coords (matches existing _fix convention)",
        )

    # Unknown or missing labeled_on — treat as original
    return vertices, w, h, f"fallback: labeled_on={labeled_on!r}, treated as orig"


def resample_vertices(
    donor_verts: list[list[int]],
    donor_w: int, donor_h: int,
    target_w: int, target_h: int,
) -> list[list[int]]:
    """Resample donor-original vertices to target resolution."""
    if donor_w == target_w and donor_h == target_h:
        return [list(v) for v in donor_verts]
    return [
        [round(x * target_w / donor_w), round(y * target_h / donor_h)]
        for x, y in donor_verts
    ]


# ---------------------------------------------------------------------------
# Fix file creation
# ---------------------------------------------------------------------------

def build_fix_dict(
    scene_id: str,
    vertices: list[list[int]],
    target_w: int,
    target_h: int,
    created_at: str,
) -> dict:
    return {
        "video_id": scene_id,
        "roi_id": "area01",
        "roi_version": 1,
        "roi_type": "AREA",
        "vertices_px": vertices,
        "image_size": {"width": target_w, "height": target_h},
        "labeled_on": "orig",
        "created_at": created_at,
    }


def check_existing_fix(
    fix_path: Path, expected: dict,
) -> tuple[bool, str]:
    """Return (needs_rewrite, reason).  False means keep existing."""
    existing = load_json(fix_path)

    if existing.get("labeled_on") != "orig":
        return True, f"labeled_on={existing.get('labeled_on')!r} != 'orig'"

    ex_img = existing.get("image_size", {})
    exp_img = expected["image_size"]
    if (ex_img.get("width") != exp_img["width"]
            or ex_img.get("height") != exp_img["height"]):
        return True, (
            f"image_size {ex_img.get('width')}x{ex_img.get('height')} "
            f"!= target {exp_img['width']}x{exp_img['height']}"
        )

    if existing.get("vertices_px") != expected["vertices_px"]:
        return True, "vertices differ from canonical recomputed result"

    return False, ""


def write_fix(fix_path: Path, data: dict) -> None:
    with fix_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print actions without writing any files.",
    )
    args = parser.parse_args()

    dry_run: bool = args.dry_run
    created_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    summary_records: list[dict] = []

    print("=" * 72)
    print("ROI Fix Generation" + (" [DRY RUN]" if dry_run else ""))
    print(f"Timestamp: {created_at}")
    print("=" * 72)

    for gid in sorted(GROUPS):
        scenes = GROUPS[gid]
        print(f"\n── Group {gid} ({len(scenes)} scenes) ──")

        donor_scene, donor_path, donor_data = find_donor(scenes)

        # ── No donor in group ──
        if donor_scene is None:
            print("  NO DONOR FOUND → all scenes require manual ROI input")
            for scene in scenes:
                roi_dir = ROIS_ROOT / scene
                if not dry_run:
                    roi_dir.mkdir(parents=True, exist_ok=True)
                summary_records.append({
                    "scene_id": scene,
                    "group_id": gid,
                    "donor_scene_id": None,
                    "donor_file_used": None,
                    "target_resolution": None,
                    "action": "manual_input_required",
                    "reason": "no donor ROI in group",
                    "output_path": None,
                })
            continue

        # ── Resolve donor geometry ──
        donor_verts, donor_w, donor_h, method_note = resolve_donor_orig_vertices(donor_data)
        print(f"  Donor: {donor_scene} ({donor_path.name})")
        print(f"  Donor resolution: {donor_w}x{donor_h}")
        print(f"  Conversion: {method_note}")

        # ── Process each scene ──
        for scene in scenes:
            roi_dir = ROIS_ROOT / scene
            if not dry_run:
                roi_dir.mkdir(parents=True, exist_ok=True)

            target_w, target_h = get_video_resolution(scene)
            if target_w is None:
                print(f"  {scene}: SKIP (cannot determine video resolution)")
                summary_records.append({
                    "scene_id": scene,
                    "group_id": gid,
                    "donor_scene_id": donor_scene,
                    "donor_file_used": str(donor_path),
                    "target_resolution": None,
                    "action": "manual_input_required",
                    "reason": "video resolution unavailable",
                    "output_path": None,
                })
                continue

            target_verts = resample_vertices(
                donor_verts, donor_w, donor_h, target_w, target_h,
            )
            fix_data = build_fix_dict(
                scene, target_verts, target_w, target_h, created_at,
            )
            fix_path = roi_dir / FIX_FILENAME

            if fix_path.is_file():
                needs_rewrite, reason = check_existing_fix(fix_path, fix_data)
                if not needs_rewrite:
                    print(f"  {scene}: KEPT (existing _fix is canonical)")
                    summary_records.append({
                        "scene_id": scene,
                        "group_id": gid,
                        "donor_scene_id": donor_scene,
                        "donor_file_used": str(donor_path),
                        "target_resolution": f"{target_w}x{target_h}",
                        "action": "kept_existing_fix",
                        "reason": "already canonical",
                        "output_path": str(fix_path),
                    })
                    continue

                print(f"  {scene}: REWRITE ({reason})")
                print(f"    path: {fix_path}")
                if not dry_run:
                    write_fix(fix_path, fix_data)
                summary_records.append({
                    "scene_id": scene,
                    "group_id": gid,
                    "donor_scene_id": donor_scene,
                    "donor_file_used": str(donor_path),
                    "target_resolution": f"{target_w}x{target_h}",
                    "action": "rewrote_fix",
                    "reason": reason,
                    "output_path": str(fix_path),
                })
            else:
                print(f"  {scene}: CREATED")
                if not dry_run:
                    write_fix(fix_path, fix_data)
                summary_records.append({
                    "scene_id": scene,
                    "group_id": gid,
                    "donor_scene_id": donor_scene,
                    "donor_file_used": str(donor_path),
                    "target_resolution": f"{target_w}x{target_h}",
                    "action": "created_fix",
                    "reason": "no existing _fix",
                    "output_path": str(fix_path),
                })

    # ── Write summary JSON ──
    summary_path = ROIS_ROOT / "roi_fix_generation_summary.json"
    if not dry_run:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary_records, f, indent=2)
            f.write("\n")

    # ── Final report ──
    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")

    actions: dict[str, int] = {}
    for r in summary_records:
        a = r["action"]
        actions[a] = actions.get(a, 0) + 1
    for action in sorted(actions):
        print(f"  {action}: {actions[action]}")

    manual = [r["scene_id"] for r in summary_records
              if r["action"] == "manual_input_required"]
    if manual:
        print(f"\nScenes requiring manual ROI input ({len(manual)}):")
        for s in manual:
            r = next(x for x in summary_records if x["scene_id"] == s)
            print(f"  {s}  (group {r['group_id']}: {r['reason']})")

    rewrites = [r for r in summary_records if r["action"] == "rewrote_fix"]
    if rewrites:
        print(f"\nRewrites ({len(rewrites)}):")
        for r in rewrites:
            print(f"  {r['scene_id']}: {r['reason']}")

    if not dry_run:
        print(f"\nSummary JSON: {summary_path}")
    else:
        print("\n[DRY RUN — no files were written]")


if __name__ == "__main__":
    main()
