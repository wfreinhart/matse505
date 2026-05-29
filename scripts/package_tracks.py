#!/usr/bin/env python3
"""
Create separate distributable bundles for CAMEL, telemetry, and composable courseware.

Usage examples:
  python scripts/package_tracks.py --list
  python scripts/package_tracks.py --dry-run camel telemetry composable
  python scripts/package_tracks.py camel telemetry composable
  python scripts/package_tracks.py --zip camel
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dist" / "separated"


@dataclass(frozen=True)
class BundleSpec:
    name: str
    description: str
    includes: tuple[str, ...]
    excludes: tuple[str, ...] = ()


COMMON_EXCLUDES = (
    ".git/**",
    ".cursor/**",
    ".vscode/**",
    ".venv/**",
    "__pycache__/**",
    "*.pyc",
    "telemetry_logs/**",
    "_staging/**",
)


BUNDLE_SPECS: dict[str, BundleSpec] = {
    "camel": BundleSpec(
        name="camel",
        description="CAMEL design/spec + HSDemo courseware artifacts",
        includes=(
            "CAMEL_Courseware_Telemetry_Spec.md",
            "intake/camel.tex",
            "library/sources/HSDemo_*.py",
            "library/modules/HSDemo_*.json",
            "lecture_defs/HSDemo_*.yaml",
            "lecture_defs/HSDemo_*.transitions.yaml",
            "lectures/HSDemo_*.py",
            "notebooks/HSDemo_*.ipynb",
            "lectures/assets/HSDemo_*.png",
            "README.md",
            "LICENSE",
        ),
        excludes=COMMON_EXCLUDES,
    ),
    "telemetry": BundleSpec(
        name="telemetry",
        description="Standalone telemetry collector/server and analysis assets",
        includes=(
            "telemetry/**/*.py",
            "library/sources/HSDemo_telemetry_activate.py",
            "library/modules/HSDemo_telemetry_activate.json",
            "lectures/Telemetry_Analysis.py",
            "notebooks/Telemetry_Analysis.ipynb",
            "CAMEL_Courseware_Telemetry_Spec.md",
            "README.md",
            "requirements.in",
            "requirements.txt",
        ),
        excludes=COMMON_EXCLUDES,
    ),
    "composable": BundleSpec(
        name="composable",
        description="Composable courseware authoring/build pipeline + HSDemo references",
        includes=(
            "scripts/compile_modules.py",
            "scripts/build_lecture.py",
            "scripts/manage_content.py",
            "scripts/demo_composability.py",
            "scripts/generate_transitions.py",
            "library/sources/HSDemo_*.py",
            "library/modules/HSDemo_*.json",
            "lecture_defs/HSDemo_*.yaml",
            "lecture_defs/HSDemo_*.transitions.yaml",
            "lectures/HSDemo_*.py",
            "notebooks/HSDemo_*.ipynb",
            "lectures/assets/HSDemo_*.png",
            "jupytext.toml",
            "README.md",
            "LICENSE",
        ),
        excludes=COMMON_EXCLUDES,
    ),
}


def _normalize(path: Path) -> str:
    return path.as_posix()


def resolve_bundle_files(spec: BundleSpec) -> list[Path]:
    selected: set[Path] = set()
    for pattern in spec.includes:
        for candidate in REPO_ROOT.glob(pattern):
            if candidate.is_file():
                selected.add(candidate.relative_to(REPO_ROOT))

    filtered = [
        rel
        for rel in selected
        if not any(fnmatch.fnmatch(_normalize(rel), pat) for pat in spec.excludes)
    ]
    return sorted(filtered, key=_normalize)


def _write_zip(bundle: BundleSpec, files: Iterable[Path], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_zip = output_dir / f"{bundle.name}.zip"
    file_list = list(files)
    manifest = {
        "bundle": bundle.name,
        "description": bundle.description,
        "source_repo": str(REPO_ROOT),
        "file_count": len(file_list),
        "files": [_normalize(f) for f in file_list],
    }

    with ZipFile(output_zip, "w", compression=ZIP_DEFLATED) as zip_handle:
        for rel_path in file_list:
            src = REPO_ROOT / rel_path
            arcname = Path(bundle.name) / rel_path
            zip_handle.write(src, arcname=_normalize(arcname))
        zip_handle.writestr(
            f"{bundle.name}/PACKAGING_MANIFEST.json", json.dumps(manifest, indent=2)
        )

    return output_zip


def _write_directory(bundle: BundleSpec, files: Iterable[Path], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_root = output_dir / bundle.name
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)

    file_list = list(files)
    manifest = {
        "bundle": bundle.name,
        "description": bundle.description,
        "source_repo": str(REPO_ROOT),
        "file_count": len(file_list),
        "files": [_normalize(f) for f in file_list],
    }

    for rel_path in file_list:
        src = REPO_ROOT / rel_path
        dst = bundle_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    (bundle_root / "PACKAGING_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return bundle_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage CAMEL/telemetry/composable assets into separate directories "
            "(with optional zip output)."
        )
    )
    parser.add_argument(
        "bundles",
        nargs="*",
        choices=sorted(BUNDLE_SPECS.keys()),
        help="Bundle(s) to create. Defaults to all.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Where bundle outputs are written (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be packaged without writing outputs.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Write each bundle as a .zip instead of a directory tree.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available bundles and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        print("Available bundles:")
        for key in sorted(BUNDLE_SPECS):
            spec = BUNDLE_SPECS[key]
            print(f"  - {spec.name}: {spec.description}")
        return 0

    bundle_names = args.bundles or sorted(BUNDLE_SPECS.keys())
    output_dir = Path(args.output_dir).resolve()

    for bundle_name in bundle_names:
        spec = BUNDLE_SPECS[bundle_name]
        files = resolve_bundle_files(spec)
        print(f"\n[{bundle_name}] {spec.description}")
        print(f"Files selected: {len(files)}")
        for rel in files:
            print(f"  - {rel.as_posix()}")

        if not files:
            print("  ! No files matched this bundle.")
            continue

        if args.dry_run:
            continue

        if args.zip:
            zip_path = _write_zip(spec, files, output_dir)
            print(f"Created zip: {zip_path}")
        else:
            dir_path = _write_directory(spec, files, output_dir)
            print(f"Created directory: {dir_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
