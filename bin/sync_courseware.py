#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _tracked_files(prefix: Path) -> list[str]:
    return sorted(
        [p.relative_to(prefix).as_posix() for p in prefix.rglob("*") if p.is_file()]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync student courseware into a consumer repo."
    )
    parser.add_argument("--tag", required=True, help="Version tag (e.g. v2026.1.0)")
    parser.add_argument(
        "--track",
        choices=["matse505", "matse419", "matse219"],
        help="Course track. Defaults from repository name.",
    )
    parser.add_argument(
        "--mode",
        choices=["copy", "subtree"],
        default="copy",
        help="copy (artifact copy) or subtree (git subtree add/pull).",
    )
    parser.add_argument(
        "--prefix",
        default="courseware",
        help="Vendored directory path inside this repo.",
    )
    parser.add_argument(
        "--core-release-dir",
        default="../matse-courseware-core/dist/releases",
        help="Path to core release artifacts for copy mode.",
    )
    parser.add_argument(
        "--core-repo",
        default="../matse-courseware-core",
        help="Path/remote of core repository for subtree mode.",
    )
    parser.add_argument(
        "--core-ref",
        default="",
        help="Git ref for subtree mode. Defaults to tag value.",
    )
    parser.add_argument(
        "--sync-manifest",
        default=".courseware-sync.json",
        help="Consumer sync manifest path.",
    )
    return parser.parse_args()


def _detect_track(repo_root: Path) -> str:
    name = repo_root.name.lower()
    for candidate in ("matse505", "matse419", "matse219"):
        if candidate in name:
            return candidate
    raise RuntimeError("Unable to infer track from repo name; pass --track.")


def _write_sync_manifest(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _copy_mode(repo_root: Path, args: argparse.Namespace, track: str) -> dict:
    bundle_root = Path(args.core_release_dir).resolve() / args.tag / track / "courseware"
    if not bundle_root.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle_root}")

    prefix = repo_root / args.prefix
    if prefix.exists():
        shutil.rmtree(prefix)
    shutil.copytree(bundle_root, prefix)

    bundle_manifest = prefix / "COURSEWARE_MANIFEST.json"
    source_manifest = {}
    if bundle_manifest.exists():
        source_manifest = json.loads(bundle_manifest.read_text(encoding="utf-8"))

    files = _tracked_files(prefix)
    return {
        "version": args.tag,
        "released_at": source_manifest.get("released_at", "unknown"),
        "source_commit": source_manifest.get("source_commit", "unknown"),
        "files": files,
        "colab_constraints_hash": source_manifest.get("colab_constraints_hash", "unknown"),
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "sync_mode": "copy",
    }


def _subtree_mode(repo_root: Path, args: argparse.Namespace, track: str) -> dict:
    ref = args.core_ref or args.tag
    prefix = repo_root / args.prefix

    check_cmd = ["git", "rev-parse", "--is-inside-work-tree"]
    subprocess.run(check_cmd, cwd=str(repo_root), check=True, capture_output=True)

    if prefix.exists():
        cmd = [
            "git",
            "subtree",
            "pull",
            f"--prefix={args.prefix}",
            args.core_repo,
            ref,
            "--squash",
        ]
    else:
        cmd = [
            "git",
            "subtree",
            "add",
            f"--prefix={args.prefix}",
            args.core_repo,
            ref,
            "--squash",
        ]
    _run(cmd, repo_root)

    files = _tracked_files(prefix)
    manifest_path = prefix / "COURSEWARE_MANIFEST.json"
    source_manifest = {}
    if manifest_path.exists():
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "version": args.tag,
        "released_at": source_manifest.get("released_at", "unknown"),
        "source_commit": source_manifest.get("source_commit", ref),
        "files": files,
        "colab_constraints_hash": source_manifest.get("colab_constraints_hash", "unknown"),
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "sync_mode": "subtree",
    }


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    track = args.track or _detect_track(repo_root)

    if args.mode == "copy":
        payload = _copy_mode(repo_root, args, track)
    else:
        payload = _subtree_mode(repo_root, args, track)

    _write_sync_manifest(repo_root / args.sync_manifest, payload)
    print(f"Synced {len(payload['files'])} files for {track} @ {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
