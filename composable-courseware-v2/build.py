#!/usr/bin/env python3
"""Build lectures by expanding module includes.

Reads .py percent-format lecture files, replaces
    # %% include: <module_name>
with the contents of modules/<module_name>.py, and writes
the expanded files to build/.

Usage:
    python build.py                          # build all lectures
    python build.py courses/219/Lecture33.py  # build one lecture
    python build.py --ipynb                  # also convert to .ipynb via jupytext
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
MODULES_DIR = ROOT / "modules"
BUILD_DIR = ROOT / "build"

INCLUDE_RE = re.compile(r"^# %% include:\s*(\S+)\s*$")

# Every module in modules/ opens with an author-facing contract comment
# (# Expects: / # Produces: / optional # Used by ...) terminated by a lone
# "# ---" line. That block is documentation for whoever edits modules/*.py,
# never meant to reach students. It must be stripped here, before inlining --
# leaving it in place merges it into whatever markdown cell precedes the
# include (jupytext only starts a new cell at a real "# %%" marker, so plain
# "#" comment lines just continue the prior cell's rendered text).
MODULE_FRONT_MATTER_RE = re.compile(r"\A(?:#[^\n]*\n)*?# ---\n\n?")


def expand(lecture_path: Path) -> str:
    # A course can keep its own modules/ alongside its lectures/ (e.g. for
    # course-specific datasets not shared with other tracks) -- check there
    # first, then fall back to the shared top-level modules/. Only applies
    # when the lecture actually lives in a lectures/ subdirectory (some
    # courses keep lecture files flat under courses/<track>/ instead).
    course_modules_dir = (
        lecture_path.parent.parent / "modules"
        if lecture_path.parent.name == "lectures"
        else MODULES_DIR
    )

    lines = lecture_path.read_text().splitlines(keepends=True)
    out = []
    for line in lines:
        m = INCLUDE_RE.match(line.rstrip())
        if m:
            mod_name = m.group(1)
            mod_path = course_modules_dir / f"{mod_name}.py"
            if not mod_path.exists():
                mod_path = MODULES_DIR / f"{mod_name}.py"
            if not mod_path.exists():
                print(f"WARNING: module not found: {mod_path}", file=sys.stderr)
                out.append(line)
                continue
            mod_text = mod_path.read_text()
            mod_text, n = MODULE_FRONT_MATTER_RE.subn("", mod_text, count=1)
            if n == 0:
                print(
                    f"WARNING: {mod_path} has no '# ---'-terminated front matter "
                    f"to strip -- check it follows the module authoring convention",
                    file=sys.stderr,
                )
            # strip trailing whitespace but ensure it ends with a newline
            out.append(mod_text.rstrip() + "\n\n")
        else:
            out.append(line)
    return "".join(out)


def build_lecture(lecture_path: Path, ipynb: bool = False) -> Path:
    # preserve courses/<track>/Lecture.py -> build/<track>/Lecture.py
    rel = lecture_path.relative_to(ROOT / "courses")
    out_path = BUILD_DIR / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    expanded = expand(lecture_path)
    out_path.write_text(expanded)
    print(f"  {rel}")

    if ipynb:
        try:
            subprocess.run(
                ["jupytext", "--to", "notebook", str(out_path)],
                check=True, capture_output=True,
            )
            print(f"  {rel.with_suffix('.ipynb')}")
        except FileNotFoundError:
            print("WARNING: jupytext not found, skipping .ipynb conversion", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"WARNING: jupytext failed for {out_path}: {e.stderr.decode()}", file=sys.stderr)

    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", help="Specific lecture files to build (default: all)")
    parser.add_argument("--ipynb", action="store_true", help="Also convert to .ipynb via jupytext")
    args = parser.parse_args()

    if args.files:
        lectures = [Path(f) for f in args.files]
    else:
        lectures = sorted((ROOT / "courses").rglob("*.py"))

    if not lectures:
        print("No lecture files found.")
        return

    print(f"Building {len(lectures)} lecture(s):")
    for lp in lectures:
        build_lecture(lp, ipynb=args.ipynb)
    print("Done.")


if __name__ == "__main__":
    main()
