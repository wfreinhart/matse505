import os
import sys
import json
import re
import time
import argparse
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from refactorer import Refactorer

def parse_source(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    cells = []
    current_cell = None

    for line in lines:
        if line.startswith('# %%'):
            if current_cell:
                cells.append(current_cell)

            is_md = '[markdown]' in line
            current_cell = {
                "type": "markdown" if is_md else "code",
                "content": []
            }
        elif current_cell:
            current_cell["content"].append(line)

    if current_cell:
        cells.append(current_cell)

    metadata = {}
    markdown_content = []
    code_content = []

    for i, cell in enumerate(cells):
        content_lines = cell["content"]

        if cell["type"] == "markdown":
            cleaned_lines = []
            in_frontmatter = False
            frontmatter_lines = []
            has_parsed_frontmatter = False

            for line in content_lines:
                stripped = re.sub(r'^# ?', '', line).rstrip()

                if stripped == '---':
                    if in_frontmatter:
                        in_frontmatter = False
                        has_parsed_frontmatter = True
                        for fline in frontmatter_lines:
                            if ':' in fline:
                                k, v = fline.split(':', 1)
                                metadata[k.strip()] = v.strip()
                    elif not has_parsed_frontmatter and not metadata:
                        in_frontmatter = True
                    else:
                        cleaned_lines.append(stripped)
                    continue

                if in_frontmatter:
                    frontmatter_lines.append(stripped)
                else:
                    cleaned_lines.append(stripped)

            md_text = "\n".join(cleaned_lines).strip()
            if md_text:
                markdown_content.append(md_text)

        elif cell["type"] == "code":
            code_content.append("".join(content_lines).strip())

    return {
        "metadata": metadata,
        "markdown": "\n\n".join([m for m in markdown_content if m]),
        "code": "\n\n".join([c for c in code_content if c])
    }

def compile_file(filepath, refactorer):
    try:
        data = parse_source(filepath)
        metadata = data["metadata"]

        if "id" not in metadata:
            metadata["id"] = Path(filepath).stem

        refactored = refactorer.refactor_code(data["code"])

        module = {
            "id": metadata["id"],
            "type": metadata.get("type", "Foundational"),
            "parent_lecture": metadata.get("parent_lecture", ""),
            "requirements": refactored["requirements"],
            "markdown_content": data["markdown"],
            "code_block": refactored["refactored_code"]
        }

        out_path = os.path.join("library/modules", f"{metadata['id']}.json")
        with open(out_path, 'w') as f:
            json.dump(module, f, indent=2)

        print(f"Compiled {Path(filepath).name} -> {Path(out_path).name}")
        return True
    except Exception as e:
        print(f"Error compiling {filepath}: {e}")
        return False

def run(watch=False):
    sources_dir = "library/sources"
    os.makedirs("library/modules", exist_ok=True)

    refactorer = Refactorer()

    files = [os.path.join(sources_dir, f) for f in os.listdir(sources_dir) if f.endswith('.py')]
    for f in files:
        compile_file(f, refactorer)

    if watch:
        print("Watching for changes...")
        mtimes = {f: os.path.getmtime(f) for f in files}
        while True:
            time.sleep(1)
            current_files = [os.path.join(sources_dir, f) for f in os.listdir(sources_dir) if f.endswith('.py')]

            for f in current_files:
                if f not in mtimes:
                    compile_file(f, refactorer)
                    mtimes[f] = os.path.getmtime(f)

            for f in list(mtimes.keys()):
                if os.path.exists(f):
                    mtime = os.path.getmtime(f)
                    if mtime > mtimes[f]:
                        compile_file(f, refactorer)
                        mtimes[f] = mtime
                else:
                    del mtimes[f]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true', help='Watch for changes')
    args = parser.parse_args()
    run(watch=args.watch)

if __name__ == "__main__":
    main()
