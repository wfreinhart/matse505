import os
import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

try:
    import yaml
except ImportError:
    yaml = None

def simple_yaml_parse(filepath):
    """
    Very simple parser for:
    id: Val
    title: Val
    modules:
      - id: Val
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {"modules": []}
    current_key = None
    in_modules = False

    for line in lines:
        line = line.rstrip()
        if not line: continue

        indent = len(line) - len(line.lstrip())
        content = line.strip()

        if indent == 0:
            if ':' in content:
                key, val = content.split(':', 1)
                key = key.strip()
                val = val.strip()
                if key == 'modules':
                    in_modules = True
                    current_key = 'modules'
                else:
                    data[key] = val
                    in_modules = False
        elif in_modules and indent >= 2:
            # List item
            if content.startswith('- '):
                # New item
                item_content = content[2:]
                item = {}
                # Handle single line dict: - id: foo
                if ':' in item_content:
                    parts = item_content.split(':', 1)
                    k = parts[0].strip()
                    v = parts[1].strip()
                    item[k] = v
                elif item_content.strip():
                     # Maybe simple string?
                     pass
                data["modules"].append(item)

    return data

def load_lecture_def(filepath):
    if yaml:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    else:
        print("Warning: PyYAML not installed, using simple parser (limited support).")
        return simple_yaml_parse(filepath)

def build_lecture(def_path, output_dir):
    try:
        lect_def = load_lecture_def(def_path)
        lect_id = lect_def.get("id", Path(def_path).stem)

        full_content = [
            "# ---",
            "# jupyter:",
            "#   jupytext:",
            "#     text_representation:",
            "#       extension: .py",
            "#       format_name: percent",
            "#   kernelspec:",
            "#     display_name: Python 3",
            "#     name: python3",
            "# ---",
            "",
            "# %% [markdown]",
            f"# # {lect_def.get('title', lect_id)}",
            "",
            "# %%",
            "class Context(dict):",
            "    def __init__(self, *args, **kwargs):",
            "        super().__init__(*args, **kwargs)",
            "        self.setdefault('data', None)",
            "        self.setdefault('tensors', {})",
            "        self.setdefault('model', None)",
            "        self.setdefault('viz', None)",
            "",
            "ctx = Context()",
            ""
        ]

        for item in lect_def.get("modules", []):
            if "id" in item:
                mod_id = item["id"]
                mod_path = os.path.join("library/modules", f"{mod_id}.json")
                if not os.path.exists(mod_path):
                    print(f"Warning: Module {mod_id} not found at {mod_path}")
                    continue

                with open(mod_path, 'r') as f:
                    mod = json.load(f)

                # Markdown
                if mod.get("markdown_content"):
                    full_content.append("# %% [markdown]")
                    for line in mod["markdown_content"].split('\n'):
                        full_content.append(f"# {line}")
                    full_content.append("")

                # Code
                if mod.get("code_block"):
                    full_content.append("# %%")
                    lines = mod["code_block"].split('\n')
                    full_content.extend(lines)
                    full_content.append("run_module(ctx)")
                    full_content.append("")

            elif "markdown" in item:
                full_content.append("# %% [markdown]")
                for line in item["markdown"].split('\n'):
                    full_content.append(f"# {line}")
                full_content.append("")

            elif "code" in item:
                full_content.append("# %%")
                full_content.append(item["code"])
                full_content.append("")

        out_path = os.path.join(output_dir, f"{lect_id}.py")
        with open(out_path, 'w') as f:
            f.write("\n".join(full_content))

        print(f"Built {lect_id} -> {out_path}")
        return True

    except Exception as e:
        print(f"Error building {def_path}: {e}")
        return False

def run(lecture=None, build_all=False):
    defs_dir = "lecture_defs"
    out_dir = "lectures"

    if lecture:
        def_path = os.path.join(defs_dir, f"{lecture}.yaml")
        if os.path.exists(def_path):
            build_lecture(def_path, out_dir)
        else:
            print(f"Lecture definition not found: {def_path}")

    elif build_all:
        files = [f for f in os.listdir(defs_dir) if f.endswith('.yaml')]
        for f in files:
            build_lecture(os.path.join(defs_dir, f), out_dir)
    else:
        print("Please specify --lecture or --all")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lecture', help='Build specific lecture ID (e.g. Lecture01)')
    parser.add_argument('--all', action='store_true', help='Build all lectures')
    args = parser.parse_args()

    run(lecture=args.lecture, build_all=args.all)

if __name__ == "__main__":
    main()
