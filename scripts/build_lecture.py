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

def expand_modules(module_list, module_dir="library/modules", group_dir="library/groups"):
    """
    Recursively expands a list of module items.
    Items can be:
    - dict with "id" (references a module or a group)
    - dict with "markdown" (inline markdown)
    - dict with "code" (inline code)
    """
    expanded = []

    for item in module_list:
        if "id" in item:
            mod_id = item["id"]
            
            # 1. Check if it's an atomic module (JSON)
            json_path = os.path.join(module_dir, f"{mod_id}.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    mod_data = json.load(f)
                expanded.append({"type": "module", "data": mod_data})
                continue

            # 2. Check if it's a group (YAML)
            yaml_path = os.path.join(group_dir, f"{mod_id}.yaml")
            if os.path.exists(yaml_path):
                print(f"  Expanding group: {mod_id}")
                group_def = load_lecture_def(yaml_path)
                # Recursively expand the group's modules
                group_items = group_def.get("modules", [])
                expanded.extend(expand_modules(group_items, module_dir, group_dir))
                continue

            print(f"Warning: Module or Group {mod_id} not found.")

        elif "markdown" in item:
            expanded.append({"type": "markdown", "data": item["markdown"]})
        elif "code" in item:
            expanded.append({"type": "code", "data": item["code"]})
    
    return expanded

def load_enrichments(lecture_id, defs_dir="lecture_defs"):
    """Load the transitions+images sidecar. Returns (transitions_dict, images_dict)."""
    sidecar_path = os.path.join(defs_dir, f"{lecture_id}.transitions.yaml")
    if not os.path.exists(sidecar_path):
        return {}, {}
    try:
        if yaml:
            with open(sidecar_path) as f:
                data = yaml.safe_load(f)
        else:
            return {}, {}
        if not data:
            return {}, {}
        transitions = {
            (t["after_id"], t["before_id"]): t["markdown"]
            for t in data.get("transitions", [])
        }
        images = data.get("images", {})
        return transitions, images
    except Exception as e:
        print(f"Warning: could not load enrichments sidecar: {e}")
        return {}, {}


def build_lecture(def_path, output_dir, enrich=False):
    try:
        lect_def = load_lecture_def(def_path)
        lect_id = lect_def.get("id", Path(def_path).stem)

        if enrich:
            sidecar_path = os.path.join("lecture_defs", f"{lect_id}.transitions.yaml")
            if not os.path.exists(sidecar_path):
                print(f"  No enrichments sidecar found for {lect_id} -- generating now...")
                try:
                    sys.path.append(os.path.dirname(__file__))
                    import generate_transitions
                    generate_transitions.process_lecture(lect_id)
                except Exception as e:
                    print(f"  Warning: enrichment generation failed: {e}")
                    print(f"  Tip: ensure GEMINI_API_KEY is set or run: python scripts/manage_content.py enrich --lecture {lect_id}")

        transitions, images = load_enrichments(lect_id) if enrich else ({}, {})
        if enrich and (transitions or images):
            print(f"  Enriching with {len(transitions)} transition(s) and {len(images)} image set(s).")
        elif enrich:
            print(f"  Warning: no enrichments available for {lect_id}.")

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
            ""
        ]

        # Expand the top-level modules list recursively
        flat_items = expand_modules(lect_def.get("modules", []))

        for idx, item in enumerate(flat_items):
            if item["type"] == "module":
                mod = item["data"]
                mod_id = mod.get("id", "")

                # Markdown
                if mod.get("markdown_content"):
                    full_content.append("# %% [markdown]")
                    for line in mod["markdown_content"].split('\n'):
                        full_content.append(f"# {line}")
                    full_content.append("")

                # Dataset image: injected between this module's markdown and code
                # when this module is identified as the dataset module in the sidecar
                if enrich and images:
                    ds_img = images.get("dataset", {})
                    if ds_img.get("module_id") == mod_id and ds_img.get("path"):
                        img_path = ds_img["path"]
                        alt = ds_img.get("alt", "Dataset illustration")
                        full_content.append("# %% [markdown]")
                        full_content.append(f'# <img src="../{img_path}" alt="{alt}" width=500>')
                        full_content.append("")

                # Code
                code_to_use = mod.get("raw_code", mod.get("code_block"))
                if code_to_use:
                    full_content.append("# %%")
                    lines = code_to_use.split('\n')
                    full_content.extend(lines)
                    full_content.append("")

                # After this module: inject transition + lesson image before next module
                if enrich and transitions:
                    next_items = [x for x in flat_items[idx+1:] if x["type"] == "module"]
                    if next_items:
                        next_id = next_items[0]["data"].get("id", "")
                        transition_text = transitions.get((mod_id, next_id))
                        if transition_text:
                            full_content.append("# %% [markdown]")
                            for line in transition_text.split('\n'):
                                full_content.append(f"# {line}")
                            full_content.append("")

                        # Lesson image: right after the transition, before the task module
                        if enrich and images:
                            lesson_img = images.get("lesson", {})
                            if transition_text and lesson_img.get("path"):
                                img_path = lesson_img["path"]
                                alt = lesson_img.get("alt", "Concept illustration")
                                full_content.append("# %% [markdown]")
                                full_content.append(f'# <img src="../{img_path}" alt="{alt}" width=500>')
                                full_content.append("")

            elif item["type"] == "markdown":
                full_content.append("# %% [markdown]")
                for line in item["data"].split('\n'):
                    full_content.append(f"# {line}")
                full_content.append("")

            elif item["type"] == "code":
                full_content.append("# %%")
                full_content.append(item["data"])
                full_content.append("")

        out_path = os.path.join(output_dir, f"{lect_id}.py")
        with open(out_path, 'w') as f:
            f.write("\n".join(full_content))

        print(f"Built {lect_id} -> {out_path}")
        return True

    except Exception as e:
        print(f"Error building {def_path}: {e}")
        return False

def run(lecture=None, build_all=False, enrich=False):
    defs_dir = "lecture_defs"
    out_dir = "lectures"

    if lecture:
        def_path = os.path.join(defs_dir, f"{lecture}.yaml")
        if os.path.exists(def_path):
            build_lecture(def_path, out_dir, enrich=enrich)
        else:
            print(f"Lecture definition not found: {def_path}")

    elif build_all:
        files = [f for f in os.listdir(defs_dir) if f.endswith('.yaml') and '.transitions' not in f]
        for f in files:
            build_lecture(os.path.join(defs_dir, f), out_dir, enrich=enrich)
    else:
        print("Please specify --lecture or --all")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lecture', help='Build specific lecture ID (e.g. Lecture01)')
    parser.add_argument('--all', action='store_true', help='Build all lectures')
    parser.add_argument('--enrich', action='store_true',
                        help='Inject cached LLM transitions from .transitions.yaml sidecar')
    args = parser.parse_args()

    run(lecture=args.lecture, build_all=args.all, enrich=args.enrich)

if __name__ == "__main__":
    main()
