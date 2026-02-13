import os
import sys
import re
from pathlib import Path

# Add src to path to import harvester
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from harvester import Harvester

def simple_yaml_dump(data, stream=None):
    """Simple YAML dumper for our specific structure."""
    lines = []
    for key, value in data.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    # For list of dicts, we format like:
                    # - key1: val1
                    #   key2: val2
                    first = True
                    for k, v in item.items():
                        if first:
                            lines.append(f"  - {k}: {v}")
                            first = False
                        else:
                            lines.append(f"    {k}: {v}")
                else:
                    lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")

    content = "\n".join(lines)
    if stream:
        stream.write(content)
    return content

def migrate():
    lectures_dir = "lectures"
    sources_dir = "library/sources"
    lecture_defs_dir = "lecture_defs"

    os.makedirs(sources_dir, exist_ok=True)
    os.makedirs(lecture_defs_dir, exist_ok=True)

    h = Harvester()
    lecture_files = sorted([f for f in os.listdir(lectures_dir) if f.startswith('Lecture') and f.endswith('.py')])

    for lect_file in lecture_files:
        lect_path = os.path.join(lectures_dir, lect_file)
        lect_id = Path(lect_file).stem

        print(f"Processing {lect_id}...")

        try:
            cells = h.parse_file(lect_path)
            segments = h.segment_cells(cells)
        except Exception as e:
            print(f"Skipping {lect_file} due to parse error: {e}")
            continue

        module_list = []

        for i, seg in enumerate(segments):
            # Replicate Harvester.run_harvest ID logic
            seg_stem = seg['id'] or str(i)
            full_id = f"{lect_id}_{seg_stem}"

            # Create Source File content
            source_content = []

            # Jupytext Header
            source_content.append("# ---")
            source_content.append("# jupyter:")
            source_content.append("#   jupytext:")
            source_content.append("#     text_representation:")
            source_content.append("#       extension: .py")
            source_content.append("#       format_name: percent")
            source_content.append("#       format_version: '1.3'")
            source_content.append("#       jupytext_version: 1.16.1")
            source_content.append("#   kernelspec:")
            source_content.append("#     display_name: Python 3")
            source_content.append("#     name: python3")
            source_content.append("# ---")
            source_content.append("")

            # Metadata Cell
            metadata = {
                "id": full_id,
                "type": "Foundational",
                "parent_lecture": lect_id,
            }

            source_content.append("# %% [markdown]")
            source_content.append("# ---")
            for k, v in metadata.items():
                source_content.append(f"# {k}: {v}")
            source_content.append("# ---")
            source_content.append("#")

            # Markdown Content
            if seg['markdown_content'].strip():
                # Check if it's already stripped of # by Harvester (yes)
                # We need to add # prefix
                md_lines = seg['markdown_content'].split('\n')
                for line in md_lines:
                    source_content.append(f"# {line}")
            else:
                source_content.append("# (No markdown content)")

            source_content.append("")

            # Code Content
            if seg['code_block'].strip():
                source_content.append("# %%")
                source_content.append(seg['code_block'].strip())
                source_content.append("")

            # Write to file
            out_path = os.path.join(sources_dir, f"{full_id}.py")
            with open(out_path, 'w') as f:
                f.write("\n".join(source_content))

            module_list.append({"id": full_id})

        # Create Lecture Definition
        lect_def = {
            "id": lect_id,
            "title": lect_id,
            "modules": module_list
        }

        def_path = os.path.join(lecture_defs_dir, f"{lect_id}.yaml")
        with open(def_path, 'w') as f:
            simple_yaml_dump(lect_def, f)

    print("Migration complete.")

if __name__ == "__main__":
    migrate()
