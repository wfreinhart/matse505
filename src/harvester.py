import re
import json
import os
from pathlib import Path
from typing import List, Dict

class Harvester:
    def __init__(self):
        # Regex for Jupytext percent format cells
        self.cell_marker = re.compile(r'^# %%(.*)$')
        self.markdown_cell = re.compile(r'^# %% \[markdown\]')
        
    def parse_file(self, file_path: str) -> List[Dict]:
        """Parses a Jupytext .py file into a list of cell dictionaries."""
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        cells = []
        current_cell = None
        
        for line in lines:
            if self.cell_marker.match(line):
                if current_cell:
                    cells.append(current_cell)
                
                is_md = "[markdown]" in line
                current_cell = {
                    "type": "markdown" if is_md else "code",
                    "content": [],
                    "metadata": line.strip()
                }
            elif current_cell:
                current_cell["content"].append(line)
        
        if current_cell:
            cells.append(current_cell)
            
        # Clean up content
        for cell in cells:
            content = "".join(cell["content"])
            if cell["type"] == "markdown":
                # Remove leading '# ' or '#' from each line in markdown cells
                cleaned = []
                for l in cell["content"]:
                    cleaned.append(re.sub(r'^# ?', '', l))
                cell["content"] = "".join(cleaned).strip()
            else:
                cell["content"] = content.strip()
                
        return cells

    def segment_cells(self, cells: List[Dict]) -> List[Dict]:
        """Groups cells into pedagogical units based on markdown headers."""
        # First, extract all global imports to ensure segments remain isolated but functional
        global_imports = []
        for cell in cells:
            if cell["type"] == "code":
                for line in cell["content"].split('\n'):
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        global_imports.append(line.strip())
        
        segments = []
        current_segment = {
            "id": "",
            "markdown_content": "",
            "code_block": "",
            "imports": list(set(global_imports)),
            "cells": []
        }
        
        for cell in cells:
            # Start a new segment on h1 or h2 headers in markdown
            if cell["type"] == "markdown":
                lines = cell["content"].split('\n')
                first_line = lines[0].strip() if lines else ""
                
                if first_line.startswith('# ') or first_line.startswith('## '):
                    if current_segment["cells"]:
                        segments.append(current_segment)
                    
                    header_text = first_line.lstrip('#').strip()
                    # Sanitize header for ID
                    safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', header_text.lower().replace(' ', '_'))
                    safe_id = re.sub(r'_+', '_', safe_id).strip('_')
                    
                    current_segment = {
                        "id": safe_id or "segment",
                        "markdown_content": cell["content"],
                        "code_block": "",
                        "imports": list(set(global_imports)),
                        "cells": [cell]
                    }
                    continue
            
            if current_segment:
                current_segment["cells"].append(cell)
                if cell["type"] == "markdown":
                    current_segment["markdown_content"] += "\n\n" + cell["content"]
                else:
                    # Skip duplication of imports within the block if they are already in global_imports
                    # but keep them for now to avoid breaking logic that might rely on specific order
                    current_segment["code_block"] += "\n\n" + cell["content"]
                    
        if current_segment["cells"]:
            segments.append(current_segment)
            
        return segments

def run_harvest(file_path: str, output_dir: str):
    h = Harvester()
    cells = h.parse_file(file_path)
    segments = h.segment_cells(cells)
    
    stem = Path(file_path).stem
    for i, seg in enumerate(segments):
        seg_id = f"{stem}_{seg['id'] or i}"
        seg["id"] = seg_id 
        seg["sequence"] = i # Preserve original order for integration testing
        seg["parent_lecture"] = stem
        with open(os.path.join(output_dir, f"{seg_id}.json"), 'w') as f:
            json.dump(seg, f, indent=2)
    return len(segments)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        count = run_harvest(sys.argv[1], "_staging/raw")
        print(f"Extracted {count} segments to _staging/raw")
