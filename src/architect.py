import json
import os
from typing import List, Dict

class Architect:
    def __init__(self, manifest_path: str):
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
    def resolve_dependencies(self, module_ids: List[str]) -> List[str]:
        """Simple ordered list based on requirements (Topological sort placeholder)."""
        # For now, just return in provided order but check manifest exists
        valid_ids = []
        for mid in module_ids:
            if any(m["id"] == mid for m in self.manifest["modules"]):
                valid_ids.append(mid)
        return valid_ids

    def assemble_lecture(self, module_ids: List[str], output_path: str):
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
            "# # Generated Composable Lecture",
            ""
        ]
        
        # Inject Context initialization
        full_content.extend([
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
        ])

        for mid in module_ids:
            # Find module file
            mod_meta = next((m for m in self.manifest["modules"] if m["id"] == mid), None)
            if not mod_meta: continue
            
            with open(mod_meta["path"], 'r') as f:
                mod = json.load(f)
            
            # Add Bridge Text (Placeholder)
            full_content.extend([
                "# %% [markdown]",
                f"# ## Section: {mod['id'].replace('_', ' ').capitalize()}",
                "#"
            ])
            for line in mod["markdown_content"].split('\n'):
                full_content.append(f"# {line}")
            full_content.append("")
            
            # Add Code block
            if mod["code_block"]:
                lines = mod["code_block"].split('\n')
                full_content.append("# %%")
                full_content.extend(lines)
                full_content.append("run_module(ctx)")
                full_content.append("")
                
        with open(output_path, 'w') as f:
            f.write("\n".join(full_content))

if __name__ == "__main__":
    import sys
    arch = Architect("manifest.json")
    # Example test: Intro to Python + Pandas Reading
    test_ids = [
        "Lecture01_introduction", 
        "Lecture01_python_syntax", 
        "Lecture02_reading_with_pandas",
        "Lecture02_dataframes"
    ]
    arch.assemble_lecture(test_ids, "test_synthetic_lecture.py")
    print("Synthesized test_synthetic_lecture.py")
