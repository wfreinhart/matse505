import ast
import json
import os
import builtins
from typing import List, Set, Dict, Any

class Refactorer:
    def __init__(self):
        self.builtins_list = dir(builtins)
        # Mapping common course variables to ctx keys
        self.mapping = {
            'data': 'data',
            'df': 'data',
            'ele_data': 'data',
            'xyz_data': 'data',
            'x': 'tensors[\'x\']',
            'y': 'tensors[\'y\']',
            'z': 'tensors[\'z\']',
            'model': 'model',
            'clf': 'model',
            'reg': 'model',
            'encoder': 'model_artifacts[\'encoder\']',
            'scaler': 'model_artifacts[\'scaler\']',
            "X_train": "tensors['X_train']",
            "X_test": "tensors['X_test']",
            "y_train": "tensors['y_train']",
            "y_test": "tensors['y_test']",
            "fig": "viz"
        }

    def get_free_variables(self, code: str) -> Set[str]:
        """Identifies variables used (Loaded) before they are defined (Stored) using in-order traversal."""
        try:
            root = ast.parse(code)
        except SyntaxError:
            return set()
            
        class VariableVisitor(ast.NodeVisitor):
            def __init__(self, builtins_list):
                self.first_usage = {}
                self.builtins_list = builtins_list

            def visit_Assign(self, node):
                self.visit(node.value)
                for t in node.targets:
                    self.visit(t)

            def visit_AnnAssign(self, node):
                if node.value:
                    self.visit(node.value)
                self.visit(node.target)

            def visit_Name(self, node):
                if node.id not in self.first_usage:
                    self.first_usage[node.id] = type(node.ctx)
                self.generic_visit(node)

            def visit_arg(self, node):
                if node.arg not in self.first_usage:
                    self.first_usage[node.arg] = ast.Store
                self.generic_visit(node)

        visitor = VariableVisitor(self.builtins_list)
        visitor.visit(root)
        
        used_before_defined = {
            name for name, ctx_type in visitor.first_usage.items() 
            if ctx_type == ast.Load and name not in self.builtins_list
        }
        
        # Heuristic: skip common library aliases
        lib_names = [
            'np', 'pd', 'plt', 'sns', 'tf', 'torch', 'stats', 'sklearn',
            'decomposition', 'cluster', 'metrics', 'model_selection', 
            'linear_model', 'preprocessing', 'ensemble', 'tree', 'neighbors',
            'svm', 'mixture', 'matplotlib', 'numpy', 'os', 'requests',
            'sys', 're', 'json', 'math', 'random', 'time', 'datetime',
            'pathlib', 'shutil', 'glob', 'argparse', 'yaml', 'urllib'
        ]
        
        return {name for name in used_before_defined if name not in lib_names}

    def refactor_code(self, code: str, extra_imports: List[str] = None) -> Dict[str, Any]:
        """Wraps code in a function and injects ctx access."""
        free_vars = self.get_free_variables(code)
        
        # Determine requirements based on mapping
        requirements = []
        injections = []
        updates = []
        
        for var in free_vars:
            # Skip likely classes or constants (PascalCase or ALL_CAPS)
            if var[0].isupper() and not var.startswith('X'):
                continue
            # Skip likely library aliases
            if var in ['np', 'pd', 'plt', 'stats', 'sklearn', 'ax', 'fig']:
                continue

            if var in self.mapping:
                ctx_key = self.mapping[var]
            else:
                ctx_key = f"tensors['{var}']"
                
            requirements.append(ctx_key)
            if '[' not in ctx_key:
                injections.append(f"{var} = ctx.get('{ctx_key}')")
            else:
                parts = ctx_key.replace("'", "").replace("[", " ").replace("]", "").split()
                key1, key2 = parts[0], parts[1]
                injections.append(f"{var} = ctx.get('{key1}', {{}}).get('{key2}')")
                
        # Heuristic for updates: assign names in top-level of module
        try:
            root = ast.parse(code)
            # Find all assignments that are NOT inside a nested FunctionDef
            for node in ast.iter_child_nodes(root):
                # If the code block is just a series of statements, root.body has them.
                # If there's a function def at top level, we don't want to track its internals.
                
                # Helper to find assignments in a node, skipping nested functions
                def find_top_level_assigns(n):
                    if isinstance(n, (ast.Assign, ast.AnnAssign)):
                        targets = n.targets if hasattr(n, 'targets') else [n.target]
                        for t in targets:
                            if isinstance(t, ast.Name):
                                yield t.id
                    elif isinstance(n, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
                        # These satisfy "top-level" if they are at root
                        for child in ast.iter_child_nodes(n):
                            # We might want to go deeper but skip FunctionDef/ClassDef
                            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                yield from find_top_level_assigns(child)

                for var_id in find_top_level_assigns(node):
                    # Don't track temporary loop variables or likely classes
                    if var_id[0].isupper() and not var_id.startswith('X'): continue
                    if var_id in ['i', 'j', 'k', 'val', 'line', 'f', 'ax', 'fig']: continue
                    
                    if var_id in self.mapping:
                        ctx_key = self.mapping[var_id]
                    else:
                        # Track any reasonable variable name that is assigned to
                        if len(var_id) > 2 or var_id in ['x', 'y', 'z', 'df']:
                            ctx_key = f"tensors['{var_id}']"
                        else:
                            continue

                    if '[' not in ctx_key:
                        updates.append(f"ctx['{ctx_key}'] = {var_id}")
                    else:
                        key_base = ctx_key.split('[')[0]
                        key_sub = ctx_key.split("'")[1]
                        updates.append(f"ctx['{key_base}']['{key_sub}'] = {var_id}")
        except Exception as e:
            # print(f"DEBUG Error: {e}")
            pass

        # Build the final function
        refactored_lines = ["def run_module(ctx):"]
        refactored_lines.append("    import numpy as np")
        refactored_lines.append("    import pandas as pd")
        refactored_lines.append("    from matplotlib import pyplot as plt")
        refactored_lines.append("    from scipy import stats")
        refactored_lines.append("    import sklearn")
        refactored_lines.append("    from sklearn import metrics, model_selection, linear_model, cluster, decomposition, preprocessing, ensemble, tree, neighbors, svm, mixture")
        
        # Inject extra imports from the original lecture
        if extra_imports:
            for imp in sorted(list(set(extra_imports))):
                refactored_lines.append(f"    {imp}")
        
        # Injections
        for inj in sorted(list(set(injections))):
            refactored_lines.append(f"    {inj}")
        
        # Original code indented
        code_lines = [l for l in code.split('\n') if l.strip()]
        if not code_lines and not injections and not updates:
            refactored_lines.append("    pass")
        else:
            for line in code.split('\n'):
                refactored_lines.append(f"    {line}")
            
        # Updates
        for upd in sorted(list(set(updates))):
            refactored_lines.append(f"    {upd}")
            
        return {
            "refactored_code": "\n".join(refactored_lines),
            "requirements": list(set(requirements))
        }

def process_raw_segment(input_path: str, output_dir: str):
    with open(input_path, 'r') as f:
        seg = json.load(f)
    
    ref = Refactorer()
    result = ref.refactor_code(seg["code_block"], extra_imports=seg.get("imports", []))
    
    module = {
        "id": seg["id"],
        "type": "Foundational", 
        "sequence": seg.get("sequence", 0),
        "parent_lecture": seg.get("parent_lecture", ""),
        "requirements": result["requirements"],
        "markdown_content": seg["markdown_content"],
        "code_block": result["refactored_code"]
    }
    
    with open(os.path.join(output_dir, f"{seg['id']}.json"), 'w') as f:
        json.dump(module, f, indent=2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process_raw_segment(sys.argv[1], "library/modules")
