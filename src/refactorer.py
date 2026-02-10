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
        self.allowed_uppercase = {
            'X', 'Y', 'Z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W'
        }

    def get_free_variables(self, code: str) -> Set[str]:
        """Identifies variables used (Loaded) but not defined (Stored) in the module scope."""
        try:
            root = ast.parse(code)
        except SyntaxError:
            return set()

        # Scope management
        scopes = [set(self.builtins_list)] # Start with builtins
        free_vars = set()

        class ScopeVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Defaults and decorators are evaluated in the outer scope
                for default in node.args.defaults:
                    self.visit(default)
                for decorator in node.decorator_list:
                    self.visit(decorator)

                # Add function name to current scope (after evaluating defaults/decorators)
                scopes[-1].add(node.name)

                # Enter new scope
                scopes.append(set())
                # Add arguments to new scope
                for arg in node.args.args:
                    scopes[-1].add(arg.arg)
                # Visit body
                for item in node.body:
                    self.visit(item)
                # Exit scope
                scopes.pop()

            def visit_AsyncFunctionDef(self, node):
                self.visit_FunctionDef(node)

            def visit_ClassDef(self, node):
                # Decorators and bases are evaluated in the outer scope
                for decorator in node.decorator_list:
                    self.visit(decorator)
                for base in node.bases:
                    self.visit(base)
                for keyword in node.keywords:
                    self.visit(keyword.value)

                # Add class name to current scope
                scopes[-1].add(node.name)
                # Enter new scope
                scopes.append(set())
                # Visit body
                for item in node.body:
                    self.visit(item)
                # Exit scope
                scopes.pop()

            def visit_Import(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name.split('.')[0]
                    scopes[-1].add(name)

            def visit_ImportFrom(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    scopes[-1].add(name)

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    scopes[-1].add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    # Check if defined in any scope from current up to root
                    is_defined = False
                    for scope in reversed(scopes):
                        if node.id in scope:
                            is_defined = True
                            break
                    if not is_defined:
                        free_vars.add(node.id)

            def visit_arg(self, node):
                scopes[-1].add(node.arg)

        visitor = ScopeVisitor()
        visitor.visit(root)
        
        # Heuristic: skip common library aliases
        lib_names = [
            'np', 'pd', 'plt', 'sns', 'tf', 'torch', 'stats', 'sklearn',
            'decomposition', 'cluster', 'metrics', 'model_selection', 
            'linear_model', 'preprocessing', 'ensemble', 'tree', 'neighbors',
            'svm', 'mixture', 'matplotlib', 'numpy', 'scipy'
        ]
        
        return {name for name in free_vars if name not in lib_names}

    def refactor_code(self, code: str, extra_imports: List[str] = None) -> Dict[str, Any]:
        """Wraps code in a function and injects ctx access."""
        free_vars = self.get_free_variables(code)
        
        # Determine requirements based on mapping
        requirements = []
        injections = []
        updates = []
        
        for var in free_vars:
            # Skip likely classes or constants (PascalCase or ALL_CAPS)
            # BUT allow allowed_uppercase matrices (e.g. S, X, Y)
            is_pascal_or_upper = var[0].isupper()
            is_allowed_upper = var in self.allowed_uppercase or var.startswith('X_')

            if is_pascal_or_upper and not is_allowed_upper:
                continue

            # Skip likely library aliases
            # Don't skip 'fig' as it is in mapping
            if var in ['np', 'pd', 'plt', 'stats', 'sklearn', 'ax'] and var not in self.mapping:
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
                # Helper to find assignments in a node, skipping nested functions
                def find_top_level_assigns(n):
                    if isinstance(n, (ast.Assign, ast.AnnAssign)):
                        targets = n.targets if hasattr(n, 'targets') else [n.target]
                        for t in targets:
                            if isinstance(t, ast.Name):
                                yield t.id
                    elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Capture function names defined at top level
                        yield n.name
                    elif isinstance(n, (ast.For, ast.While, ast.If, ast.With, ast.Try)):
                        # These satisfy "top-level" if they are at root
                        for child in ast.iter_child_nodes(n):
                            # Recurse into blocks but stop at nested definitions
                            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                yield from find_top_level_assigns(child)
                            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                # Yield the function name itself, but don't recurse into it
                                yield child.name

                for var_id in find_top_level_assigns(node):
                    # Don't track temporary loop variables
                    if var_id in ['i', 'j', 'k', 'val', 'line', 'f', 'ax']: continue

                    # Logic for Uppercase variables in updates
                    is_pascal_or_upper = var_id[0].isupper()
                    is_allowed_upper = var_id in self.allowed_uppercase or var_id.startswith('X_')

                    if is_pascal_or_upper and not is_allowed_upper:
                        continue
                    
                    if var_id in self.mapping:
                        ctx_key = self.mapping[var_id]
                    else:
                        # Track any reasonable variable name that is assigned to
                        # Include allowed single letters
                        if len(var_id) > 2 or var_id in ['x', 'y', 'z', 'df'] or var_id in self.allowed_uppercase:
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
