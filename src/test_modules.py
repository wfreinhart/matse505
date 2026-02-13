import json
import os
import sys
import ast
import argparse
import builtins
from pathlib import Path
from typing import List, Dict, Any, Set
from unittest.mock import MagicMock, patch

class MockEnvironment:
    """Context manager to mock missing scientific libraries."""
    def __init__(self):
        self.modules_to_mock = [
            'numpy', 'pandas', 'matplotlib', 'matplotlib.pyplot',
            'scipy', 'scipy.stats', 'scipy.cluster', 'scipy.cluster.hierarchy',
            'scipy.spatial', 'scipy.spatial.distance', 'scipy.optimize',
            'sklearn', 'sklearn.metrics',
            'sklearn.model_selection', 'sklearn.linear_model',
            'sklearn.cluster', 'sklearn.decomposition',
            'sklearn.preprocessing', 'sklearn.ensemble',
            'sklearn.tree', 'sklearn.neighbors', 'sklearn.svm',
            'sklearn.mixture', 'sklearn.datasets', 'sklearn.feature_selection',
            'seaborn', 'tensorflow', 'torch',
            'plotly', 'plotly.express', 'plotly.graph_objects',
            'requests', 'urllib', 'urllib.request',
            'ax', 'ax.service', 'ax.service.ax_client', 'ax.service.managed_loop', 'ax.plot', 'ax.plot.contour', 'ax.plot.trace', 'ax.utils', 'ax.utils.notebook', 'ax.utils.notebook.plotting',
            'tqdm', 'umap', 'pygad'
        ]
        self.patchers = []

    def __enter__(self):
        # We need to mock sys.modules so that 'import numpy' works
        # and returns a MagicMock object.
        for mod_name in self.modules_to_mock:
            patcher = patch.dict(sys.modules, {mod_name: MagicMock()})
            patcher.start()
            self.patchers.append(patcher)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for patcher in self.patchers:
            patcher.stop()

class TestRunner:
    def __init__(self, modules_dir: str = "library/modules"):
        self.modules_dir = Path(modules_dir)
        self.modules = self._load_all_modules()

    def _load_all_modules(self) -> List[Dict]:
        modules = []
        # Sort to ensure consistent order
        files = sorted(list(self.modules_dir.glob("*.json")))
        for f in files:
            with open(f, 'r') as f_in:
                try:
                    mod = json.load(f_in)
                    mod['_filename'] = str(f)
                    modules.append(mod)
                except Exception as e:
                    print(f"Failed to load {f}: {e}")
        return modules

    def run_robust_static_validation(self, mod: Dict) -> List[str]:
        """Performs deeper analysis on variable usage."""
        errors = []
        code = mod.get('code_block', '')

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"SyntaxError: {e}"]

        # Find the run_module function
        run_func = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'run_module':
                run_func = node
                break

        if not run_func:
            return ["run_module function not found"]

        # Analyze variable usage within run_module
        # We want to catch variables that are Loaded but not Stored (and not builtins/imports)

        defined_vars = set()
        loaded_vars = set()

        # Add function arguments (ctx)
        for arg in run_func.args.args:
            defined_vars.add(arg.arg)

        class UsageVisitor(ast.NodeVisitor):
            def __init__(self):
                self.defined = set(defined_vars)
                self.loaded = set()
                self.imports = set()

            def visit_Import(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name.split('.')[0]
                    self.defined.add(name)
                    self.imports.add(name)

            def visit_ImportFrom(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    self.defined.add(name)
                    self.imports.add(name)

            def visit_Assign(self, node):
                # Visit value first (load)
                self.visit(node.value)
                # Then visit targets (store)
                for target in node.targets:
                    self._mark_defined(target)

            def visit_AnnAssign(self, node):
                if node.value:
                    self.visit(node.value)
                self._mark_defined(node.target)

            def visit_For(self, node):
                self.visit(node.iter)
                self._mark_defined(node.target)
                for item in node.body:
                    self.visit(item)
                for item in node.orelse:
                    self.visit(item)

            def visit_FunctionDef(self, node):
                # Don't recurse into nested functions for now to keep simple
                # but mark the function name as defined
                self.defined.add(node.name)

            def _mark_defined(self, node):
                if isinstance(node, ast.Name):
                    self.defined.add(node.id)
                elif isinstance(node, (ast.Tuple, ast.List)):
                    for elt in node.elts:
                        self._mark_defined(elt)

            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Load):
                    self.loaded.add(node.id)

        visitor = UsageVisitor()
        # Visit the body of the function
        for node in run_func.body:
            visitor.visit(node)

        # Check for loaded vars that were never defined
        # Exclude builtins
        builtin_names = dir(builtins)
        # Exclude common test/execution context globals that might be implicitly available?
        # No, refactored code should be self-contained within run_module + passed ctx.

        # Common false positives
        ignored = ['__name__', '__file__', 'display', 'get_ipython', 'exit', 'quit']

        potential_leaks = []
        for var in visitor.loaded:
            if var not in visitor.defined and var not in builtin_names and var not in ignored:
                potential_leaks.append(var)

        if potential_leaks:
            errors.append(f"Potential undefined variables (leaks): {sorted(potential_leaks)}")

        return errors

    def run_static_validation(self):
        """Tier 1: fast, deterministic syntax and structure checks."""
        print("--- Tier 1: Static Validation ---")
        passed = 0
        failed = 0
        
        for mod in self.modules:
            mod_id = mod.get('id', 'unknown')
            try:
                # 1. Check structure
                required = ['id', 'markdown_content', 'code_block']
                for field in required:
                    if field not in mod:
                        raise ValueError(f"Missing required field: {field}")
                
                # 2. Check syntax
                tree = ast.parse(mod['code_block'])
                
                # 3. Check for run_module function
                found_func = False
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == 'run_module':
                        found_func = True
                        break
                if not found_func:
                    raise ValueError("run_module(ctx) function not found")
                
                # 4. Robust checks
                robust_errors = self.run_robust_static_validation(mod)
                if robust_errors:
                    # Treat robust errors as warnings for now unless critical?
                    # The user asked for "better testing procedures", so let's report them.
                    # We won't fail Tier 1 yet, but we'll print them.
                    print(f"  WARN [Static]: {mod_id} -> {robust_errors}")

                passed += 1
            except Exception as e:
                print(f"FAIL [Static]: {mod_id} in {mod.get('_filename')} ({type(e).__name__}: {e})")
                failed += 1
                
        print(f"Tier 1 Summary: {passed} passed, {failed} failed")
        return failed == 0

    def run_integration_test(self, lecture_name: str = None):
        """Tier 2: sequential execution per lecture."""
        print(f"--- Tier 2: Integration Test ({lecture_name or 'All Lectures'}) ---")
        
        # Group modules by lecture
        lectures = {}
        for mod in self.modules:
            parent = mod.get('parent_lecture', 'unknown')
            if lecture_name and parent != lecture_name:
                continue
            if parent not in lectures:
                lectures[parent] = []
            lectures[parent].append(mod)
            
        total_passed = 0
        total_failed = 0
        
        # Use the Mock Environment
        with MockEnvironment():
            for lecture, mods in sorted(lectures.items()):
                if lecture == 'unknown': continue

                print(f"\nTesting Lecture: {lecture}")
                # Sort by sequence
                sorted_mods = sorted(mods, key=lambda x: x.get('sequence', 0))

                # Shared context for this lecture
                # The ctx is a real dictionary, but the objects inside might be Mocks if generated by mocked libs
                ctx = {
                    'data': None,
                    'tensors': {},
                    'model': None,
                    'model_artifacts': {'encoder': None, 'scaler': None},
                    'viz': None
                }

                lecture_failed = False
                for mod in sorted_mods:
                    mod_id = mod['id']
                    try:
                        # Execute
                        local_scope = {}
                        # We use globals() to ensure libraries imported in the block are available
                        exec(mod['code_block'], globals(), local_scope)
                        if 'run_module' in local_scope:
                            # Pass the ctx. Mocks will handle method calls gracefully (returning more Mocks)
                            local_scope['run_module'](ctx)
                            # Verify ctx modifications?
                            # Ideally we check if ctx['data'] changed if the module was supposed to load data.
                            # But that's hard to know statically.
                        else:
                            raise ValueError("run_module not found in scope")
                    except Exception as e:
                        print(f"  FAIL: {mod_id} ({type(e).__name__}: {e})")
                        lecture_failed = True

                if not lecture_failed:
                    print(f"SUCCESS: All modules in {lecture} passed in sequence.")
                    total_passed += 1
                else:
                    total_failed += 1
                
        print(f"\nTier 2 Summary: {total_passed} lectures passed, {total_failed} lectures failed")
        return total_failed == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true", help="Run Tier 1 static validation")
    parser.add_argument("--lecture", type=str, help="Run Tier 2 for a specific lecture")
    parser.add_argument("--all", action="store_true", help="Run both Tier 1 and Tier 2 for all lectures")
    args = parser.parse_args()
    
    runner = TestRunner()
    overall_success = True
    
    if args.static or args.all:
        if not runner.run_static_validation():
            overall_success = False
            
    if args.lecture or args.all:
        if not runner.run_integration_test(args.lecture):
            overall_success = False
            
    if not (args.static or args.lecture or args.all):
        # Default behavior: run both
        if not runner.run_static_validation(): overall_success = False
        if not runner.run_integration_test(): overall_success = False

    if not overall_success:
        sys.exit(1)
