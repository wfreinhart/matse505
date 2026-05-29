import json
import os
import sys
import ast
import argparse
import builtins
from pathlib import Path
from typing import List, Dict, Any, Set
from unittest.mock import MagicMock, patch

class CourseMock(MagicMock):
    """A mock specifically designed for scientific library patterns in courseware."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Avoid infinite recursion in __getattr__ by pre-setting some flags if needed
        # but MagicMock handles most of this.

    def __getattr__(self, name):
        if name.startswith('_'):
            if name in ['__str__', '__repr__']:
                return lambda: "CourseMock"
            return super().__getattr__(name)
        
        m = CourseMock()
        # Heuristic return values for common functions
        if name in ['train_test_split']:
            m.return_value = [CourseMock() for _ in range(4)]
        elif name in ['subplots', 'span']:
            m.return_value = [CourseMock() for _ in range(2)]
        elif name in ['shape', 'size', 'columns']:
            return (10, 2)
        elif name in ['classification_report', 'summary']:
            def mock_func(*args, **kwargs): return "mock_report"
            return mock_func
        elif name in ['loc', 'iloc']:
            # This is an attribute that is also indexable
            return self
        else:
            # By default, functions return another CourseMock
            m.return_value = m
            
        return m

    def __str__(self):
        return "CourseMock"
    def __repr__(self):
        return "CourseMock"
    def __iter__(self):
        # Default iteration returns 2 items for common unpacking (shape, subplots)
        return iter([CourseMock(), CourseMock()])
    def __getitem__(self, key):
        return self
    def __setitem__(self, key, value):
        pass
    def __len__(self):
        return 10

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
            'tqdm', 'umap', 'pygad',
            'rdkit', 'rdkit.Chem', 'rdkit.Chem.AllChem', 'rdkit.Chem.Draw',
            'torch_geometric', 'torch_geometric.data', 'torch_geometric.nn', 'torch_geometric.nn.conv',
            'sdv', 'missforest', 'nflows', 'pytorch_lightning', 'IPython'
        ]
        self.patchers = []

    def __enter__(self):
        for mod_name in self.modules_to_mock:
            patcher = patch.dict(sys.modules, {mod_name: CourseMock()})
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
        errors = []
        code = mod.get('code_block', '')
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"SyntaxError: {e}"]
        run_func = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'run_module':
                run_func = node
                break
        if not run_func:
            return ["run_module function not found"]
        defined_vars = set()
        for var in run_func.args.args:
            defined_vars.add(var.arg)
        class UsageVisitor(ast.NodeVisitor):
            def __init__(self):
                self.defined = set(defined_vars)
                self.loaded = set()
            def visit_Import(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name.split('.')[0]
                    self.defined.add(name)
            def visit_ImportFrom(self, node):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    self.defined.add(name)
            def visit_Assign(self, node):
                self.visit(node.value)
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
        for node in run_func.body:
            visitor.visit(node)
        builtin_names = dir(builtins)
        ignored = ['__name__', '__file__', 'display', 'get_ipython', 'exit', 'quit']
        potential_leaks = []
        for var in visitor.loaded:
            if var not in visitor.defined and var not in builtin_names and var not in ignored:
                potential_leaks.append(var)
        if potential_leaks:
            errors.append(f"Potential undefined variables (leaks): {sorted(potential_leaks)}")
        return errors

    def run_static_validation(self):
        print("--- Tier 1: Static Validation ---")
        passed = 0
        failed = 0
        for mod in self.modules:
            mod_id = mod.get('id', 'unknown')
            try:
                required = ['id', 'markdown_content', 'code_block']
                for field in required:
                    if field not in mod:
                        raise ValueError(f"Missing required field: {field}")
                tree = ast.parse(mod['code_block'])
                found_func = False
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == 'run_module':
                        found_func = True
                        break
                if not found_func:
                    raise ValueError("run_module(ctx) function not found")
                robust_errors = self.run_robust_static_validation(mod)
                if robust_errors:
                    print(f"  WARN [Static]: {mod_id} -> {robust_errors}")
                passed += 1
            except Exception as e:
                print(f"FAIL [Static]: {mod_id} in {mod.get('_filename')} ({type(e).__name__}: {e})")
                failed += 1
        print(f"Tier 1 Summary: {passed} passed, {failed} failed")
        return failed == 0

    def run_integration_test(self, lecture_name: str = None):
        print(f"--- Tier 2: Integration Test ({lecture_name or 'All Lectures'}) ---")
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
        with MockEnvironment():
            for lecture, mods in sorted(lectures.items()):
                if lecture == 'unknown': continue
                print(f"\nTesting Lecture: {lecture}")
                sorted_mods = sorted(mods, key=lambda x: x.get('sequence', 0))
                
                # Context with standard expected keys
                ctx = {
                    'data': CourseMock(),
                    'tensors': {},
                    'model': CourseMock(),
                    'model_artifacts': {'encoder': CourseMock(), 'scaler': CourseMock()},
                    'viz': CourseMock()
                }
                
                lecture_failed = False
                for mod in sorted_mods:
                    mod_id = mod['id']
                    try:
                        local_scope = {}
                        exec(mod['code_block'], globals(), local_scope)
                        if 'run_module' in local_scope:
                            local_scope['run_module'](ctx)
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
        if not runner.run_static_validation(): overall_success = False
        if not runner.run_integration_test(): overall_success = False
    if not overall_success:
        sys.exit(1)
