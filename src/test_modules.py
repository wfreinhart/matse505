import json
import os
import sys
import ast
import argparse
from pathlib import Path
from typing import List, Dict, Any

class TestRunner:
    def __init__(self, modules_dir: str = "library/modules"):
        self.modules_dir = Path(modules_dir)
        self.modules = self._load_all_modules()

    def _load_all_modules(self) -> List[Dict]:
        modules = []
        for f in self.modules_dir.glob("*.json"):
            with open(f, 'r') as f_in:
                modules.append(json.load(f_in))
        return modules

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
                
                passed += 1
            except Exception as e:
                print(f"FAIL [Static]: {mod_id} ({type(e).__name__}: {e})")
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
        
        for lecture, mods in sorted(lectures.items()):
            if lecture == 'unknown': continue
            
            print(f"\nTesting Lecture: {lecture}")
            # Sort by sequence
            sorted_mods = sorted(mods, key=lambda x: x.get('sequence', 0))
            
            # Shared context for this lecture
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
                        local_scope['run_module'](ctx)
                        # print(f"  PASS: {mod_id}")
                    else:
                        raise ValueError("run_module not found in scope")
                except Exception as e:
                    print(f"  FAIL: {mod_id} ({type(e).__name__}: {e})")
                    lecture_failed = True
                    # In integration testing, we often stop after the first failure in a lecture 
                    # because state becomes invalid, but for now we'll try to continue
            
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
