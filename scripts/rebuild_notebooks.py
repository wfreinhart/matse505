#!/usr/bin/env python3
import subprocess
import os
import sys

def get_changed_files():
    """Returns a list of files changed relative to HEAD (staged + unstaged)."""
    try:
        # git diff --name-only HEAD returns paths relative to repo root
        output = subprocess.check_output(["git", "diff", "--name-only", "HEAD"], text=True)
        return [f.strip() for f in output.splitlines() if f.strip()]
    except subprocess.CalledProcessError as e:
        print(f"Error checking git diff: {e}")
        sys.exit(1)

def main():
    changed_files = get_changed_files()
    
    # Filter for python files in lectures/ directory
    lecture_py_files = [f for f in changed_files if f.startswith('lectures/') and f.endswith('.py')]
    
    if not lecture_py_files:
        print("No changed lecture files found.")
        return

    print(f"Found {len(lecture_py_files)} changed lecture files.")
    
    for py_file in lecture_py_files:
        # Determine the target notebook path
        # Using the standard repo structure: lectures/Name.py -> notebooks/Name.ipynb
        notebook_path = py_file.replace('lectures/', 'notebooks/').replace('.py', '.ipynb')
        
        print(f"Rebuilding {notebook_path} from {py_file}...")
        
        # Command to rebuild the notebook. 
        # --to ipynb converts the script to a notebook
        # --output ensures it goes to the correct location
        cmd = [
            "jupytext",
            "--to", "ipynb",
            "--output", notebook_path,
            py_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully rebuilt {notebook_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error rebuilding {notebook_path}: {e}")

if __name__ == "__main__":
    main()
