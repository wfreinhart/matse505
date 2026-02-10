import json
import os

def generate_manifest(modules_dir: str, output_path: str):
    manifest = {
        "modules": [],
        "datasets": []
    }
    
    for f in os.listdir(modules_dir):
        if f.endswith('.json'):
            with open(os.path.join(modules_dir, f), 'r') as file:
                mod = json.load(file)
                manifest["modules"].append({
                    "id": mod["id"],
                    "requirements": mod["requirements"],
                    "type": mod["type"],
                    "path": f"library/modules/{f}"
                })
                
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    generate_manifest("library/modules", "manifest.json")
