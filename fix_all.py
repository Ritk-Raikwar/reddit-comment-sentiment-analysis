import json
import os

folder_path = './notebooks'

for filename in os.listdir(folder_path):
    if filename.endswith(".ipynb"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                nb = json.load(f)
            except Exception as e:
                print(f"Could not read {filename}: {e}")
                continue
        
        if "metadata" in nb and "widgets" in nb["metadata"]:
            del nb["metadata"]["widgets"]
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=2)
            print(f"Fixed: {filename}")
