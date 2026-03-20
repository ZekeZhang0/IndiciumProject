import pandas as pd
import shutil
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

JSON_LABELS_PATH = "report_fixed.json"
BUNDLE_PATH = "chexpert/bundle1"
OUTPUT_BASE_DIR = "./organized_data/"

PATHO_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", 
    "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", 
    "Support Devices"
]

def get_patient_id(path_obj):
    # path_obj is a pathlib Path. Structure: .../train/patientXXXXX/studyX/viewX.png
    # We want the parent of the parent of the file.
    return path_obj.parts[-3] 

def organize_single_bundle():
    # 1. Load JSON into a dictionary for fast lookup
    print("Loading JSON labels...")
    with open(JSON_LABELS_PATH, 'r') as f:
        json_data = json.load(f)
    
    # Create a lookup table where key is the 'path_to_image'
    label_lookup = {item['path_to_image']: item for item in json_data}

    # 2. Find all PNGs in the bundle
    print(f"Scanning {BUNDLE_PATH} for images...")
    bundle_path = Path(BUNDLE_PATH)
    image_files = list(bundle_path.rglob("*.png"))
    
    if not image_files:
        print("No images found! Check your BUNDLE_PATH.")
        return

    # 3. Extract Unique Patients
    patient_to_files = {}
    for img_path in image_files:
        pid = get_patient_id(img_path)
        if pid not in patient_to_files:
            patient_to_files[pid] = []
        patient_to_files[pid].append(img_path)
    
    unique_patients = list(patient_to_files.keys())
    print(f"Found {len(unique_patients)} unique patients.")

    # 4. Split Patients (70/10/20)
    train_val, test_pts = train_test_split(unique_patients, test_size=0.20, random_state=42)
    train_pts, val_pts = train_test_split(train_val, test_size=0.125, random_state=42)

    split_map = {
        **{p: 'train' for p in train_pts},
        **{p: 'val' for p in val_pts},
        **{p: 'test' for p in test_pts}
    }

    # 5. Process Files
    print("Organizing files...")
    count = 0
    for pid, files in patient_to_files.items():
        current_split = split_map[pid]
        
        for src_path in files:
            # Prepare path for JSON lookup
            # We need to convert: 'chexpert/bundle1/train/p1/s1/v1.png' 
            # to match JSON: 'train/p1/s1/v1.jpg'
            path_parts = src_path.parts
            # Find index of 'train' or 'valid' to match CheXpert standard
            try:
                start_idx = path_parts.index("train")
            except ValueError:
                start_idx = path_parts.index("valid")
            
            json_match_path = "/".join(path_parts[start_idx:]).replace(".png", ".jpg")
            
            # Find Label
            label_data = label_lookup.get(json_match_path)
            primary_label = "No_Finding"
            
            if label_data:
                for l in PATHO_LABELS:
                    if label_data.get(l) == 1.0:
                        primary_label = l.replace(" ", "_")
                        break
            
            # Create destination
            dest_folder = Path(OUTPUT_BASE_DIR) / current_split / primary_label
            dest_folder.mkdir(parents=True, exist_ok=True)
            
            # Unique filename (patient_study_view.png)
            new_name = "_".join(src_path.parts[-3:])
            dest_path = dest_folder / new_name

            # Copy or Symlink
            if not dest_path.exists():
                # Alternatively, use shutil.copy2
                os.symlink(src_path.absolute(), dest_path)
                count += 1

    print(f"Done! Processed {count} images into {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    organize_single_bundle()