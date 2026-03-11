"""
Christina
Mar 10th, 2026
Organize the PNG Bundles into Train (70%)/validation (10%)/Test (20%) sets
The PNG Bundles are grouped based on patient_id to make sure images of 1 patient stays in 1 set
Images grouped based on pathology labels in the report_fixed,json in the label.zip
"""
import pandas as pd
import shutil
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# ADJUSTMENT REQUIRED: Locations of files
JSON_LABELS_PATH = "report_fixed.json"      # Path to .json file
# Image_bundles is a list of all bundle folder locations! 
IMAGE_BUNDLES = ["chexpert/bundle1", "chexpert/bundle2", "chexpert/bundle3"]
OUTPUT_BASE_DIR = "./organized_data/" # Where you want the output to be

# We will group based on pathology labels - Alphabetical order
# No finding when all other labels are non-positive (including uncertain & null)
# IMPORTANT LOGIC: if patient has multiple disease, they will be placed under the label in the front of the list
PATHO_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", 
    "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", 
    "Support Devices"
]

def get_images(json_path) -> None:
    # The json file has the image path in .jpg, if actually bundle .jpg, skip this
    actual_path = json_path.replace(".jpg", ".png")
    for p in IMAGE_BUNDLES:
        full_path = os.path.join(p, actual_path)
        if os.path.exists(full_path):
            return full_path
    return None

def get_patient_id(path_string) -> str:
    # Extracts 'patient42142' from 'train/patient42142/study5/view1_frontal.jpg'
    parts = path_string.split('/')
    return parts[1] if len(parts) > 1 else "unknown"

def organize_from_json():
    # Load JSON data
    with open(JSON_LABELS_PATH, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # Extract Patient IDs for splitting
    df['patient_id'] = df['path_to_image'].apply(get_patient_id)
    unique_patients = df['patient_id'].unique()

    # Patient Split (70/10/20)
    # First split: Train + Val 80 - Test 20
    train_val, test_pts = train_test_split(unique_patients, test_size=0.20, random_state=42)
    # 2nd split: Val from Train
    train_pts, val_pts = train_test_split(train_val, test_size=0.125, random_state=42)

    split_sets = {'train': set(train_pts), 'val': set(val_pts), 'test': set(test_pts)}

    # Create Folders and Copy Files
    for _, row in df.iterrows():
        p_id = row['patient_id']
        current_set = next((s for s, ids in split_sets.items() if p_id in ids), None)
        
        if current_set:
            # Determine folder by finding the first 1.0 label
            primary_label = "No_Finding"
            for l in PATHO_LABELS:
                if row[l] == 1.0:
                    primary_label = l.replace(" ", "_")
                    break
            
            # Get Image Path
            src_path = get_images(row['path_to_image'])
            dest_folder = os.path.join(OUTPUT_BASE_DIR, current_set, primary_label)
            Path(dest_folder).mkdir(parents=True, exist_ok=True)
            
            # Unique filename to avoid overwriting (patientID_study_view)
            file_parts = src_path.replace(os.sep, "/").split("/")
            file_name = "_".join(file_parts[-3:])
            dest_path = os.path.join(dest_folder, file_name)

            if os.path.exists(src_path) and not os.path.exists(dest_path):
                # If sufficient storage: shutil.copy2(src_path, dest_path)
                # If no storage:
                os.symlin(os.path.abspath(src_path), dest_path)

    print(f"Done! Data split and organized in {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    organize_from_json()
