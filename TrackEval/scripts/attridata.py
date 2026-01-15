import os
import shutil
import subprocess
import sys

# ==============================================================================
# --- 1. Global Configuration: Configure all your paths here ---
# ==============================================================================

# --- Paths for Data Preprocessing ---
# Your original results root directory (e.g., '.../results_epoch79')
INPUT_RESULTS_DIR = '/path/to/your/results/visdrone'
# Your attribute annotation files root directory (e.g., '.../Attribute')
INPUT_ATTRIBUTES_DIR = '/path/to/your/attributes/Attribute'
# Output directory for preprocessed data (will be created automatically)
PROCESSED_DATA_ROOT = './processed_attribute_results'

# --- Paths for Evaluation ---
# Your original seqmap file containing all sequences and expressions
ORIGINAL_SEQMAP_FILE = "/path/to/your/seqmap.txt"
# The name you need to specify for --TRACKERS_TO_EVAL parameter in evaluation command
# Path to run_mot_challenge.py script
EVAL_SCRIPT_PATH = 'run_mot_challenge.py'


# ==============================================================================
# --- 2. Phase One: Data Preprocessing Functions ---
# ==============================================================================

def preprocess_data_by_attribute():
    """
    Filter and reorganize GT and prediction data based on attribute files.
    """
    print("=" * 70)
    print("--- Phase One: Starting Data Preprocessing ---")
    print("=" * 70)

    # Define attribute names
    attributes = [
        'Day', 'Night', 'ViewPoint_Change', 'Scale_Variation',
        'Occlusion', 'Fast_Motion', 'Rotation', 'Low_Resolution'
    ] 
    """
    attributes = [
        'ViewPoint_Change'
    ]"""
    if os.path.exists(PROCESSED_DATA_ROOT):
        print(f"INFO: Output directory '{PROCESSED_DATA_ROOT}' already exists, clearing...")
        shutil.rmtree(PROCESSED_DATA_ROOT)

    if not os.path.isdir(INPUT_RESULTS_DIR):
        print(f"Error: Results root directory '{INPUT_RESULTS_DIR}' does not exist.")
        return False

    # Iterate through each sequence
    for seq_name in sorted(os.listdir(INPUT_RESULTS_DIR)):
        seq_path = os.path.join(INPUT_RESULTS_DIR, seq_name)
        if not os.path.isdir(seq_path):
            continue

        print(f"\n--- Processing sequence: {seq_name} ---")

        # Load attribute file
        attr_file_path = os.path.join(INPUT_ATTRIBUTES_DIR, f"{seq_name}.txt")
        if not os.path.isfile(attr_file_path):
            print(f"  - Warning: Attribute file for sequence '{seq_name}' not found, skipping.")
            continue
        
        frame_attributes = {}
        try:
            with open(attr_file_path, 'r') as f:
                for i, line in enumerate(f):
                    frame_id = i + 1  # Assuming frame numbers start from 1
                    attr_values = [int(v) for v in line.strip().split(',')]
                    frame_attributes[frame_id] = attr_values
        except Exception as e:
            print(f"  - Error: Failed to read attribute file '{attr_file_path}': {e}")
            continue

        # Iterate through each description
        for rmot_desc_name in sorted(os.listdir(seq_path)):
            rmot_desc_path = os.path.join(seq_path, rmot_desc_name)
            if not os.path.isdir(rmot_desc_path):
                continue
            
            print(f"  - Processing description: '{rmot_desc_name}'")

            filtered_gt = {attr: [] for attr in attributes}
            filtered_pred = {attr: [] for attr in attributes}
            
            # Process GT file
            gt_path = os.path.join(rmot_desc_path, 'gt.txt')
            if os.path.isfile(gt_path):
                with open(gt_path, 'r') as f:
                    for line in f:
                        try:
                            frame_id = int(line.strip().split(',')[0])
                            if frame_id in frame_attributes:
                                attr_flags = frame_attributes[frame_id]
                                for i, attr_name in enumerate(attributes):
                                    if attr_flags[i] == 1:
                                        filtered_gt[attr_name].append(line)
                        except (ValueError, IndexError):
                            continue
            
            # Process Prediction file
            pred_path = os.path.join(rmot_desc_path, 'predict.txt')
            if os.path.isfile(pred_path):
                with open(pred_path, 'r') as f:
                    for line in f:
                        try:
                            frame_id = int(line.strip().split(',')[0])
                            if frame_id in frame_attributes:
                                attr_flags = frame_attributes[frame_id]
                                for i, attr_name in enumerate(attributes):
                                    if attr_flags[i] == 1:
                                        filtered_pred[attr_name].append(line)
                        except (ValueError, IndexError):
                            continue

            # Write filtered results to new files
            for attr_name in attributes:
                if filtered_gt[attr_name] or filtered_pred[attr_name]:
                    target_dir = os.path.join(PROCESSED_DATA_ROOT, attr_name, seq_name, rmot_desc_name)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    with open(os.path.join(target_dir, 'gt.txt'), 'w') as f_out:
                        f_out.writelines(filtered_gt[attr_name])
                    with open(os.path.join(target_dir, 'predict.txt'), 'w') as f_out:
                        f_out.writelines(filtered_pred[attr_name])

    print("\n--- Phase One: Data Preprocessing Complete! ---")
    return True


# ==============================================================================
# --- 3. Phase Two: Evaluation Execution Functions ---
# ==============================================================================

def run_evaluation_by_attribute():
    """
    Iterate through preprocessed result directories, generate temporary seqmap for each attribute and run evaluation.
    """
    print("\n" + "=" * 70)
    print("--- Phase Two: Starting Evaluation ---")
    print("=" * 70)
    
    if not os.path.isfile(ORIGINAL_SEQMAP_FILE):
        print(f"Error: Original seqmap file '{ORIGINAL_SEQMAP_FILE}' not found!")
        return
    if not os.path.isdir(PROCESSED_DATA_ROOT):
        print(f"Error: Preprocessed results directory '{PROCESSED_DATA_ROOT}' not found.")
        return
    
    try:
        with open(ORIGINAL_SEQMAP_FILE, 'r') as f:
            original_seqmap_lines = f.readlines()
        original_seqmap_header = original_seqmap_lines[0] if original_seqmap_lines else ''
    except Exception as e:
        print(f"Error: Failed to read original seqmap file: {e}")
        return

    # Iterate through each attribute folder
    for attr_name in sorted(os.listdir(PROCESSED_DATA_ROOT)):
        attr_dir_path = os.path.join(PROCESSED_DATA_ROOT, attr_name)
        if not os.path.isdir(attr_dir_path):
            continue

        print(f"\n--- Preparing evaluation for attribute '{attr_name}' ---")
        temp_seqmap_file_path = os.path.join(attr_dir_path, "seqmap_temp.txt")
        # Iterate through each line in the original seqmap (e.g., "M0203+Automobiles moving from upper to lower in field of view")
        lines_to_write = []  
        for line in original_seqmap_lines:
            # Ensure line format is correct
            if '+' not in line:
                continue
            line = line.strip() 
            # Split sequence name and description name
            seq_name, rmot_desc_name = line.split('+', 1)
            
            # Check if gt.txt file exists for this "sequence+description" combination under current attribute folder
            # This is the most reliable method to determine if this combination is valid
            expected_gt_path = os.path.join(attr_dir_path, seq_name, rmot_desc_name, 'gt.txt')
            # Add these two lines for debugging
            #print(f"Checking path: {os.path.abspath(expected_gt_path)}")
            #print(f"File exists? {os.path.isfile(expected_gt_path)}")
                      
            # Only add to temporary seqmap if gt file exists (meaning it has at least one frame belonging to current attribute)
            if os.path.isfile(expected_gt_path):
                lines_to_write.append(line + '\n')
        
        if len(lines_to_write) <= 1:
            print(f"  - INFO: No valid entries found in original seqmap for '{attr_name}', skipping.")
            continue
        
        with open(temp_seqmap_file_path, 'w') as f:
            f.writelines(lines_to_write)
        print(f"  - INFO: Generated temporary seqmap file for attribute '{attr_name}'.")
        
        




# ==============================================================================
# --- 4. Main Program Entry ---
# ==============================================================================

if __name__ == '__main__':
    # First execute data preprocessing
    success = preprocess_data_by_attribute()
    
    # If preprocessing succeeds, execute evaluation
    if success:
        run_evaluation_by_attribute()
    
    print("\nAll workflows completed.")