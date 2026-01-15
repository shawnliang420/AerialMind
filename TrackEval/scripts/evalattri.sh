#!/bin/bash

# --- Configuration ---
# The root directory where your processed attribute folders (Day, Night, etc.) are located.
PROCESSED_ROOT="./processed_attribute_results"

# The root directory of your image dataset.
GT_DATA_ROOT="/home/data/RMOT/UAV-RMOT/rmot_train/training/image_02"

# --- Script Logic ---
echo "Starting batch evaluation for all attributes..."

# Check if the processed data directory exists
if [ ! -d "$PROCESSED_ROOT" ]; then
    echo "Error: Processed data directory not found at '$PROCESSED_ROOT'"
    exit 1
fi

# Loop through each subdirectory in the processed_attribute_results folder
for ATTR_DIR in "$PROCESSED_ROOT"/*; do
    if [ -d "$ATTR_DIR" ]; then
        ATTR_NAME=$(basename "$ATTR_DIR")
        SEQMAP_FILE="$ATTR_DIR/seqmap_temp.txt"

        # Check if the temporary seqmap file exists
        if [ ! -f "$SEQMAP_FILE" ]; then
            echo "Warning: No seqmap_temp.txt found for attribute '$ATTR_NAME'. Skipping."
            continue
        fi

        echo "------------------------------------------------------------"
        echo ">>> Running evaluation for attribute: $ATTR_NAME"
        echo "------------------------------------------------------------"

        # Construct and execute the evaluation command for the current attribute
        python3 run_mot_challenge_my.py \
            --METRICS HOTA \
            --SEQMAP_FILE "$SEQMAP_FILE" \
            --SKIP_SPLIT_FOL True \
            --GT_FOLDER "$GT_DATA_ROOT" \
            --TRACKERS_FOLDER "$PROCESSED_ROOT/$ATTR_NAME" \
            --GT_LOC_FORMAT "{gt_folder}/{video_id}/{expression_id}/gt.txt" \
            --TRACKERS_TO_EVAL "$PROCESSED_ROOT/$ATTR_NAME" \
            --USE_PARALLEL True \
            --NUM_PARALLEL_CORES 2 \
            --PLOT_CURVES False

        # Check if the last command was successful
        if [ $? -ne 0 ]; then
            echo "!!! Error during evaluation for attribute: $ATTR_NAME"
        else
            echo ">>> Successfully completed evaluation for: $ATTR_NAME"
        fi
    fi
done

echo "============================================================"
echo "All attribute evaluations are complete."
echo "============================================================"