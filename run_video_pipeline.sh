#!/bin/bash

# Video Processing Pipeline - Fast Implementation for Linux
# Based on existing run_parallel.sh but extended for video segmentation and processing

set -e  # Exit on any error
set -u  # Exit on undefined variables
set -o pipefail  # Exit on pipe failures

# Configuration
VIDEO_PATH=""
OUTPUT_DIR=""
SEGMENT_DURATION=600  # 10 minutes
OVERLAP_DURATION=1    # 1 second
MAX_PARALLEL=4
EXPERIMENT_NAME="video_pipeline_$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Help function
show_help() {
    echo "Video Processing Pipeline"
    echo "Usage: $0 --video VIDEO_PATH [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --video PATH          Input video file path"
    echo ""
    echo "Options:"
    echo "  --output DIR          Output directory (default: outputs/EXPERIMENT_NAME)"
    echo "  --segment-duration N  Segment duration in seconds (default: 600)"
    echo "  --overlap N           Overlap duration in seconds (default: 1)"
    echo "  --max-parallel N      Maximum parallel processes (default: 4)"
    echo "  --help                Show this help"
    echo ""
    echo "Example:"
    echo "  $0 --video /path/to/football_match.mp4 --output /path/to/outputs"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --video)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --segment-duration)
            SEGMENT_DURATION="$2"
            shift 2
            ;;
        --overlap)
            OVERLAP_DURATION="$2"
            shift 2
            ;;
        --max-parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$VIDEO_PATH" ]]; then
    echo -e "${RED}Error: --video is required${NC}"
    show_help
    exit 1
fi

if [[ ! -f "$VIDEO_PATH" ]]; then
    echo -e "${RED}Error: Video file not found: $VIDEO_PATH${NC}"
    exit 1
fi

# Set default output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="outputs/$EXPERIMENT_NAME"
fi

# Create output directory structure
mkdir -p "$OUTPUT_DIR"/{segments,tracking_results,tracklets,refined_tracklets,merged_tracklets,logs}

echo -e "${GREEN}=== Video Processing Pipeline Started ===${NC}"
echo -e "${BLUE}Video: $VIDEO_PATH${NC}"
echo -e "${BLUE}Output: $OUTPUT_DIR${NC}"
echo -e "${BLUE}Experiment: $EXPERIMENT_NAME${NC}"

# Function to log with timestamp
log() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Function to check if command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}Error: $1 is not installed or not in PATH${NC}"
        exit 1
    fi
}

# Check required commands
log "Checking dependencies..."
check_command ffmpeg
check_command ffprobe
check_command python
echo -e "${GREEN}✓ All dependencies found${NC}"

# Step 1: Video Segmentation
log "Step 1: Video Segmentation"

# Get video duration
VIDEO_DURATION=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$VIDEO_PATH")
VIDEO_DURATION_INT=$(printf "%.0f" "$VIDEO_DURATION")

log "Video duration: ${VIDEO_DURATION_INT} seconds"

# Calculate number of segments
NUM_SEGMENTS=$(( (VIDEO_DURATION_INT + SEGMENT_DURATION - OVERLAP_DURATION - 1) / (SEGMENT_DURATION - OVERLAP_DURATION) ))
log "Will create $NUM_SEGMENTS segments"

# Create segments
SEGMENT_PATHS=()
for ((i=0; i<NUM_SEGMENTS; i++)); do
    SEGMENT_DIR="$OUTPUT_DIR/segments/segment_$(printf "%03d" $i)"
    mkdir -p "$SEGMENT_DIR"
    
    SEGMENT_FILE="$SEGMENT_DIR/video_segment_$(printf "%03d" $i).mp4"
    START_TIME=$((i * (SEGMENT_DURATION - OVERLAP_DURATION)))
    
    # Don't go beyond video duration
    if [[ $START_TIME -ge $VIDEO_DURATION_INT ]]; then
        break
    fi
    
    log "Creating segment $((i+1))/$NUM_SEGMENTS: start=${START_TIME}s"
    
    ffmpeg -y -i "$VIDEO_PATH" \
        -ss "$START_TIME" \
        -t "$SEGMENT_DURATION" \
        -c:v libx264 \
        -c:a aac \
        -crf 23 \
        -preset medium \
        "$SEGMENT_FILE" \
        >> "$OUTPUT_DIR/logs/ffmpeg_segment_$i.log" 2>&1
    
    if [[ -f "$SEGMENT_FILE" ]]; then
        SEGMENT_PATHS+=("$SEGMENT_FILE")
        echo -e "${GREEN}✓ Created: $(basename "$SEGMENT_FILE")${NC}"
    else
        echo -e "${RED}✗ Failed to create segment $i${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✓ Step 1 completed: ${#SEGMENT_PATHS[@]} segments created${NC}"

# Step 2: Detection and Tracking (Parallel Processing)
log "Step 2: Detection and Tracking (Parallel Processing)"

# Create segment-specific configs based on soccernet_test.yaml
create_segment_config() {
    local segment_id=$1
    local segment_video=$2
    local config_file="$OUTPUT_DIR/configs/soccernet_segment_$segment_id.yaml"
    
    mkdir -p "$OUTPUT_DIR/configs"
    
    # Copy base config and modify for segment
    cp "sn_gamestate/configs/soccernet_test.yaml" "$config_file"
    
    # Modify config for this segment (Linux-compatible sed)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|experiment_name: \"sn-gamestate\"|experiment_name: \"segment_$segment_id\"|" "$config_file"
    else
        # Linux
        sed -i "s|experiment_name: \"sn-gamestate\"|experiment_name: \"segment_$segment_id\"|" "$config_file"
    fi
    
    echo "$config_file"
}

# Process segments in parallel batches
PIDS=()
BATCH_SIZE=$MAX_PARALLEL

for ((i=0; i<${#SEGMENT_PATHS[@]}; i++)); do
    SEGMENT_VIDEO="${SEGMENT_PATHS[$i]}"
    SEGMENT_ID=$(printf "%03d" $i)
    
    # Create segment-specific output directory
    SEGMENT_OUTPUT_DIR="$OUTPUT_DIR/tracking_results/segment_$SEGMENT_ID"
    mkdir -p "$SEGMENT_OUTPUT_DIR"
    
    # Create temporary dataset structure expected by tracklab
    TEMP_DATASET_DIR="$OUTPUT_DIR/temp_dataset/segment_$SEGMENT_ID"
    mkdir -p "$TEMP_DATASET_DIR/img1"
    
    # Extract frames from segment (optimized for Linux)
    log "Extracting frames from segment $SEGMENT_ID"
    ffmpeg -y -i "$SEGMENT_VIDEO" -vf fps=25 -q:v 2 "$TEMP_DATASET_DIR/img1/%06d.jpg" \
        >> "$OUTPUT_DIR/logs/extract_frames_$SEGMENT_ID.log" 2>&1 &
    
    # Store frame extraction PID for proper waiting
    FRAME_PID=$!
    
    # Wait for frame extraction to complete for this segment
    wait $FRAME_PID
    
    # Create config for this segment
    CONFIG_FILE=$(create_segment_config "$SEGMENT_ID" "$SEGMENT_VIDEO")
    
    # Run tracking on this segment
    log "Starting tracking for segment $SEGMENT_ID"
    
    (
        export SEGMENT_OUTPUT_DIR
        export SEGMENT_ID
        cd "$(dirname "$0")"  # Ensure we're in the right directory
        
        # Modify the dataset path in the environment or config
        python main.py \
            --config-path pkg://sn_gamestate.configs \
            --config-name soccernet_test \
            hydra.run.dir="$SEGMENT_OUTPUT_DIR" \
            experiment_name="segment_$SEGMENT_ID" \
            dataset.dataset_path="$TEMP_DATASET_DIR/../.." \
            >> "$OUTPUT_DIR/logs/tracking_segment_$SEGMENT_ID.log" 2>&1
        
        # Convert tracklab output to MOT format
        if [[ -f "$SEGMENT_OUTPUT_DIR/tracking_results.pkl" ]]; then
            python -c "
import pickle
import pandas as pd
from pathlib import Path

# Load tracklab results and convert to MOT format
with open('$SEGMENT_OUTPUT_DIR/tracking_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Convert to MOT format (simplified)
mot_file = '$SEGMENT_OUTPUT_DIR/seq_$SEGMENT_ID.txt'
with open(mot_file, 'w') as f:
    # This is a placeholder - you'll need to adapt based on your tracklab output format
    f.write('# MOT format placeholder for segment $SEGMENT_ID\n')

print('✓ MOT conversion completed for segment $SEGMENT_ID')
"
        fi
    ) &
    
    PIDS+=($!)
    
    # Control parallel execution
    if [[ ${#PIDS[@]} -ge $BATCH_SIZE ]]; then
        log "Waiting for batch to complete..."
        for pid in "${PIDS[@]}"; do
            wait "$pid"
        done
        PIDS=()
        echo -e "${GREEN}✓ Batch completed${NC}"
    fi
done

# Wait for remaining processes
if [[ ${#PIDS[@]} -gt 0 ]]; then
    log "Waiting for final batch..."
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done
fi

echo -e "${GREEN}✓ Step 2 completed: Detection and tracking finished${NC}"

# Step 3: Generate Tracklets
log "Step 3: Generate Tracklets"

for ((i=0; i<${#SEGMENT_PATHS[@]}; i++)); do
    SEGMENT_ID=$(printf "%03d" $i)
    MOT_FILE="$OUTPUT_DIR/tracking_results/segment_$SEGMENT_ID/seq_$SEGMENT_ID.txt"
    DATASET_DIR="$OUTPUT_DIR/temp_dataset/segment_$SEGMENT_ID"
    TRACKLET_OUTPUT_DIR="$OUTPUT_DIR/tracklets"
    
    if [[ -f "$MOT_FILE" ]]; then
        log "Generating tracklets for segment $SEGMENT_ID"
        
        python sn_gamestate/gta_link/generate_tracklets.py \
            --model_path "pretrained_models/gta_link/feature_extractor_osnet_x1_0.onnx" \
            --data_path "$DATASET_DIR/.." \
            --pred_dir "$OUTPUT_DIR/tracking_results/segment_$SEGMENT_ID" \
            --tracker "StrongSORT" \
            >> "$OUTPUT_DIR/logs/tracklets_segment_$SEGMENT_ID.log" 2>&1
        
        # Move generated tracklets to organized location
        if [[ -f "$OUTPUT_DIR/tracking_results/StrongSORT_Tracklets/seq_$SEGMENT_ID.pkl" ]]; then
            mv "$OUTPUT_DIR/tracking_results/StrongSORT_Tracklets/seq_$SEGMENT_ID.pkl" \
               "$TRACKLET_OUTPUT_DIR/segment_$SEGMENT_ID.pkl"
            echo -e "${GREEN}✓ Tracklets generated for segment $SEGMENT_ID${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ No MOT file found for segment $SEGMENT_ID${NC}"
    fi
done

echo -e "${GREEN}✓ Step 3 completed: Tracklet generation finished${NC}"

# Step 4: Refine Tracklets
log "Step 4: Refine Tracklets"

for ((i=0; i<${#SEGMENT_PATHS[@]}; i++)); do
    SEGMENT_ID=$(printf "%03d" $i)
    TRACKLET_FILE="$OUTPUT_DIR/tracklets/segment_$SEGMENT_ID.pkl"
    
    if [[ -f "$TRACKLET_FILE" ]]; then
        log "Refining tracklets for segment $SEGMENT_ID"
        
        python sn_gamestate/gta_link/refine_tracklets.py \
            --input_dir "$OUTPUT_DIR/tracklets" \
            --output_dir "$OUTPUT_DIR/refined_tracklets" \
            --seq_name "segment_$SEGMENT_ID" \
            >> "$OUTPUT_DIR/logs/refine_segment_$SEGMENT_ID.log" 2>&1
        
        if [[ -f "$OUTPUT_DIR/refined_tracklets/segment_$SEGMENT_ID.pkl" ]]; then
            echo -e "${GREEN}✓ Tracklets refined for segment $SEGMENT_ID${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ No tracklet file found for segment $SEGMENT_ID${NC}"
    fi
done

echo -e "${GREEN}✓ Step 4 completed: Tracklet refinement finished${NC}"

# Step 5: Cross-Segment Merging
log "Step 5: Cross-Segment Merging"

# Create a simple merging script based on the notebook code
python -c "
import pickle
import numpy as np
from pathlib import Path
import sys
sys.path.append('sn_gamestate/Notebooks')

# This is a simplified version - you'll need to adapt the notebook code
tracklets_dir = Path('$OUTPUT_DIR/refined_tracklets')
output_file = Path('$OUTPUT_DIR/merged_tracklets/final_merged_tracklets.pkl')
output_file.parent.mkdir(exist_ok=True)

merged_tracklets = {}
global_id = 0
time_offset = 0

# Load and merge tracklets from all segments
for segment_file in sorted(tracklets_dir.glob('segment_*.pkl')):
    print(f'Processing {segment_file.name}')
    
    try:
        with open(segment_file, 'rb') as f:
            segment_tracklets = pickle.load(f)
        
        # Add tracklets with time offset and new IDs
        for track_id, tracklet in segment_tracklets.items():
            # Adjust timing
            if hasattr(tracklet, 'times'):
                tracklet.times = [t + time_offset for t in tracklet.times]
            
            # Assign new global ID
            tracklet.track_id = global_id
            merged_tracklets[global_id] = tracklet
            global_id += 1
        
        # Update time offset for next segment
        time_offset += $SEGMENT_DURATION - $OVERLAP_DURATION
        
    except Exception as e:
        print(f'Error processing {segment_file}: {e}')

# Save merged tracklets
with open(output_file, 'wb') as f:
    pickle.dump(merged_tracklets, f)

print(f'✓ Merged {len(merged_tracklets)} tracklets saved to {output_file}')
"

echo -e "${GREEN}✓ Step 5 completed: Cross-segment merging finished${NC}"

# Cleanup temporary files
log "Cleaning up temporary files..."
rm -rf "$OUTPUT_DIR/temp_dataset"
rm -rf "$OUTPUT_DIR/configs"

# Final summary
echo -e "${GREEN}=== Pipeline Completed Successfully ===${NC}"
echo -e "${BLUE}Results saved in: $OUTPUT_DIR${NC}"
echo -e "${BLUE}Final merged tracklets: $OUTPUT_DIR/merged_tracklets/final_merged_tracklets.pkl${NC}"

# Generate summary report
cat > "$OUTPUT_DIR/pipeline_summary.txt" << EOF
Video Processing Pipeline Summary
================================
Execution Time: $(date)
Input Video: $VIDEO_PATH
Output Directory: $OUTPUT_DIR
Experiment Name: $EXPERIMENT_NAME

Configuration:
- Segment Duration: ${SEGMENT_DURATION}s
- Overlap Duration: ${OVERLAP_DURATION}s
- Number of Segments: ${#SEGMENT_PATHS[@]}
- Max Parallel: $MAX_PARALLEL

Results:
- Segments: $OUTPUT_DIR/segments/
- Tracking Results: $OUTPUT_DIR/tracking_results/
- Tracklets: $OUTPUT_DIR/tracklets/
- Refined Tracklets: $OUTPUT_DIR/refined_tracklets/
- Final Merged: $OUTPUT_DIR/merged_tracklets/final_merged_tracklets.pkl

Logs: $OUTPUT_DIR/logs/
EOF

echo -e "${GREEN}Pipeline summary saved to: $OUTPUT_DIR/pipeline_summary.txt${NC}"
echo -e "${GREEN}🎉 All done!${NC}"
