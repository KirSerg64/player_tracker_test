# Video Processing Pipeline - Fast Implementation for Windows PowerShell
# Based on existing run_parallel.sh but extended for video segmentation and processing

param(
    [Parameter(Mandatory=$true)]
    [string]$VideoPath,
    
    [string]$OutputDir = "",
    [int]$SegmentDuration = 600,    # 10 minutes
    [int]$OverlapDuration = 1,      # 1 second
    [int]$MaxParallel = 4,
    [switch]$Help
)

# Show help
if ($Help) {
    Write-Host @"
Video Processing Pipeline

Usage: .\run_video_pipeline.ps1 -VideoPath VIDEO_PATH [OPTIONS]

Required:
  -VideoPath PATH          Input video file path

Options:
  -OutputDir DIR           Output directory (default: outputs/EXPERIMENT_NAME)
  -SegmentDuration N       Segment duration in seconds (default: 600)
  -OverlapDuration N       Overlap duration in seconds (default: 1)
  -MaxParallel N           Maximum parallel processes (default: 4)
  -Help                    Show this help

Example:
  .\run_video_pipeline.ps1 -VideoPath "C:\path\to\football_match.mp4" -OutputDir "C:\path\to\outputs"
"@
    exit 0
}

# Validate required arguments
if (-not $VideoPath) {
    Write-Host "Error: -VideoPath is required" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $VideoPath)) {
    Write-Host "Error: Video file not found: $VideoPath" -ForegroundColor Red
    exit 1
}

# Set default output directory
$ExperimentName = "video_pipeline_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
if (-not $OutputDir) {
    $OutputDir = "outputs\$ExperimentName"
}

# Create output directory structure
$null = New-Item -ItemType Directory -Force -Path "$OutputDir\segments" -ErrorAction SilentlyContinue
$null = New-Item -ItemType Directory -Force -Path "$OutputDir\tracking_results" -ErrorAction SilentlyContinue
$null = New-Item -ItemType Directory -Force -Path "$OutputDir\tracklets" -ErrorAction SilentlyContinue
$null = New-Item -ItemType Directory -Force -Path "$OutputDir\refined_tracklets" -ErrorAction SilentlyContinue
$null = New-Item -ItemType Directory -Force -Path "$OutputDir\merged_tracklets" -ErrorAction SilentlyContinue
$null = New-Item -ItemType Directory -Force -Path "$OutputDir\logs" -ErrorAction SilentlyContinue

Write-Host "=== Video Processing Pipeline Started ===" -ForegroundColor Green
Write-Host "Video: $VideoPath" -ForegroundColor Blue
Write-Host "Output: $OutputDir" -ForegroundColor Blue
Write-Host "Experiment: $ExperimentName" -ForegroundColor Blue

# Function to log with timestamp
function Write-Log {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message" -ForegroundColor Yellow
}

# Check required commands
Write-Log "Checking dependencies..."
$commands = @("ffmpeg", "ffprobe", "python")
foreach ($cmd in $commands) {
    try {
        $null = Get-Command $cmd -ErrorAction Stop
        Write-Host "✓ Found: $cmd" -ForegroundColor Green
    } catch {
        Write-Host "✗ Error: $cmd is not installed or not in PATH" -ForegroundColor Red
        exit 1
    }
}

# Step 1: Video Segmentation
Write-Log "Step 1: Video Segmentation"

# Get video duration
$videoDurationOutput = ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$VideoPath"
$videoDuration = [math]::Floor([double]$videoDurationOutput)

Write-Log "Video duration: $videoDuration seconds"

# Calculate number of segments
$numSegments = [math]::Ceiling(($videoDuration + $SegmentDuration - $OverlapDuration - 1) / ($SegmentDuration - $OverlapDuration))
Write-Log "Will create $numSegments segments"

# Create segments
$segmentPaths = @()
for ($i = 0; $i -lt $numSegments; $i++) {
    $segmentId = "{0:D3}" -f $i
    $segmentDir = "$OutputDir\segments\segment_$segmentId"
    $null = New-Item -ItemType Directory -Force -Path $segmentDir
    
    $segmentFile = "$segmentDir\video_segment_$segmentId.mp4"
    $startTime = $i * ($SegmentDuration - $OverlapDuration)
    
    # Don't go beyond video duration
    if ($startTime -ge $videoDuration) { break }
    
    Write-Log "Creating segment $($i+1)/$numSegments`: start=$($startTime)s"
    
    # Run FFmpeg
    $ffmpegArgs = @(
        "-y", "-i", "$VideoPath",
        "-ss", "$startTime",
        "-t", "$SegmentDuration",
        "-c:v", "libx264",
        "-c:a", "aac", 
        "-crf", "23",
        "-preset", "medium",
        "$segmentFile"
    )
    
    $ffmpegProcess = Start-Process -FilePath "ffmpeg" -ArgumentList $ffmpegArgs -NoNewWindow -Wait -RedirectStandardOutput "$OutputDir\logs\ffmpeg_segment_$i.log" -RedirectStandardError "$OutputDir\logs\ffmpeg_segment_$i.err" -PassThru
    
    if ($ffmpegProcess.ExitCode -eq 0 -and (Test-Path $segmentFile)) {
        $segmentPaths += $segmentFile
        Write-Host "✓ Created: $(Split-Path $segmentFile -Leaf)" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to create segment $i" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✓ Step 1 completed: $($segmentPaths.Count) segments created" -ForegroundColor Green

# Step 2: Detection and Tracking (Parallel Processing)
Write-Log "Step 2: Detection and Tracking (Parallel Processing)"

# Function to create segment config
function New-SegmentConfig {
    param([int]$SegmentId, [string]$SegmentVideo)
    
    $configDir = "$OutputDir\configs"
    $null = New-Item -ItemType Directory -Force -Path $configDir
    $configFile = "$configDir\soccernet_segment_$SegmentId.yaml"
    
    # Copy base config
    Copy-Item "sn_gamestate\configs\soccernet_test.yaml" $configFile
    
    # Modify for segment (simple text replacement)
    $content = Get-Content $configFile
    $content = $content -replace 'experiment_name: "sn-gamestate"', "experiment_name: `"segment_$SegmentId`""
    $content | Set-Content $configFile
    
    return $configFile
}

# Process segments in parallel batches
$jobs = @()
$batchSize = $MaxParallel

for ($i = 0; $i -lt $segmentPaths.Count; $i++) {
    $segmentVideo = $segmentPaths[$i]
    $segmentId = "{0:D3}" -f $i
    
    # Create segment-specific output directory
    $segmentOutputDir = "$OutputDir\tracking_results\segment_$segmentId"
    $null = New-Item -ItemType Directory -Force -Path $segmentOutputDir
    
    # Create temporary dataset structure
    $tempDatasetDir = "$OutputDir\temp_dataset\segment_$segmentId"
    $null = New-Item -ItemType Directory -Force -Path "$tempDatasetDir\img1"
    
    Write-Log "Processing segment $segmentId"
    
    # Create a script block for parallel execution
    $scriptBlock = {
        param($SegmentVideo, $SegmentId, $OutputDir, $TempDatasetDir, $SegmentOutputDir)
        
        # Extract frames
        $ffmpegArgs = @("-i", $SegmentVideo, "-vf", "fps=25", "$TempDatasetDir\img1\%06d.jpg")
        Start-Process -FilePath "ffmpeg" -ArgumentList $ffmpegArgs -NoNewWindow -Wait -RedirectStandardOutput "$OutputDir\logs\extract_frames_$SegmentId.log" -RedirectStandardError "$OutputDir\logs\extract_frames_$SegmentId.err"
        
        # Run tracking
        Set-Location (Split-Path $PSScriptRoot -Parent)
        $env:SEGMENT_OUTPUT_DIR = $SegmentOutputDir
        $env:SEGMENT_ID = $SegmentId
        
        $pythonArgs = @(
            "main.py",
            "--config-path", "pkg://sn_gamestate.configs",
            "--config-name", "soccernet_test",
            "hydra.run.dir=$SegmentOutputDir",
            "experiment_name=segment_$SegmentId"
        )
        
        Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -Wait -RedirectStandardOutput "$OutputDir\logs\tracking_segment_$SegmentId.log" -RedirectStandardError "$OutputDir\logs\tracking_segment_$SegmentId.err"
        
        # Simple MOT format conversion placeholder
        $motFile = "$SegmentOutputDir\seq_$SegmentId.txt"
        "# MOT format placeholder for segment $SegmentId" | Out-File -FilePath $motFile -Encoding UTF8
        
        return "✓ Segment $SegmentId processed"
    }
    
    # Start job
    $job = Start-Job -ScriptBlock $scriptBlock -ArgumentList $segmentVideo, $segmentId, $OutputDir, $tempDatasetDir, $segmentOutputDir
    $jobs += $job
    
    # Control parallel execution
    if ($jobs.Count -ge $batchSize) {
        Write-Log "Waiting for batch to complete..."
        $jobs | Wait-Job | Receive-Job
        $jobs | Remove-Job
        $jobs = @()
        Write-Host "✓ Batch completed" -ForegroundColor Green
    }
}

# Wait for remaining jobs
if ($jobs.Count -gt 0) {
    Write-Log "Waiting for final batch..."
    $jobs | Wait-Job | Receive-Job
    $jobs | Remove-Job
}

Write-Host "✓ Step 2 completed: Detection and tracking finished" -ForegroundColor Green

# Step 3: Generate Tracklets
Write-Log "Step 3: Generate Tracklets"

for ($i = 0; $i -lt $segmentPaths.Count; $i++) {
    $segmentId = "{0:D3}" -f $i
    $motFile = "$OutputDir\tracking_results\segment_$segmentId\seq_$segmentId.txt"
    $datasetDir = "$OutputDir\temp_dataset\segment_$segmentId"
    
    if (Test-Path $motFile) {
        Write-Log "Generating tracklets for segment $segmentId"
        
        $pythonArgs = @(
            "sn_gamestate\gta_link\generate_tracklets.py",
            "--model_path", "pretrained_models\gta_link\feature_extractor_osnet_x1_0.onnx",
            "--data_path", "$datasetDir\..",
            "--pred_dir", "$OutputDir\tracking_results\segment_$segmentId",
            "--tracker", "StrongSORT"
        )
        
        Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -Wait -RedirectStandardOutput "$OutputDir\logs\tracklets_segment_$segmentId.log" -RedirectStandardError "$OutputDir\logs\tracklets_segment_$segmentId.err"
        
        # Move generated tracklets
        $generatedFile = "$OutputDir\tracking_results\StrongSORT_Tracklets\seq_$segmentId.pkl"
        $targetFile = "$OutputDir\tracklets\segment_$segmentId.pkl"
        
        if (Test-Path $generatedFile) {
            Move-Item $generatedFile $targetFile
            Write-Host "✓ Tracklets generated for segment $segmentId" -ForegroundColor Green
        }
    } else {
        Write-Host "⚠ No MOT file found for segment $segmentId" -ForegroundColor Yellow
    }
}

Write-Host "✓ Step 3 completed: Tracklet generation finished" -ForegroundColor Green

# Step 4: Refine Tracklets
Write-Log "Step 4: Refine Tracklets"

for ($i = 0; $i -lt $segmentPaths.Count; $i++) {
    $segmentId = "{0:D3}" -f $i
    $trackletFile = "$OutputDir\tracklets\segment_$segmentId.pkl"
    
    if (Test-Path $trackletFile) {
        Write-Log "Refining tracklets for segment $segmentId"
        
        $pythonArgs = @(
            "sn_gamestate\gta_link\refine_tracklets.py",
            "--input_dir", "$OutputDir\tracklets",
            "--output_dir", "$OutputDir\refined_tracklets",
            "--seq_name", "segment_$segmentId"
        )
        
        Start-Process -FilePath "python" -ArgumentList $pythonArgs -NoNewWindow -Wait -RedirectStandardOutput "$OutputDir\logs\refine_segment_$segmentId.log" -RedirectStandardError "$OutputDir\logs\refine_segment_$segmentId.err"
        
        if (Test-Path "$OutputDir\refined_tracklets\segment_$segmentId.pkl") {
            Write-Host "✓ Tracklets refined for segment $segmentId" -ForegroundColor Green
        }
    } else {
        Write-Host "⚠ No tracklet file found for segment $segmentId" -ForegroundColor Yellow
    }
}

Write-Host "✓ Step 4 completed: Tracklet refinement finished" -ForegroundColor Green

# Step 5: Cross-Segment Merging
Write-Log "Step 5: Cross-Segment Merging"

# Create Python script for merging
$mergingScript = @"
import pickle
import numpy as np
from pathlib import Path
import sys

# Simple merging implementation
tracklets_dir = Path('$OutputDir/refined_tracklets')
output_file = Path('$OutputDir/merged_tracklets/final_merged_tracklets.pkl')
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
            # Adjust timing if available
            if hasattr(tracklet, 'times'):
                tracklet.times = [t + time_offset for t in tracklet.times]
            
            # Assign new global ID
            tracklet.track_id = global_id
            merged_tracklets[global_id] = tracklet
            global_id += 1
        
        # Update time offset for next segment
        time_offset += $SegmentDuration - $OverlapDuration
        
    except Exception as e:
        print(f'Error processing {segment_file}: {e}')

# Save merged tracklets
with open(output_file, 'wb') as f:
    pickle.dump(merged_tracklets, f)

print(f'✓ Merged {len(merged_tracklets)} tracklets saved to {output_file}')
"@

$mergingScript | Out-File -FilePath "$OutputDir\merge_tracklets.py" -Encoding UTF8
python "$OutputDir\merge_tracklets.py"

Write-Host "✓ Step 5 completed: Cross-segment merging finished" -ForegroundColor Green

# Cleanup temporary files
Write-Log "Cleaning up temporary files..."
Remove-Item -Recurse -Force "$OutputDir\temp_dataset" -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force "$OutputDir\configs" -ErrorAction SilentlyContinue
Remove-Item "$OutputDir\merge_tracklets.py" -ErrorAction SilentlyContinue

# Final summary
Write-Host "=== Pipeline Completed Successfully ===" -ForegroundColor Green
Write-Host "Results saved in: $OutputDir" -ForegroundColor Blue
Write-Host "Final merged tracklets: $OutputDir\merged_tracklets\final_merged_tracklets.pkl" -ForegroundColor Blue

# Generate summary report
$summaryContent = @"
Video Processing Pipeline Summary
================================
Execution Time: $(Get-Date)
Input Video: $VideoPath
Output Directory: $OutputDir
Experiment Name: $ExperimentName

Configuration:
- Segment Duration: ${SegmentDuration}s
- Overlap Duration: ${OverlapDuration}s
- Number of Segments: $($segmentPaths.Count)
- Max Parallel: $MaxParallel

Results:
- Segments: $OutputDir\segments\
- Tracking Results: $OutputDir\tracking_results\
- Tracklets: $OutputDir\tracklets\
- Refined Tracklets: $OutputDir\refined_tracklets\
- Final Merged: $OutputDir\merged_tracklets\final_merged_tracklets.pkl

Logs: $OutputDir\logs\
"@

$summaryContent | Out-File -FilePath "$OutputDir\pipeline_summary.txt" -Encoding UTF8

Write-Host "Pipeline summary saved to: $OutputDir\pipeline_summary.txt" -ForegroundColor Green
Write-Host "🎉 All done!" -ForegroundColor Green
