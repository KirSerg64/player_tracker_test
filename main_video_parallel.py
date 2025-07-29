#!/usr/bin/env python3
"""
Main Video Parallel Processing Script

Combines video splitting with parallel processing using Hydra overrides.
Integrates video_splitter.py functionality with main.py processing pipeline.

Usage:
    python main_video_parallel.py input_video.mp4 ./output --config soccernet_test --workers 4
"""

import argparse
import concurrent.futures
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import shutil

import hydra
import torch
import rich.logging
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# Import tracklab components
from tracklab.datastruct import TrackerState
from tracklab.pipeline import Pipeline
from tracklab.utils import monkeypatch_hydra, progress, wandb
from tracklab.engine.video import VideoOnlineTrackingEngine

# Import local video splitter
from sn_gamestate.utils.video_splitter import VideoSplitter

# Setup environment
os.environ["HYDRA_FULL_ERROR"] = "1"
log = logging.getLogger(__name__)


class ParallelVideoProcessor:
    """
    Handles video splitting and parallel processing with Hydra config integration
    """
    
    def __init__(self, input_video: str, output_dir: str, base_config: str = "soccernet_test",
                 segment_duration: int = 600, overlap: float = 1.0, 
                 max_workers: int = 4, debug: bool = False):
        """
        Initialize the parallel video processor
        
        Args:
            input_video: Path to input video file
            output_dir: Base output directory for all results
            base_config: Base Hydra config name (e.g., 'soccernet_test')
            segment_duration: Duration of each segment in seconds
            overlap: Overlap between segments in seconds
            max_workers: Number of parallel workers
            debug: Enable debug logging
        """
        self.input_video = Path(input_video)
        self.output_dir = Path(output_dir)
        self.base_config = base_config
        self.segment_duration = segment_duration
        self.overlap = overlap
        self.max_workers = max_workers
        self.debug = debug
        
        # Create main output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup subdirectories
        self.segments_dir = self.output_dir / "segments"
        self.results_dir = self.output_dir / "tracking_results"
        self.logs_dir = self.output_dir / "logs"
        
        for directory in [self.segments_dir, self.results_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Validate input video
        if not self.input_video.exists():
            raise FileNotFoundError(f"Input video not found: {self.input_video}")
        
        # Setup logging
        self._setup_logging()
        
        log.info(f"Initialized ParallelVideoProcessor:")
        log.info(f"  Input video: {self.input_video}")
        log.info(f"  Output directory: {self.output_dir}")
        log.info(f"  Base config: {self.base_config}")
        log.info(f"  Segment duration: {self.segment_duration}s")
        log.info(f"  Overlap: {self.overlap}s")
        log.info(f"  Max workers: {self.max_workers}")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        level = logging.DEBUG if self.debug else logging.INFO
        
        # File handler for logs
        file_handler = logging.FileHandler(self.logs_dir / "parallel_processing.log")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            handlers=[file_handler, console_handler],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def split_video(self) -> List[Path]:
        """Split input video into overlapping segments"""
        log.info("Starting video segmentation...")
        
        # Create video splitter
        splitter = VideoSplitter(
            input_video=str(self.input_video),
            output_dir=str(self.segments_dir),
            segment_duration=self.segment_duration,
            overlap=self.overlap,
            debug=self.debug
        )
        
        # Split video
        segments = splitter.split_video()
        
        log.info(f"Video split completed. Created {len(segments)} segments:")
        for i, segment in enumerate(segments):
            log.info(f"  Segment {i+1}: {segment.name}")
        
        return segments
    
    def process_single_segment(self, segment_path: Path, segment_id: int) -> Dict[str, Any]:
        """
        Process a single video segment using main.py functionality with Hydra overrides
        
        Args:
            segment_path: Path to the video segment
            segment_id: Segment identifier
            
        Returns:
            Processing result dictionary
        """
        segment_name = segment_path.stem
        segment_output_dir = self.results_dir / f"segment_{segment_id:03d}"
        segment_output_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        try:
            log.info(f"Processing segment {segment_id+1}: {segment_name}")
            
            # Build Hydra command with overrides
            cmd = [
                sys.executable, 'main.py',
                '--config-path', 'sn_gamestate/configs',
                '--config-name', self.base_config,
                # Hydra overrides
                f'dataset.video_path={segment_path}',
                f'hydra.run.dir={segment_output_dir}',
                f'hydra.job.name=segment_{segment_id:03d}',
                # Disable wandb for parallel processing
                'use_wandb=false',
                # Ensure rich logging is disabled for parallel processing
                'use_rich=false'
            ]
            
            # Add debug flag if needed
            if self.debug:
                cmd.append('print_config=true')
            
            log.debug(f"Executing command: {' '.join(cmd)}")
            
            # Execute main.py with segment-specific config
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                cwd=Path.cwd(),  # Ensure we're in the correct working directory
                timeout=3600  # 1 hour timeout per segment
            )
            
            processing_time = time.time() - start_time
            
            # Log processing results
            if self.debug and result.stdout:
                log.debug(f"Segment {segment_id+1} stdout: {result.stdout[-500:]}")  # Last 500 chars
            
            if result.stderr and self.debug:
                log.debug(f"Segment {segment_id+1} stderr: {result.stderr[-500:]}")
            
            # Save processing logs
            log_file = segment_output_dir / "processing.log"
            with open(log_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n\n")
                f.write(f"STDOUT:\n{result.stdout}\n\n")
                f.write(f"STDERR:\n{result.stderr}\n")
            
            log.info(f"✓ Completed segment {segment_id+1} in {processing_time:.1f}s: {segment_name}")
            
            return {
                'segment_id': segment_id,
                'segment_path': str(segment_path),
                'segment_name': segment_name,
                'output_dir': str(segment_output_dir),
                'success': True,
                'processing_time': processing_time,
                'stdout_length': len(result.stdout),
                'stderr_length': len(result.stderr)
            }
            
        except subprocess.TimeoutExpired:
            processing_time = time.time() - start_time
            error_msg = f"Segment {segment_id+1} timed out after {processing_time:.1f}s"
            log.error(f"✗ {error_msg}: {segment_name}")
            
            return {
                'segment_id': segment_id,
                'segment_path': str(segment_path),
                'segment_name': segment_name,
                'output_dir': str(segment_output_dir),
                'success': False,
                'error': error_msg,
                'processing_time': processing_time
            }
            
        except subprocess.CalledProcessError as e:
            processing_time = time.time() - start_time
            error_msg = f"Segment processing failed with return code {e.returncode}"
            
            log.error(f"✗ Failed segment {segment_id+1} after {processing_time:.1f}s: {segment_name}")
            if self.debug:
                log.error(f"Error stdout: {e.stdout[-300:] if e.stdout else 'None'}")
                log.error(f"Error stderr: {e.stderr[-300:] if e.stderr else 'None'}")
            
            # Save error logs
            error_log_file = segment_output_dir / "error.log"
            with open(error_log_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n\n")
                f.write(f"Return code: {e.returncode}\n\n")
                f.write(f"STDOUT:\n{e.stdout}\n\n")
                f.write(f"STDERR:\n{e.stderr}\n")
            
            return {
                'segment_id': segment_id,
                'segment_path': str(segment_path),
                'segment_name': segment_name,
                'output_dir': str(segment_output_dir),
                'success': False,
                'error': error_msg,
                'processing_time': processing_time,
                'return_code': e.returncode,
                'stdout': e.stdout[-500:] if e.stdout else '',
                'stderr': e.stderr[-500:] if e.stderr else ''
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}"
            
            log.error(f"✗ Exception in segment {segment_id+1} after {processing_time:.1f}s: {segment_name}")
            log.error(f"Exception details: {error_msg}")
            
            return {
                'segment_id': segment_id,
                'segment_path': str(segment_path),
                'segment_name': segment_name,
                'output_dir': str(segment_output_dir),
                'success': False,
                'error': error_msg,
                'processing_time': processing_time
            }
    
    def process_segments_parallel(self, segments: List[Path]) -> List[Dict[str, Any]]:
        """
        Process video segments in parallel using ThreadPoolExecutor
        
        Args:
            segments: List of segment paths to process
            
        Returns:
            List of processing results
        """
        log.info(f"Starting parallel processing of {len(segments)} segments with {self.max_workers} workers")
        
        start_time = time.time()
        results = []
        
        # Use ThreadPoolExecutor for I/O bound subprocess calls
        # ProcessPoolExecutor would be better for CPU-bound tasks, but subprocess calls are I/O bound
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_segment = {
                executor.submit(self.process_single_segment, segment, i): (segment, i)
                for i, segment in enumerate(segments)
            }
            
            # Process completed tasks
            completed = 0
            for future in concurrent.futures.as_completed(future_to_segment):
                segment, segment_id = future_to_segment[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    if result['success']:
                        log.info(f"Progress: {completed}/{len(segments)} - ✓ {result['segment_name']}")
                    else:
                        log.error(f"Progress: {completed}/{len(segments)} - ✗ {result['segment_name']}: {result['error']}")
                        
                except Exception as e:
                    log.error(f"Future exception for segment {segment.name}: {e}")
                    results.append({
                        'segment_id': segment_id,
                        'segment_path': str(segment),
                        'segment_name': segment.stem,
                        'success': False,
                        'error': f"Future exception: {str(e)}",
                        'processing_time': 0
                    })
                    completed += 1
        
        total_time = time.time() - start_time
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        log.info(f"Parallel processing completed in {total_time:.1f}s:")
        log.info(f"  Successful: {len(successful)}/{len(segments)}")
        log.info(f"  Failed: {len(failed)}/{len(segments)}")
        log.info(f"  Success rate: {len(successful)/len(segments)*100:.1f}%")
        
        if failed:
            log.warning("Failed segments:")
            for failure in failed:
                log.warning(f"  - {failure['segment_name']}: {failure['error']}")
        
        return results
    
    def save_processing_summary(self, segments: List[Path], results: List[Dict[str, Any]]) -> Path:
        """Save processing summary to JSON file"""
        summary = {
            'input_video': str(self.input_video),
            'processing_config': {
                'base_config': self.base_config,
                'segment_duration': self.segment_duration,
                'overlap': self.overlap,
                'max_workers': self.max_workers
            },
            'segments': {
                'total_count': len(segments),
                'successful_count': len([r for r in results if r['success']]),
                'failed_count': len([r for r in results if not r['success']]),
                'success_rate': len([r for r in results if r['success']]) / len(results) if results else 0
            },
            'processing_times': {
                'total_time': sum(r.get('processing_time', 0) for r in results),
                'average_time': sum(r.get('processing_time', 0) for r in results) / len(results) if results else 0,
                'min_time': min((r.get('processing_time', 0) for r in results if r['success']), default=0),
                'max_time': max((r.get('processing_time', 0) for r in results if r['success']), default=0)
            },
            'detailed_results': results
        }
        
        summary_file = self.output_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        log.info(f"Processing summary saved to: {summary_file}")
        return summary_file
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete video splitting and parallel processing pipeline
        
        Returns:
            Dictionary containing processing results and summary
        """
        log.info("="*60)
        log.info("Starting Video Parallel Processing Pipeline")
        log.info("="*60)
        
        total_start_time = time.time()
        
        try:
            # Step 1: Split video into segments
            log.info("Step 1: Video Segmentation")
            segments = self.split_video()
            
            if not segments:
                raise RuntimeError("No segments were created from the input video")
            
            # Step 2: Process segments in parallel
            log.info("Step 2: Parallel Processing")
            results = self.process_segments_parallel(segments)
            
            # Step 3: Save summary
            log.info("Step 3: Saving Results Summary")
            summary_file = self.save_processing_summary(segments, results)
            
            total_time = time.time() - total_start_time
            
            # Final summary
            successful_count = len([r for r in results if r['success']])
            success_rate = successful_count / len(results) * 100 if results else 0
            
            log.info("="*60)
            log.info("Pipeline Completed Successfully!")
            log.info(f"Total execution time: {total_time:.1f}s")
            log.info(f"Segments processed: {len(segments)}")
            log.info(f"Success rate: {success_rate:.1f}% ({successful_count}/{len(results)})")
            log.info(f"Results saved in: {self.output_dir}")
            log.info("="*60)
            
            return {
                'success': True,
                'total_time': total_time,
                'segments_count': len(segments),
                'successful_count': successful_count,
                'failed_count': len(results) - successful_count,
                'success_rate': success_rate,
                'output_directory': str(self.output_dir),
                'summary_file': str(summary_file),
                'segments': [str(s) for s in segments],
                'results': results
            }
            
        except Exception as e:
            total_time = time.time() - total_start_time
            error_msg = f"Pipeline failed after {total_time:.1f}s: {str(e)}"
            
            log.error("="*60)
            log.error("Pipeline Failed!")
            log.error(error_msg)
            log.error("="*60)
            
            return {
                'success': False,
                'error': error_msg,
                'total_time': total_time,
                'output_directory': str(self.output_dir)
            }


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Parallel Video Processing with Tracklab',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main_video_parallel.py input.mp4 ./output
  
  # Custom configuration
  python main_video_parallel.py match.mp4 ./results --config soccernet_test --workers 6
  
  # Debug mode with custom segments
  python main_video_parallel.py video.mp4 ./output --segment_duration 300 --overlap 2.0 --debug
        """
    )
    
    # Required arguments
    parser.add_argument('input_video', type=str, help='Path to input video file')
    parser.add_argument('output_dir', type=str, help='Output directory for all results')
    
    # Optional arguments
    parser.add_argument('--config', type=str, default='soccernet_test',
                       help='Base Hydra config name (default: soccernet_test)')
    parser.add_argument('--segment_duration', type=int, default=600,
                       help='Segment duration in seconds (default: 600)')
    parser.add_argument('--overlap', type=float, default=1.0,
                       help='Overlap between segments in seconds (default: 1.0)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging and verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    input_video = Path(args.input_video)
    if not input_video.exists():
        print(f"Error: Input video not found: {input_video}")
        return 1
    
    if args.workers < 1:
        print("Error: Number of workers must be at least 1")
        return 1
    
    if args.segment_duration < 10:
        print("Error: Segment duration must be at least 10 seconds")
        return 1
    
    try:
        # Create and run processor
        processor = ParallelVideoProcessor(
            input_video=str(input_video),
            output_dir=args.output_dir,
            base_config=args.config,
            segment_duration=args.segment_duration,
            overlap=args.overlap,
            max_workers=args.workers,
            debug=args.debug
        )
        
        # Run the pipeline
        result = processor.run()
        
        # Print final summary
        if result['success']:
            print(f"\n🎉 Processing completed successfully!")
            print(f"📁 Results saved in: {result['output_directory']}")
            print(f"📊 Success rate: {result['success_rate']:.1f}%")
            print(f"⏱️  Total time: {result['total_time']:.1f}s")
            return 0
        else:
            print(f"\n❌ Processing failed: {result['error']}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
