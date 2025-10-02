import json
import os
from pathlib import Path
from segmentation_scripts.torch_check import torch_check
from segmentation_scripts.llava_main import llava_main
from segmentation_scripts.clip_main import chapter_detection
from segmentation_scripts.score_fusion import create_final_chapters

def seconds_to_mmss(seconds):
    """Convert seconds to MM:SS format"""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"

def to_python_type(obj):
    import numpy as np
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    return obj

def process_single_video(video_path, output_prefix=None):
    """
    Process a single video through the complete pipeline
    
    Args:
        video_path (str): Path to the video file
        output_prefix (str): Prefix for output files (defaults to video filename without extension)
    
    Returns:
        dict: Results containing final chapters and metadata
    """
    
    # Generate output prefix from video filename if not provided
    if output_prefix is None:
        output_prefix = Path(video_path).stem
    
    print(f"\n{'='*60}")
    print(f"Processing video: {video_path}")
    print(f"Output prefix: {output_prefix}")
    print(f"{'='*60}")
    
    # Define output filenames with prefix
    clip_results_file = f"{output_prefix}_clip_chapter_results.json"
    bert_results_file = f"../bert_score_files/{output_prefix}_bert_chapter_results3.csv"
    final_results_file = f"{output_prefix}_final_chapters.json"
    
    try:
        # Step 1: Run LLaVA analysis
        print("\\n🎬 Step 1: Running LLaVA scene analysis...")
        llava_results = llava_main(video_path, output_prefix)
        print("✅ LLaVA analysis completed")
        
        # Step 2: Run CLIP chapter detection
        print("\\n🔍 Step 2: Running CLIP chapter detection...")
        frame_map_file = f"segments/{output_prefix}_frame_map.csv"
        desc_file = f"{output_prefix}_scene_descriptions.txt"
        
        chapters, ranked_chapters = chapter_detection(
            video_path=video_path,
            frame_map_file=frame_map_file, 
            desc_file=desc_file
        )
        
        # Save CLIP results
        with open(clip_results_file, "w", encoding="utf-8") as f:
            json.dump({
                "video_path": video_path,
                "output_prefix": output_prefix,
                "chapters": to_python_type(chapters),
                "ranked_chapters": to_python_type(ranked_chapters),
            }, f, indent=2)
        
        print(f"✅ Chapter detection completed. Results saved to {clip_results_file}")
        
        # Step 3: Create final chapters using score fusion
        print("\\n⚡ Step 3: Running score fusion...")
        final_chapters = create_final_chapters(
            clip_file=clip_results_file,
            bert_file=bert_results_file,
            output_file=final_results_file
        )
        
        print("✅ Score fusion completed")
        
        # Display results
        print(f"\\n🎯 FINAL RESULTS FOR {output_prefix}:")
        print(f"{'='*50}")
        print("YouTube Chapters:")
        for i, chapter in enumerate(final_chapters, 1):
            print(f"{i:2d}. {seconds_to_mmss(chapter['start'])} - {chapter['name']}")
        
        return {
            "video_path": video_path,
            "output_prefix": output_prefix,
            "final_chapters": final_chapters,
            "clip_results_file": clip_results_file,
            "bert_results_file": bert_results_file,
            "final_results_file": final_results_file,
            "success": True
        }
        
    except Exception as e:
        print(f"❌ Error processing {video_path}: {str(e)}")
        return {
            "video_path": video_path,
            "output_prefix": output_prefix,
            "error": str(e),
            "success": False
        }

def process_multiple_videos(video_list, output_dir="results"):
    """
    Process multiple videos sequentially
    
    Args:
        video_list (list): List of video file paths or dict with video_path and output_prefix
        output_dir (str): Directory to save results
    
    Returns:
        list: Results for each video processed
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Change to output directory for processing
    original_cwd = os.getcwd()
    os.chdir(output_dir)
    
    all_results = []
    
    try:
        print(f"🚀 Starting batch processing of {len(video_list)} videos")
        print(f"📁 Working directory: {os.path.abspath(output_dir)}")
        
        for i, video_item in enumerate(video_list, 1):
            print(f"\\n{'='*80}")
            print(f"PROCESSING VIDEO {i}/{len(video_list)}")
            print(f"{'='*80}")
            
            # Handle different input formats
            if isinstance(video_item, str):
                video_path = video_item
                output_prefix = None
            elif isinstance(video_item, dict):
                video_path = video_item['video_path']
                output_prefix = video_item.get('output_prefix')
            else:
                print(f"❌ Invalid video item format: {video_item}")
                continue
            
            # Convert to absolute path if relative
            if not os.path.isabs(video_path):
                video_path = os.path.join(original_cwd, video_path)
            
            # Check if video file exists
            if not os.path.exists(video_path):
                print(f"❌ Video file not found: {video_path}")
                all_results.append({
                    "video_path": video_path,
                    "error": "File not found",
                    "success": False
                })
                continue
            
            # Process the video
            result = process_single_video(video_path, output_prefix)
            all_results.append(result)
            
            # Save intermediate batch results
            batch_results_file = "batch_processing_results.json"
            with open(batch_results_file, "w", encoding="utf-8") as f:
                json.dump({
                    "total_videos": len(video_list),
                    "processed_count": i,
                    "results": all_results
                }, f, indent=2)
        
        # Final summary
        successful = [r for r in all_results if r.get('success', False)]
        failed = [r for r in all_results if not r.get('success', False)]
        
        print(f"\\n{'='*80}")
        print("🎉 BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Successfully processed: {len(successful)}/{len(video_list)} videos")
        
        if failed:
            print(f"❌ Failed videos:")
            for fail in failed:
                print(f"   - {fail['video_path']}: {fail.get('error', 'Unknown error')}")
        
        print(f"\\n📄 Detailed results saved to: {os.path.abspath(batch_results_file)}")
        
    finally:
        # Return to original directory
        os.chdir(original_cwd)
    
    return all_results

def get_videos_from_directory(videos_dir="../videos/", extensions=('.mp4', '.avi', '.mov', '.mkv')):
    """
    Automatically discover video files from a directory
    
    Args:
        videos_dir (str): Path to directory containing videos
        extensions (tuple): Video file extensions to look for
    
    Returns:
        list: List of video file paths found
    """
    videos_path = Path(videos_dir)
    
    if not videos_path.exists():
        print(f"❌ Videos directory not found: {videos_path.absolute()}")
        return []
    
    video_files = []
    for ext in extensions:
        video_files.extend(videos_path.glob(f"*{ext}"))
        video_files.extend(videos_path.glob(f"*{ext.upper()}"))  # Include uppercase extensions
    
    # Convert to string paths and sort
    video_paths = [str(video) for video in sorted(video_files)]
    
    if video_paths:
        print(f"📹 Found {len(video_paths)} video files in {videos_path.absolute()}:")
        for video in video_paths:
            print(f"   - {Path(video).name}")
    else:
        print(f"❌ No video files found in {videos_path.absolute()}")
        print(f"   Looking for extensions: {extensions}")
    
    return video_paths

def main():
    """
    Main function with support for single or multiple video processing
    """
    
    # Configuration
    SINGLE_VIDEO_MODE = True  # Set to False for batch processing
    VIDEOS_DIRECTORY = "../videos/"  # Change this to your videos directory
    
    if SINGLE_VIDEO_MODE:
        # Single video processing (original behavior)
        video_path = "../videos/30min_vid.mp4"  # Change this to your video
        result = process_single_video(video_path)
        
        if result['success']:
            print("\\n🎯 Processing completed successfully!")
        else:
            print(f"\\n❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    else:
        # Multiple video processing - automatically discover videos
        video_list = get_videos_from_directory(VIDEOS_DIRECTORY)
        
        if not video_list:
            print("\\n❌ No videos found for batch processing!")
            print(f"   Please add video files to: {Path(VIDEOS_DIRECTORY).absolute()}")
            return
        
        print(f"\\n🚀 Starting batch processing of {len(video_list)} videos...")
        
        # Process all videos
        results = process_multiple_videos(video_list, output_dir="batch_results")
        
        # Summary
        successful_count = sum(1 for r in results if r.get('success', False))
        print(f"\\n🎉 Batch processing complete: {successful_count}/{len(video_list)} videos processed successfully")

if __name__ == "__main__":
    main()