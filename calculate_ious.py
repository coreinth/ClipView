import os
import re
from typing import List, Tuple, Dict
import numpy as np

def parse_time_to_seconds(time_str: str) -> float:
    """Convert time string (MM:SS or H:MM:SS) to seconds."""
    time_str = time_str.strip()
    parts = time_str.split(':')
    
    if len(parts) == 2:  # MM:SS
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:  # H:MM:SS
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        return float(time_str)

def parse_segment_file(file_path: str) -> List[Tuple[float, float, str]]:
    """Parse a segment file and return list of (start_time, end_time, title) tuples."""
    segments = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse format: "start_time -> end_time title"
                match = re.match(r'(\d+:\d+(?:\.\d+)?)\s*->\s*(\d+:\d+(?:\.\d+)?)\s*(.+)', line)
                if match:
                    start_str, end_str, title = match.groups()
                    start_time = parse_time_to_seconds(start_str)
                    end_time = parse_time_to_seconds(end_str)
                    segments.append((start_time, end_time, title.strip()))
    
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []
    
    return segments

def calculate_iou(seg1: Tuple[float, float], seg2: Tuple[float, float]) -> float:
    """Calculate IoU (Intersection over Union) between two time segments."""
    start1, end1 = seg1
    start2, end2 = seg2
    
    # Calculate intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    
    # Calculate union
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start
    
    return intersection / union if union > 0 else 0

def find_best_match(target_seg: Tuple[float, float, str], 
                   candidate_segs: List[Tuple[float, float, str]]) -> Tuple[int, float]:
    """Find the best matching segment based on IoU score."""
    best_idx = -1
    best_iou = 0
    
    target_time = (target_seg[0], target_seg[1])
    
    for i, candidate_seg in enumerate(candidate_segs):
        candidate_time = (candidate_seg[0], candidate_seg[1])
        iou = calculate_iou(target_time, candidate_time)
        
        if iou > best_iou:
            best_iou = iou
            best_idx = i
    
    return best_idx, best_iou

def calculate_segment_accuracy(creator_segs: List[Tuple[float, float, str]], 
                             clipview_segs: List[Tuple[float, float, str]], 
                             iou_threshold: float = 0.5) -> Dict:
    """Calculate accuracy metrics between creator and clipview segments."""
    
    print(f"Creator segments: {len(creator_segs)}")
    print(f"ClipView segments: {len(clipview_segs)}")
    print(f"IoU threshold: {iou_threshold}")
    print()
    
    matched_pairs = []
    unmatched_creator = []
    used_clipview = set()
    
    # For each creator segment, find best match in clipview
    for i, creator_seg in enumerate(creator_segs):
        best_idx, best_iou = find_best_match(creator_seg, clipview_segs)
        
        print(f"Creator {i+1}: {creator_seg[0]:.0f}-{creator_seg[1]:.0f}s '{creator_seg[2]}'")
        
        if best_iou >= iou_threshold and best_idx not in used_clipview:
            matched_pairs.append((i, best_idx, best_iou))
            used_clipview.add(best_idx)
            clipview_seg = clipview_segs[best_idx]
            print(f"  → MATCHED with ClipView {best_idx+1}: IoU={best_iou:.3f}")
            print(f"    ClipView: {clipview_seg[0]:.0f}-{clipview_seg[1]:.0f}s '{clipview_seg[2]}'")
        else:
            unmatched_creator.append(i)
            if best_idx >= 0:
                clipview_seg = clipview_segs[best_idx]
                print(f"  → No match (best IoU={best_iou:.3f} < {iou_threshold})")
                print(f"    Best candidate: {clipview_seg[0]:.0f}-{clipview_seg[1]:.0f}s '{clipview_seg[2]}'")
            else:
                print(f"  → No match found")
        print()
    
    # Identify unmatched clipview segments
    unmatched_clipview = [i for i in range(len(clipview_segs)) if i not in used_clipview]
    
    # Calculate metrics
    true_positives = len(matched_pairs)
    false_positives = len(unmatched_clipview)  # ClipView segments with no creator match
    false_negatives = len(unmatched_creator)   # Creator segments with no clipview match
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Average IoU of matched segments
    avg_iou = np.mean([iou for _, _, iou in matched_pairs]) if matched_pairs else 0
    
    return {
        'total_creator_segments': len(creator_segs),
        'total_clipview_segments': len(clipview_segs),
        'matched_segments': true_positives,
        'unmatched_creator': false_negatives,
        'unmatched_clipview': false_positives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'average_iou': avg_iou,
        'matched_pairs': matched_pairs,
        'unmatched_creator_indices': unmatched_creator,
        'unmatched_clipview_indices': unmatched_clipview
    }

def compare_files(creator_file: str, clipview_file: str, iou_threshold: float = 0.5):
    """Compare two specific segment files."""
    
    print(f"Comparing: {creator_file} vs {clipview_file}")
    print("=" * 80)
    
    creator_segs = parse_segment_file(creator_file)
    clipview_segs = parse_segment_file(clipview_file)
    
    if not creator_segs:
        print(f"No segments found in creator file: {creator_file}")
        return None
    if not clipview_segs:
        print(f"No segments found in clipview file: {clipview_file}")
        return None
    
    metrics = calculate_segment_accuracy(creator_segs, clipview_segs, iou_threshold)
    
    # Print results
    print("RESULTS:")
    print("-" * 40)
    print(f"Matched segments: {metrics['matched_segments']}")
    print(f"Unmatched creator segments: {metrics['unmatched_creator']}")
    print(f"Unmatched clipview segments: {metrics['unmatched_clipview']}")
    print()
    print(f"Precision: {metrics['precision']:.3f} ({metrics['matched_segments']}/{metrics['matched_segments'] + metrics['unmatched_clipview']})")
    print(f"Recall: {metrics['recall']:.3f} ({metrics['matched_segments']}/{metrics['matched_segments'] + metrics['unmatched_creator']})")
    print(f"F1-Score: {metrics['f1_score']:.3f}")
    print(f"Average IoU: {metrics['average_iou']:.3f}")
    
    if metrics['unmatched_clipview_indices']:
        print(f"\nUnmatched ClipView segments:")
        for idx in metrics['unmatched_clipview_indices']:
            seg = clipview_segs[idx]
            print(f"  ClipView {idx+1}: {seg[0]:.0f}-{seg[1]:.0f}s '{seg[2]}'")
    
    return metrics

def compare_all_files(creator_dir: str = "creator_segments", 
                     clipview_dir: str = "clipview_segments", 
                     iou_threshold: float = 0.3):
    """Compare all matching files between creator and clipview directories."""
    
    # Get list of files in both directories
    try:
        creator_files = os.listdir(creator_dir)
        clipview_files = os.listdir(clipview_dir)
    except FileNotFoundError as e:
        print(f"Directory not found: {e}")
        return
    
    # Find matching files (case-insensitive)
    matching_files = []
    for creator_file in creator_files:
        for clipview_file in clipview_files:
            if creator_file.lower() == clipview_file.lower():
                matching_files.append((creator_file, clipview_file))
                break
    
    if not matching_files:
        print("No matching files found!")
        return
    
    print(f"Found {len(matching_files)} matching files:")
    for cf, vf in matching_files:
        print(f"  {cf}")
    print()
    
    all_results = []
    
    for creator_file, clipview_file in matching_files:
        creator_path = os.path.join(creator_dir, creator_file)
        clipview_path = os.path.join(clipview_dir, clipview_file)
        
        print(f"\n{'='*80}")
        result = compare_files(creator_path, clipview_path, iou_threshold)
        
        if result:
            result['filename'] = creator_file
            all_results.append(result)
    
    # Overall summary
    if all_results:
        print(f"\n{'='*80}")
        print("OVERALL SUMMARY ACROSS ALL FILES")
        print(f"{'='*80}")
        
        total_creator = sum(r['total_creator_segments'] for r in all_results)
        total_clipview = sum(r['total_clipview_segments'] for r in all_results)
        total_matched = sum(r['matched_segments'] for r in all_results)
        total_unmatched_creator = sum(r['unmatched_creator'] for r in all_results)
        total_unmatched_clipview = sum(r['unmatched_clipview'] for r in all_results)
        
        overall_precision = total_matched / (total_matched + total_unmatched_clipview) if (total_matched + total_unmatched_clipview) > 0 else 0
        overall_recall = total_matched / (total_matched + total_unmatched_creator) if (total_matched + total_unmatched_creator) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        # Weighted average IoU
        total_iou_weight = sum(r['average_iou'] * r['matched_segments'] for r in all_results)
        overall_avg_iou = total_iou_weight / total_matched if total_matched > 0 else 0
        
        print(f"Files analyzed: {len(all_results)}")
        print(f"Total creator segments: {total_creator}")
        print(f"Total clipview segments: {total_clipview}")
        print(f"Total matched segments: {total_matched}")
        print()
        print(f"Overall Precision: {overall_precision:.3f} ({overall_precision:.1%})")
        print(f"Overall Recall: {overall_recall:.3f} ({overall_recall:.1%})")
        print(f"Overall F1-Score: {overall_f1:.3f}")
        print(f"Overall Average IoU: {overall_avg_iou:.3f}")
        
        print(f"\nPer-file breakdown:")
        print("-" * 60)
        for r in all_results:
            print(f"{r['filename']:<20} P:{r['precision']:.3f} R:{r['recall']:.3f} F1:{r['f1_score']:.3f} IoU:{r['average_iou']:.3f}")
    
    return all_results

def main():
    """Main function to run the comparison."""
    
    # Set IoU threshold (0.3 = 30% overlap required for match)
    iou_threshold = 0.3
    
    print("IoU and Accuracy Calculator for Video Segments")
    print("=" * 80)
    print("Choose an option:")
    print("1. Compare single file (NotPetya example)")
    print("2. Compare all files")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Single file comparison
        creator_file = "creator_segments/NotPetya.txt"
        clipview_file = "clipview_segments/NotPetya.txt"
        result = compare_files(creator_file, clipview_file, iou_threshold)
        
        if result:
            print(f"\n{'='*80}")
            print("SUMMARY")
            print(f"{'='*80}")
            print(f"The ClipView segments achieved:")
            print(f"  • {result['precision']:.1%} precision (how many detected segments were correct)")
            print(f"  • {result['recall']:.1%} recall (how many actual segments were detected)")
            print(f"  • {result['f1_score']:.3f} F1-score (overall accuracy measure)")
            print(f"  • {result['average_iou']:.3f} average IoU (temporal overlap quality)")
    
    elif choice == "2":
        # Compare all files
        compare_all_files(iou_threshold=iou_threshold)
    
    else:
        print("Invalid choice. Running single file comparison by default.")
        creator_file = "creator_segments/NotPetya.txt"
        clipview_file = "clipview_segments/NotPetya.txt"
        compare_files(creator_file, clipview_file, iou_threshold)

if __name__ == "__main__":
    main()
