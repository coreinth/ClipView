import os
import csv
import re
import sys
import argparse
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Popular open-source LLMs for title generation
AVAILABLE_MODELS = {
    # Fully open models (no gating/auth required)
    "phi3": "microsoft/Phi-3.5-mini-instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct", 
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

class LLMTitleGenerator:
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct"):
        """Initialize the LLM for title generation"""
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu"
        )
        print("Model loaded successfully!")
    
    def generate_title(self, transcript: str, max_words: int = 6) -> str:
        """Generate a chapter title from transcript text"""
        if not transcript or len(transcript.strip()) < 10:
            return "Chapter"
        
        # Truncate very long transcripts
        transcript = transcript[:800] if len(transcript) > 800 else transcript
        
        prompt = f"""Create a concise, engaging chapter title for this video segment:

Transcript: "{transcript}"

Requirements:
- Maximum {max_words} words
- Descriptive and clear
- Captures main topic
- Professional tone

Title:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=25,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract title from response
        if "Title:" in response:
            title = response.split("Title:")[-1].strip()
        else:
            # Fallback: take the generated part after the prompt
            title = response[len(prompt):].strip()
        
        # Clean up the title
        title = title.split('\n')[0].strip()  # Take first line only
        title = re.sub(r'^["\'-]+|["\'-]+$', '', title)  # Remove quotes
        
        # Ensure word limit
        words = title.split()[:max_words]
        final_title = " ".join(words)
        
        return final_title if final_title else "Chapter"

def parse_time_to_seconds(time_str: str) -> float:
    """Convert time string (MM:SS or H:MM:SS) to seconds"""
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
    """Parse a segment file and return list of (start_time, end_time, title) tuples"""
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

def load_bert_scores(csv_file: str) -> List[Dict]:
    """Load BERT scores from CSV file"""
    bert_scores = []
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bert_scores.append({
                    'score': float(row['score']),
                    'start': float(row['start']),
                    'end': float(row['end']),
                    'transcript': row['transcript']
                })
    except FileNotFoundError:
        print(f"Error: BERT score file not found: {csv_file}")
        return []
    except Exception as e:
        print(f"Error loading BERT scores: {e}")
        return []
    
    return bert_scores

def find_best_transcript_for_segment(segment_start: float, segment_end: float, 
                                   bert_scores: List[Dict]) -> str:
    """Find the best transcript snippet for a given segment"""
    segment_center = (segment_start + segment_end) / 2
    segment_duration = segment_end - segment_start
    
    # Find all BERT entries that overlap with this segment
    overlapping_entries = []
    
    for entry in bert_scores:
        bert_start = entry['start']
        bert_end = entry['end']
        
        # Check for overlap
        if (bert_start <= segment_end and bert_end >= segment_start):
            # Calculate overlap amount
            overlap_start = max(bert_start, segment_start)
            overlap_end = min(bert_end, segment_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            overlapping_entries.append({
                'entry': entry,
                'overlap_duration': overlap_duration,
                'score': entry['score']
            })
    
    if not overlapping_entries:
        return ""
    
    # Sort by overlap duration and score
    overlapping_entries.sort(key=lambda x: (x['overlap_duration'], x['score']), reverse=True)
    
    # Combine transcripts from top overlapping entries
    combined_transcript = ""
    for item in overlapping_entries[:3]:  # Take top 3 overlapping entries
        transcript = item['entry']['transcript'].strip()
        if transcript and transcript not in combined_transcript:
            combined_transcript += " " + transcript
    
    return combined_transcript.strip()

def generate_llm_titles_for_segments(segments_dir: str = "clipview_segments", 
                                   bert_scores_dir: str = "bert_score_files",
                                   output_dir: str = "llm_titled_segments",
                                   model_name: str = "microsoft/Phi-3.5-mini-instruct"):
    """Generate LLM titles for all segment files"""
    
    # Create output directory with model-specific subfolder
    model_suffix = model_name.split('/')[-1].replace('-', '_')
    full_output_dir = os.path.join(output_dir, model_suffix)
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Initialize LLM
    print("Initializing LLM for title generation...")
    llm = LLMTitleGenerator(model_name)
    
    # Get all segment files
    segment_files = [f for f in os.listdir(segments_dir) if f.endswith('.txt')]
    
    for segment_file in segment_files:
        print(f"\nProcessing: {segment_file}")
        
        # Load segments
        segment_path = os.path.join(segments_dir, segment_file)
        segments = parse_segment_file(segment_path)
        
        if not segments:
            print(f"No segments found in {segment_file}")
            continue
        
        # Find corresponding BERT score file (exact name match)
        base_name = segment_file.replace('.txt', '')
        bert_filename = base_name + '.csv'
        bert_file = os.path.join(bert_scores_dir, bert_filename)
        
        if not os.path.exists(bert_file):
            print(f"Warning: BERT score file not found: {bert_filename}")
            # Copy original file if no BERT data
            output_path = os.path.join(output_dir, segment_file)
            with open(segment_path, 'r', encoding='utf-8') as src, \
                 open(output_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
            continue
        
        # Load BERT scores
        bert_scores = load_bert_scores(bert_file)
        if not bert_scores:
            print(f"No BERT scores loaded from {bert_file}")
            continue
        
        # Generate new titles for each segment
        updated_segments = []
        
        for i, (start_time, end_time, original_title) in enumerate(segments):
            print(f"  Segment {i+1}/{len(segments)}: {start_time:.0f}-{end_time:.0f}s")
            
            # Find best transcript for this segment
            transcript = find_best_transcript_for_segment(start_time, end_time, bert_scores)
            
            if transcript:
                # Generate LLM title
                llm_title = llm.generate_title(transcript)
                print(f"    Original: {original_title}")
                print(f"    LLM: {llm_title}")
            else:
                llm_title = original_title
                print(f"    No transcript found, keeping original: {original_title}")
            
            updated_segments.append((start_time, end_time, llm_title))
        
        # Save updated segments
        output_path = os.path.join(full_output_dir, segment_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            for start_time, end_time, title in updated_segments:
                start_str = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
                end_str = f"{int(end_time//60):02d}:{int(end_time%60):02d}"
                f.write(f"{start_str} -> {end_str} {title}\n")
        
        print(f"  Saved: {output_path}")
    
    print(f"\nAll files processed! Check the '{full_output_dir}' directory for results.")

def compare_models(model_list: List[str] = None):
    """Compare multiple models on the same segments"""
    if model_list is None:
        # Use smaller models for comparison by default
        model_list = ["phi3", "qwen2.5-1.5b", "gemma2-2b"]
    
    print("Model Comparison Mode")
    print("=" * 50)
    
    # Check for gated models and warn user
    gated_models = ["llama3.2-1b", "llama3.2-3b", "mistral-7b", "codellama-7b"]
    has_gated = any(model in gated_models for model in model_list)
    
    if has_gated:
        print("\n⚠️  WARNING: Some models require HuggingFace authentication!")
        print("   Gated models in your list:", [m for m in model_list if m in gated_models])
        print("   Consider using these freely available alternatives:")
        print("   - Instead of mistral-7b: try qwen2.5-7b")
        print("   - Instead of llama3.2-3b: try qwen2.5-3b or phi3")
        print("   - Instead of codellama-7b: try qwen2.5-7b")
        print()
    
    successful_models = []
    failed_models = []
    
    for model_key in model_list:
        if model_key in AVAILABLE_MODELS:
            model_name = AVAILABLE_MODELS[model_key]
            print(f"\n🚀 Testing model: {model_key} ({model_name})")
            try:
                generate_llm_titles_for_segments(model_name=model_name)
                print(f"✅ Completed: {model_key}")
                successful_models.append(model_key)
            except Exception as e:
                error_msg = str(e)
                if "gated repo" in error_msg or "401 Client Error" in error_msg:
                    print(f"🔒 Gated model: {model_key} - Requires HuggingFace authentication")
                    print(f"   Visit: https://huggingface.co/{model_name}")
                    print(f"   Then run: huggingface-cli login")
                else:
                    print(f"❌ Failed: {model_key} - {error_msg}")
                failed_models.append(model_key)
        else:
            print(f"❌ Unknown model: {model_key}")
            failed_models.append(model_key)
    
    print(f"\n" + "="*50)
    print(f"📊 COMPARISON SUMMARY")
    print(f"✅ Successful: {successful_models}")
    print(f"❌ Failed: {failed_models}")
    if successful_models:
        print(f"\n📁 Results saved in: llm_titled_segments/")
        for model in successful_models:
            model_folder = AVAILABLE_MODELS[model].split('/')[-1].replace('-', '_')
            print(f"   - {model_folder}/")

def list_available_models():
    """List all available models"""
    print("Available Models:")
    print("=" * 50)
    for key, model_name in AVAILABLE_MODELS.items():
        print(f"  {key:15} -> {model_name}")
    print("\nUsage examples:")
    print("  python llm_title_generator.py --model phi3")
    print("  python llm_title_generator.py --compare phi3 qwen2.5-1.5b gemma2-2b")
    print("  python llm_title_generator.py --list")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LLM Title Generator for Video Segments")
    parser.add_argument("--model", "-m", type=str, default="phi3", 
                       help="Model to use (e.g., phi3, qwen2.5-1.5b, gemma2-2b)")
    parser.add_argument("--compare", "-c", nargs="+", 
                       help="Compare multiple models (e.g., --compare phi3 qwen2.5-1.5b)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List available models")
    
    args = parser.parse_args()
    
    print("LLM Title Generator for Video Segments")
    print("=" * 50)
    
    if args.list:
        list_available_models()
        return
    
    # Check if required directories exist
    if not os.path.exists("clipview_segments"):
        print("Error: 'clipview_segments' directory not found!")
        return
    
    if not os.path.exists("bert_score_files"):
        print("Error: 'bert_score_files' directory not found!")
        return
    
    if args.compare:
        compare_models(args.compare)
    else:
        # Single model mode
        model_key = args.model
        if model_key in AVAILABLE_MODELS:
            model_name = AVAILABLE_MODELS[model_key]
            print(f"Using model: {model_key} ({model_name})")
            generate_llm_titles_for_segments(model_name=model_name)
        else:
            print(f"Error: Unknown model '{model_key}'")
            print("Use --list to see available models")

if __name__ == "__main__":
    main()