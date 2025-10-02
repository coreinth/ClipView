import os
import csv
import re
from typing import List, Dict, Tuple

# Lightweight alternative using sentence-transformers + simple generation
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class SimpleTitleGenerator:
    """Lightweight title generator using keyword extraction + templates"""
    
    def __init__(self):
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
    
    def extract_key_phrases(self, text: str, max_phrases: int = 3) -> List[str]:
        """Extract key phrases from text"""
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Remove stopwords
        meaningful_words = [w for w in words if w not in self.stopwords and len(w) > 2]
        
        # Create bigrams and trigrams
        phrases = []
        
        # Single important words
        word_freq = {}
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get most frequent meaningful words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        phrases.extend([word for word, freq in top_words])
        
        # Bigrams
        for i in range(len(meaningful_words) - 1):
            bigram = f"{meaningful_words[i]} {meaningful_words[i+1]}"
            phrases.append(bigram)
        
        return phrases[:max_phrases]
    
    def generate_title(self, transcript: str, max_words: int = 6) -> str:
        """Generate title from transcript"""
        if not transcript or len(transcript.strip()) < 10:
            return "Chapter"
        
        # Extract key phrases
        key_phrases = self.extract_key_phrases(transcript)
        
        if not key_phrases:
            # Fallback: first few meaningful words
            words = transcript.split()[:max_words]
            return " ".join(words)
        
        # Create title from top phrases
        title_parts = []
        word_count = 0
        
        for phrase in key_phrases:
            phrase_words = phrase.split()
            if word_count + len(phrase_words) <= max_words:
                title_parts.append(phrase.title())
                word_count += len(phrase_words)
            else:
                break
        
        title = " ".join(title_parts)
        
        # Ensure it's not too long
        if len(title.split()) > max_words:
            title = " ".join(title.split()[:max_words])
        
        return title if title else "Chapter"

def parse_time_to_seconds(time_str: str) -> float:
    """Convert time string to seconds"""
    time_str = time_str.strip()
    parts = time_str.split(':')
    
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        return float(time_str)

def parse_segment_file(file_path: str) -> List[Tuple[float, float, str]]:
    """Parse segment file"""
    segments = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                match = re.match(r'(\d+:\d+(?:\.\d+)?)\s*->\s*(\d+:\d+(?:\.\d+)?)\s*(.+)', line)
                if match:
                    start_str, end_str, title = match.groups()
                    start_time = parse_time_to_seconds(start_str)
                    end_time = parse_time_to_seconds(end_str)
                    segments.append((start_time, end_time, title.strip()))
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []
    
    return segments

def load_bert_scores(csv_file: str) -> List[Dict]:
    """Load BERT scores from CSV"""
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
    except Exception as e:
        print(f"Error loading BERT scores: {e}")
        return []
    
    return bert_scores

def find_transcript_for_segment(segment_start: float, segment_end: float, 
                              bert_scores: List[Dict]) -> str:
    """Find transcript for segment"""
    overlapping_transcripts = []
    
    for entry in bert_scores:
        if (entry['start'] <= segment_end and entry['end'] >= segment_start):
            overlap_start = max(entry['start'], segment_start)
            overlap_end = min(entry['end'], segment_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            
            if overlap_duration > 0:
                overlapping_transcripts.append({
                    'transcript': entry['transcript'],
                    'overlap': overlap_duration,
                    'score': entry['score']
                })
    
    if not overlapping_transcripts:
        return ""
    
    # Sort by overlap and score
    overlapping_transcripts.sort(key=lambda x: (x['overlap'], x['score']), reverse=True)
    
    # Combine top transcripts
    combined = " ".join([t['transcript'] for t in overlapping_transcripts[:2]])
    return combined

def quick_title_generation():
    """Quick title generation without heavy LLM"""
    
    segments_dir = "clipview_segments"
    bert_dir = "bert_score_files"
    output_dir = "improved_titles"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize simple title generator
    title_gen = SimpleTitleGenerator()
    
    # Process each segment file
    segment_files = [f for f in os.listdir(segments_dir) if f.endswith('.txt')]
    
    for segment_file in segment_files:
        print(f"Processing: {segment_file}")
        
        # Load segments
        segments = parse_segment_file(os.path.join(segments_dir, segment_file))
        if not segments:
            continue
        
        # Find BERT file (exact name match)
        base_name = segment_file.replace('.txt', '')
        bert_filename = base_name + '.csv'
        bert_file = os.path.join(bert_dir, bert_filename)
        
        if not os.path.exists(bert_file):
            print(f"  BERT file not found: {bert_filename}")
            continue
        
        # Load BERT scores
        bert_scores = load_bert_scores(bert_file)
        if not bert_scores:
            continue
        
        # Generate new titles
        updated_segments = []
        
        for start_time, end_time, original_title in segments:
            transcript = find_transcript_for_segment(start_time, end_time, bert_scores)
            
            if transcript:
                new_title = title_gen.generate_title(transcript)
                print(f"  {start_time:.0f}s: {original_title} → {new_title}")
            else:
                new_title = original_title
                print(f"  {start_time:.0f}s: {original_title} (no change)")
            
            updated_segments.append((start_time, end_time, new_title))
        
        # Save results
        output_path = os.path.join(output_dir, segment_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            for start_time, end_time, title in updated_segments:
                start_str = f"{int(start_time//60):02d}:{int(start_time%60):02d}"
                end_str = f"{int(end_time//60):02d}:{int(end_time%60):02d}"
                f.write(f"{start_str} -> {end_str} {title}\n")
        
        print(f"  Saved: {output_path}")

if __name__ == "__main__":
    print("Quick Title Generator (No LLM Required)")
    print("=" * 40)
    quick_title_generation()
    print("Done! Check 'improved_titles' directory.")