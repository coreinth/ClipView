import json
import csv
from rake_nltk import Rake

bert_file = "bert_chapter_results3.csv"

def load_clip_data(clip_file="clip_chapter_results.json"):
    """Load CLIP visual analysis results"""
    with open(clip_file, 'r') as f:
        clip_data = json.load(f)
    return clip_data['chapters']

def load_bert_data(bert_file=bert_file):
    """Load BERT audio analysis results"""
    bert_scores = []
    with open(bert_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            bert_scores.append({
                'score': float(row['score']),
                'start': float(row['start']),
                'end': float(row['end']),
                'transcript': row['transcript']
            })
    return bert_scores

def mmss_to_seconds(mmss):
    """Convert MM:SS to seconds"""
    minutes, seconds = map(int, mmss.split(':'))
    return minutes * 60 + seconds

def seconds_to_mmss(seconds):
    """Convert seconds to MM:SS format"""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"

def combine_clip_bert_scores(clip_chapters, bert_scores, time_tolerance=10):
    """Combine CLIP visual scores with BERT audio scores"""
    combined_scores = []
    
    for clip_chapter in clip_chapters:
        clip_time = mmss_to_seconds(clip_chapter['timestamp_mmss'])
        
        best_bert_score = 0
        best_bert_match = None
        
        for bert_entry in bert_scores:
            bert_start = bert_entry['start']
            bert_end = bert_entry['end']
            bert_center = (bert_start + bert_end) / 2
            
            if (bert_start <= clip_time <= bert_end or 
                abs(clip_time - bert_center) <= time_tolerance):
                
                if bert_entry['score'] > best_bert_score:
                    best_bert_score = bert_entry['score']
                    best_bert_match = bert_entry
        
        combined_score = {
            'filename': clip_chapter['filename'],
            'timestamp_mmss': clip_chapter['timestamp_mmss'],
            'timestamp_seconds': clip_time,
            'visual_change': clip_chapter['visual_change'],
            'similarity': clip_chapter['similarity'],
            'bert_score': best_bert_score,
            'bert_transcript': best_bert_match['transcript'] if best_bert_match else '',
            'combined_score': (
                clip_chapter['visual_change'] + 
                clip_chapter['similarity'] + 
                best_bert_score * 2
            )
        }
        
        combined_scores.append(combined_score)
    
    return combined_scores

def filter_chapters(chapters, min_gap_seconds=60, max_chapters=10):
    """Filter chapters by distributing them throughout the video"""
    if not chapters:
        return []
    
    sorted_chapters = sorted(chapters, key=lambda x: x['timestamp_seconds'])
    
    if len(sorted_chapters) <= max_chapters:
        filtered = []
        last_time = -min_gap_seconds
        
        for chapter in sorted_chapters:
            current_time = chapter['timestamp_seconds']
            if current_time - last_time >= min_gap_seconds:
                filtered.append(chapter)
                last_time = current_time
        
        return filtered
    
    video_duration = sorted_chapters[-1]['timestamp_seconds']
    segment_duration = video_duration / max_chapters
    
    distributed_chapters = []
    
    for i in range(max_chapters):
        segment_start = i * segment_duration
        segment_end = (i + 1) * segment_duration
        
        segment_chapters = [
            ch for ch in sorted_chapters 
            if segment_start <= ch['timestamp_seconds'] < segment_end
        ]
        
        if not segment_chapters:
            segment_center = segment_start + segment_duration / 2
            closest_chapter = min(
                sorted_chapters,
                key=lambda x: abs(x['timestamp_seconds'] - segment_center)
            )
            if closest_chapter not in distributed_chapters:
                distributed_chapters.append(closest_chapter)
        else:
            best_chapter = max(segment_chapters, key=lambda x: x['combined_score'])
            if best_chapter not in distributed_chapters:
                distributed_chapters.append(best_chapter)
    
    distributed_chapters.sort(key=lambda x: x['timestamp_seconds'])
        
    filtered = []
    last_time = -min_gap_seconds
    
    for chapter in distributed_chapters:
        current_time = chapter['timestamp_seconds']
        if current_time - last_time >= min_gap_seconds:
            filtered.append(chapter)
            last_time = current_time
    
    return filtered[:max_chapters]

def generate_chapter_title(bert_transcript, max_words=8, method="rake"):
    """ Generate chapter title using various methods """
    
    if not bert_transcript:
        return "Chapter"
    
    if method == "llm_local":
        return generate_chapter_title_llm_local(bert_transcript, max_words)
    elif method == "llm_cloud":
        return generate_chapter_title_llm_cloud(bert_transcript, max_words)
    else:  # Default: RAKE
        return generate_chapter_title_rake(bert_transcript, max_words)

def generate_chapter_title_rake(bert_transcript, max_words=8):
    """Original RAKE-based title generation"""
    r = Rake()
    r.extract_keywords_from_text(bert_transcript)
    phrases = r.get_ranked_phrases()
    
    if phrases:
        return " ".join(phrases[0].split()[:max_words])
    
    return " ".join(bert_transcript.strip().split()[:max_words])

def generate_chapter_title_llm_local(bert_transcript, max_words=8):
    """Generate title using local LLM (Ollama)"""
    try:
        import ollama
        
        prompt = f"""Create a {max_words}-word chapter title for this video segment:
        
        "{bert_transcript[:500]}"
        
        Make it descriptive and engaging. No quotes in response."""
        
        response = ollama.chat(model='llama3.2:3b', messages=[
            {'role': 'user', 'content': prompt}
        ])
        
        title = response['message']['content'].strip().replace('"', '')
        words = title.split()[:max_words]
        return " ".join(words)
        
    except Exception as e:
        print(f"LLM local failed: {e}, falling back to RAKE")
        return generate_chapter_title_rake(bert_transcript, max_words)

def generate_chapter_title_llm_cloud(bert_transcript, max_words=8):
    """Generate title using cloud LLM (OpenAI)"""
    try:
        import openai
        
        client = openai.OpenAI(api_key="your-api-key-here")
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user", 
                "content": f"Create a {max_words}-word chapter title for: {bert_transcript[:300]}"
            }],
            max_tokens=20,
            temperature=0.3
        )
        
        title = response.choices[0].message.content.strip().replace('"', '')
        words = title.split()[:max_words]
        return " ".join(words)
        
    except Exception as e:
        print(f"LLM cloud failed: {e}, falling back to RAKE")
        return generate_chapter_title_rake(bert_transcript, max_words)

def create_final_chapters():
    """Main function to create final chapter list"""
    
    print("Loading CLIP and BERT data...")
    clip_chapters = load_clip_data()
    bert_scores = load_bert_data()
    
    print("Combining CLIP and BERT scores...")
    combined_chapters = combine_clip_bert_scores(clip_chapters, bert_scores)
    combined_chapters.sort(key=lambda x: x['combined_score'], reverse=True)
    
    print("Filtering chapters...")
    final_chapters = filter_chapters(combined_chapters)
    final_chapters.sort(key=lambda x: x['timestamp_seconds'])
    
    youtube_chapters = []
    for i, chapter in enumerate(final_chapters):
        title = generate_chapter_title(chapter['bert_transcript'])
        youtube_chapters.append({
            "name": title,
            "start": int(chapter['timestamp_seconds'])
        })
    
    with open("RENAME_THIS.json", "w") as f:
        json.dump(youtube_chapters, f, indent=2)
    
    print(f"Created {len(youtube_chapters)} final chapters")
    print("Results saved file: RENAME_THIS.json")
    
    return youtube_chapters