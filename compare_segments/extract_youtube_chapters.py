import yt_dlp
import json
import os

# =============== CONFIGURATION ===============
videos = [
    {"title": "Why Pantone Colors Are So Expensive | So Expensive | Business Insider",
     "url": "https://www.youtube.com/watch?v=xnpyTNK4U9U",
     "filename": "pantone_colors"},
    {"title": "What Happens After Capitalism?",
     "url": "https://www.youtube.com/watch?v=IBbUdTcWD50",
     "filename": "capitalism"},
    {"title": "Is breakfast really the most important meal of the day? - The Food Chain podcast, BBC World Service",
     "url": "https://www.youtube.com/watch?v=YHITn1yAH_M",
     "filename": "breakfast_importance"},
    {"title": "How much water should I drink a day? - The Food Chain podcast, BBC World Service",
     "url": "https://www.youtube.com/watch?v=KEHyOrjYVk0",
     "filename": "water_intake"},
    {"title": "These Numbers Can Make AI Dangerous [Subliminal Learning]",
     "url": "https://www.youtube.com/watch?v=NUAb6zHXqdI",
     "filename": "ai_dangerous"},
    {"title": "The F=ma of Artificial Intelligence [Backpropagation, How Models Learn Part 2]",
     "url": "https://www.youtube.com/watch?v=VkHfRKewkWw",
     "filename": "ai_backpropagation"},
    {"title": "Why Deep Learning Works Unreasonably Well [How Models Learn Part 3]",
     "url": "https://www.youtube.com/watch?v=qx7hirqgfuU",
     "filename": "deep_learning_works"},
    {"title": "The Map of Quantum Computing - Quantum Computing Explained",
     "url": "https://www.youtube.com/watch?v=-UlxHPIEVqA",
     "filename": "quantum_computing_map"},
    {"title": "The Map of Engineering",
     "url": "https://www.youtube.com/watch?v=pQgxiQAMTTo",
     "filename": "engineering_map"},
    {"title": "The Fascinating Map of Fungi",
     "url": "https://www.youtube.com/watch?v=5FqFg-rjzPo",
     "filename": "fungi_map"},
    {"title": "These are the asteroids to worry about",
     "url": "https://www.youtube.com/watch?v=4Wrc4fHSCpw",
     "filename": "asteroids_to_worry"},
    {"title": "The Man Who Accidentally Killed The Most People In History",
     "url": "https://www.youtube.com/watch?v=IV3dnLzthDA",
     "filename": "most_people_killed"},
    {"title": "Reinforcement Learning with Neural Networks: Mathematical Details",
     "url": "https://www.youtube.com/watch?v=DVGmsnxB2UQ",
     "filename": "rl_math_details"},
    {"title": "Reinforcement Learning with Neural Networks: Essential Concepts",
     "url": "https://www.youtube.com/watch?v=9hbQieQh7-o",
     "filename": "rl_essential_concepts"},
    {"title": "Why Airport Food Is So Expensive",
     "url": "https://www.youtube.com/watch?v=TTCDLykCk5I",
     "filename": "airport_food"},
    {"title": "What's the curse of the Schwarz lantern?",
     "url": "https://www.youtube.com/watch?v=yAEveAH2KwI",
     "filename": "schwarz_lantern"},
    {"title": "Differential Equations: The Language of Change",
     "url": "https://www.youtube.com/watch?v=vTTlzmCRwU4",
     "filename": "differential_equations"},
    {"title": "Every Major Human Mistake That Changed History Forever",
     "url": "https://www.youtube.com/watch?v=41i2XIV8mCI",
     "filename": "human_mistakes"},
    {"title": "But how does bitcoin actually work?",
     "url": "https://www.youtube.com/watch?v=bBC-nXj3Ng4",
     "filename": "bitcoin_explained"},
    {"title": "LeetCode Was Hard Until I Learned THESE 8 Patterns (With Templates!)",
     "url": "https://www.youtube.com/watch?v=RYT08CaYq6A",
     "filename": "leetcode_patterns"},
    {"title": "Linux File System Structure Explained: From / to /usr | Linux Basics",
     "url": "https://www.youtube.com/watch?v=ISJ44S5sZu8",
     "filename": "linux_file_system"}
]

creator_output_dir = "../creator_segments"
clipview_output_dir = "../clipview_segments"

os.makedirs(creator_output_dir, exist_ok=True)
os.makedirs(clipview_output_dir, exist_ok=True)

# =============== HELPERS ===============

def format_time(seconds):
    """Convert seconds to H:MM:SS or M:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def process_json_segments(json_path, output_path, video_duration):
    """Convert your segment JSON into formatted text chapters."""
    with open(json_path, "r") as f:
        segments = json.load(f)

    lines = []
    for i, seg in enumerate(segments):
        start_time = format_time(seg["start"])
        if i + 1 < len(segments):
            end_time = format_time(segments[i + 1]["start"])
        else:
            end_time = format_time(video_duration)

        chapter_title = f"{seg['name'].capitalize()}"
        if i == 0:
            chapter_title = "Intro"

        lines.append(f"{start_time} -> {end_time} {chapter_title}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ ClipView segments saved to {output_path}")

# =============== MAIN ===============

ydl_opts = {'quiet': True, 'skip_download': True, 'extract_flat': False}

for vid in videos:
    print(f"\n🎬 Processing: {vid['title']}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(vid["url"], download=False)
            chapters = info.get("chapters", [])
            duration = info.get("duration", 0)
            video_filename = vid["filename"]

        # ---- 1️⃣ Save YouTube creator chapters ----
        if chapters:
            creator_path = os.path.join(creator_output_dir, f"{video_filename}.txt")
            with open(creator_path, "w", encoding="utf-8") as f:
                for i, ch in enumerate(chapters):
                    start = format_time(ch["start_time"])
                    end = format_time(chapters[i + 1]["start_time"]) if i + 1 < len(chapters) else format_time(duration)
                    f.write(f"{start} -> {end} {ch['title']}\n")
            print(f"✅ Creator chapters saved to {creator_path}")
        else:
            print("⚠️  No creator chapters found.")

        # ---- 2️⃣ Process your own ClipView JSON if exists ----
        json_path = os.path.join("../segmentation_scripts/batch_results", f"{video_filename}_final_chapters.json")
        print(f"Looking for ClipView JSON at: {json_path}")
        if os.path.exists(json_path):
            output_path = os.path.join(clipview_output_dir, f"{video_filename}.txt")
            process_json_segments(json_path, output_path, duration)
        else:
            print(f"⚠️  No ClipView JSON found for {video_filename}")

    except Exception as e:
        print(f"❌ Error processing {vid['filename']}: {e}")