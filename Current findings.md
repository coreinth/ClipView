# Current Findings
## Strengths
- Works best with videos with clear topic transitions, i.e. audio cues pointing to the next topic, or drastic scene changes
  - Videos with categorical topics
  - Educational videos with phases/sections (i.e. general topic with subtopics)
- Good with medium-lengthed segments (around 2-3 minutes)
## Weaknesses
- Does not work as well with videos that have a lot of topics or rapid topic changes
  - Videos with many segments or rapid compilations
  - Streamlined documentary narrative (like telling one whole story)
- Videos with a lot of segments are not detected due to the threshold settings
- Narrative documentaries do not perform well because of the coherence in sentences
  - Works best with semantic transitions than thematic transitions because the audio preprocessing technique uses semantic analysis
  - Narrative documentaries (and same types of content structure) use the same vocabulary and terminology which gives a high similarity score thus giving a weak boundary signal 