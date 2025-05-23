import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

MODEL_PATH = "models/gpt2_custom"

@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.set_page_config(page_title="AI Dungeon Story Generator", layout="centered")
st.title("🧙 AI Dungeon Story Generator")
st.markdown("Craft immersive stories with proper structure and climax!")

# Genre-driven structure templates
genre_templates = {
    "Fantasy": "Story Title: The Glowing Eyes\n\nOnce upon a time in a forgotten kingdom, ",
    "Science Fiction": "Story Title: The Last Contact\n\nIn the year 3022, humanity reached beyond the stars. ",
    "Mystery": "Story Title: The Silent Clue\n\nDetective Elara arrived at the old manor. Something felt wrong. ",
    "Adventure": "Story Title: The Hidden Temple\n\nBeneath the jungles of South America, legends spoke of a lost temple. ",
    "Horror": "Story Title: Whispers in the Walls\n\nThe abandoned asylum hadn't seen light in decades. "
}

# UI Elements
genre = st.selectbox("Choose a genre", list(genre_templates.keys()) + ["Custom"])

if genre != "Custom":
    base_prompt = genre_templates[genre]
    prompt = st.text_area("Story begins with:", value=base_prompt, height=150)
else:
    prompt = st.text_area("Enter your custom story prompt", height=150)

length = st.slider("Story Length (tokens)", 100, 700, 300, step=50)

# Generate Button
if st.button("🔮 Generate Story"):
    with st.spinner("Creating your structured story..."):
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_length=length,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            temperature=0.85,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        story_full = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean extra text after ending punctuation
        import re
        story_trimmed = re.split(r'(?<=[\.\!\?])\s', story_full, maxsplit=1)[-1]
        full_story = prompt.strip() + " " + story_trimmed.strip()

    st.subheader("📖 Your Structured Story")
    st.write(full_story)

    st.download_button(
        label="📥 Download Story",
        data=full_story,
        file_name="structured_ai_story.txt",
        mime="text/plain"
    )

st.markdown("---")
st.caption("✨ Powered by GPT-2 | Optimized for narrative flow")
