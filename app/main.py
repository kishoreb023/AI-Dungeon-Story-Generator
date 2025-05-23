import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import random

# Load your custom GPT-2 model
MODEL_PATH = "models/gpt2_custom"  # Update to your actual model path
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)

# Set page config
st.set_page_config(page_title="AI Dungeon Story Generator", layout="wide")

# Custom background color using HTML/CSS
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f2f0eb;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("🧙‍♂️ AI Dungeon Story Generator")
st.markdown("Create high-quality fantasy stories using your own GPT-2 model.")

# Sample prompts
sample_prompts = [
    "A lone warrior enters the cursed forest of whispers...",
    "In a village hidden under eternal night, a child is born with glowing eyes...",
    "The gates to the Dragon King's tomb trembled as the sky cracked open...",
    "A mysterious book appears on a librarian's desk, bound in midnight leather...",
    "In the icy north, a flame never extinguished begins to roar again...",
]

# Prompt selection
st.subheader("Choose or Type a Story Prompt")
prompt_mode = st.radio("Prompt Mode", ["Type your own", "Random prompt"])
if prompt_mode == "Type your own":
    prompt = st.text_area("Enter your story prompt:", height=100)
else:
    prompt = random.choice(sample_prompts)
    st.success(f"Random prompt selected: **{prompt}**")

# Generation settings
max_length = st.slider("Story Length (words)", 150, 600, 400, step=50)
temperature = st.slider("Creativity (Temperature)", 0.5, 1.5, 1.0)
top_p = st.slider("Top-p (Nucleus Sampling)", 0.5, 1.0, 0.9)

# Generate story
if st.button("Generate Story"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Crafting your story..."):
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            output = model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id
            )
            story = tokenizer.decode(output[0], skip_special_tokens=True)
            story = story[len(prompt):].strip()  # remove prompt from generated part
        st.markdown("### Your Story")
        st.write(f"**Prompt:** {prompt}\n\n{story}")

        # Allow download
        story_filename = "generated_story.txt"
        st.download_button(
            label="Download Story",
            data=f"{prompt}\n\n{story}",
            file_name=story_filename,
            mime="text/plain"
        )
