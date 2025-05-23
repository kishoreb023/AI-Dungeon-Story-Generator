# app/story_saver.py

from datetime import datetime

def save_story(prompt, continuations, genre):
    filename = "generated_story.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Genre: {genre}\nPrompt: {prompt}\n\n")
        for i, story in enumerate(continuations, 1):
            f.write(f"--- Story {i} ---\n{story}\n\n")
        f.write(f"Saved on: {datetime.now()}\n")
