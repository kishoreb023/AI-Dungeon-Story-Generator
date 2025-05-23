import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model_and_tokenizer(model_name="gpt2"):
    print(f"Loading model and tokenizer: {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def generate_story(model, tokenizer, prompt, max_length=800, temperature=1.0, top_k=50, top_p=0.95):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text = generated_text[len(prompt):].strip()
    return format_story_long(generated_text)


def format_story_long(text, max_lines=50, words_per_line=12):
    words = text.split()
    lines = []
    max_words = max_lines * words_per_line
    for i in range(0, min(len(words), max_words), words_per_line):
        line = ' '.join(words[i:i + words_per_line])
        lines.append(line)
        if len(lines) >= max_lines:
            break

    # Insert simple story structure headings
    if len(lines) >= 50:
        lines.insert(0, "**Beginning:**")
        lines.insert(18, "\n**Build-Up:**")
        lines.insert(38, "\n**Climax:**")
    else:
        lines.insert(0, "**Beginning:**")
        lines.insert(len(lines)//2, "\n**Build-Up:**")
        lines.append("\n**Climax:**")

    return '\n\n'.join(lines)


def generate_multiple_stories(model, tokenizer, prompt, n=3):
    stories = []
    for i in range(n):
        print(f"Generating story {i + 1} of {n}...")
        story = generate_story(model, tokenizer, prompt)
        stories.append(f"--- Story Sample {i + 1} ---\n\n{story}\n")
    return '\n'.join(stories)


if __name__ == "__main__":
    prompt = "In the icy north, a flame never extinguished begins to roar again..."

    model_name = "gpt2"  # Or your custom model folder
    model, tokenizer = load_model_and_tokenizer(model_name)

    all_stories = generate_multiple_stories(model, tokenizer, prompt, n=3)
    print(all_stories)
