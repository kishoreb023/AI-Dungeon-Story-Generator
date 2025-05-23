# 🧙‍♂️ AI Dungeon Story Generator

## 🎯 Objective

Create interactive, genre-based fantasy stories using powerful generative AI models like GPT-2/GPT-Neo, with a user-friendly interface built using Streamlit.

---

## 📖 Project Overview

The **AI Dungeon Story Generator** allows users to enter custom prompts, select a story genre (Fantasy, Mystery, Sci-Fi, etc.), and generate multiple imaginative story continuations using pretrained language models. The application enhances creativity, storytelling, and interaction using AI.

---

## 🛠️ Tools & Technologies

* **Language Model**: GPT-2 / GPT-Neo via Hugging Face Transformers
* **Frontend/UI**: Streamlit
* **Language**: Python
* **Others**: Streamlit-Lottie (for animations), JSON, tqdm

---

## ⚙️ Features

✅ Genre selection (Fantasy, Mystery, Sci-Fi, Adventure, etc.)
✅ Prompt-based story generation
✅ Multiple story continuations
✅ Downloadable story text files
✅ Aesthetic and colorful UI
✅ Streamlit-based deployment

---

## 🚀 How It Works

1. Select a genre from the dropdown.
2. Enter your custom story prompt.
3. Click **Generate** to get multiple unique story continuations.
4. Download your favorite version as a `.txt` file.

---

## 📁 Project Structure

```
AI_Dungeon_Story_Generator/
├── app/
│   ├── main.py
│   └── generator.py
├── assets/
│   └── prompts.json
├── requirements.txt
└── README.md
```

---

## 📸 Sample Prompts

* "The warrior stepped into the dragon’s cave..."
* "On a stormy night, the detective received a call..."
* "In the distant galaxy of Zorna, a rebellion brews..."
* *(More prompts in the project)*

---

## 📄 How to Run

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

---

## 📌 Deliverables

* Streamlit Web App
* Clean and readable Python Codebase
* Sample prompts file
* User-generated story download feature

---

## 💡 Future Enhancements

* Voice prompt input
* Custom model fine-tuning
* Dark/light mode toggle
* Integration with text-to-speech

