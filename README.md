# Vision Reader

https://github.com/user-attachments/assets/96688a2b-15f3-4512-b02a-3ba7aa70ae7c

**Real-time AI illustration for long reading — running entirely on your laptop.**

---

## What it does

Vision Reader is a local document reader that generates images as you read. As you scroll through a PDF, novel, or essay, the app silently runs SDXL-Turbo locally in the background and produces illustrations. 

On a M3 MacBook each generation takes 1-3 seconds, paired with generating one image ahead, you never need to wait for machine. 


**Features:**
- Supported formats: PDF · TXT · Markdown · DOCX · EPUB · RTF
- Continuous scrolling text with a cinematic cylinder-fade effect that keeps focus on the current line
- Vertical indicator bar showing exactly which passage is driving the current image
- Select any text and click **⊙ Generate from selection** to illustrate an arbitrary excerpt 
- Adjustable fade opacity, dark / light theme, resizable panels, persistent settings

---

## Installation

### Requirements


Install dependencies:

```bash
pip install -r requirements.txt
```

### Running

```bash
python app.py
```

Then open [http://localhost:5050](http://localhost:5050) in your browser.

### First run — model download

Vision Reader uses [`stabilityai/sdxl-turbo`](https://huggingface.co/stabilityai/sdxl-turbo) from HuggingFace. The first time you run the app, the model weights (~6.7 GB) will be downloaded automatically and cached locally. This takes a few minutes depending on your connection. Subsequent launches load from cache and are fast.

The status indicator in the top-right corner shows **Loading model…** while the pipeline initialises, then turns green when ready to generate.

---

## Philosophy

*Why does an AI-generated illustration help you understand what you're reading?*

The transformer is anchored in token co-occurrence statistics. Human intelligence is anchored in the physical world. 

Human understanding is not built on tokens. It is built on a lifetime of physical experience — things seen, felt, walked through, sat inside of. When you read a word, you don't look up a definition in a dictionary in your head. You reach for a memory that has weight and texture and light. System 2 reasoning — the slow, deliberate kind — runs on top of System 1 pattern recognition that was built over years of perceiving the real world.

This is why a Chinese reader who has never encountered Western architecture genuinely cannot picture a medieval great hall. The words are there. The grammar parses. But there is no image to anchor to. The same is true in reverse: an American reader opening a Ming dynasty novel for the first time meets a world where every object, every social gesture, every spatial arrangement is foreign. 

Vision Reader is a small attempt to address this. Not by explaining — it just means more text — but by quietly placing an image next to each passage as you go. A visual impression for your system 2 reading to anchor on. 

There is an irony here. AI tools have made shallow reading easier, and deep reading rarer. Summarise this, explain that, just give me the answer. The result is a gradual erosion of the capacity to sit with a text long enough for it to matter. Vision Reader is built against that tendency. It does not summarise. It does not extract. It makes reading longer and more immersive, not shorter and more efficient. 

---

## Future directions

The current version is a proof of concept. Where this goes next:

**Character consistency.** A recurring character should look the same on page 300 as they did on page 3. User could customize character appearance at start. 

**Theme consistency.** When a book is opened, an optional one-time summarisation pass extracts a global theme prompt — period, mood, visual register — that is prepended to every image generation call as a style frame. 

**Dedicated model training.** SDXL-Turbo is a general-purpose model. A model trained specifically on literary illustration — book covers, historical engravings, architectural drawings, maps — would produce images far better suited to the task. The visual vocabulary of a well-read person is very different from the visual vocabulary of the internet.

**Diagram and figure generation.** For non-fiction and academic reading, the most useful illustration is often not a scene but a diagram — a chart of relationships, a timeline, a spatial layout. Generating structured figures from text is a natural extension.

**Second language learning.** The same visual anchoring that helps a reader engage with an unfamiliar culture helps a language learner acquire vocabulary. Personal word list, flash cards, user sharing market, are all useful features. 


---

## Hire / Invest

This project is being built by someone thinking seriously about how reading tools should work — not as productivity software, but as instruments for attention and understanding.

If you are building something in the space of reading, language learning, educational technology, or human-AI interaction and this resonates, I would like to talk.

If you are a researcher or engineer interested in the dedicated model training direction, same.

Reach out via GitHub issues or sincerexuyuan@gmail.com 

---


## Licence

MIT — do what you like, run it locally, modify freely.
