import torch
from flask import Flask, render_template, request, jsonify
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import pdfplumber
import io, base64, os, threading, re

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))

pipe = None
pipe_lock = threading.Lock()
is_loading = False

def load_pipeline():
    global pipe, is_loading
    is_loading = True
    print("Loading SDXL Turbo pipeline...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipeline.to("mps")
    pipe = pipeline
    is_loading = False
    print("Pipeline ready.")

threading.Thread(target=load_pipeline, daemon=True).start()

# ── text extraction helpers ────────────────────────────────────────────

def extract_pdf(file_bytes):
    import pdfplumber
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            # Use extract_words() so we have per-word y-positions.
            # We group words into lines by y-proximity, then detect
            # paragraph breaks where the vertical gap between consecutive
            # lines is significantly larger than the normal line height.
            words = page.extract_words(keep_blank_chars=False, use_text_flow=True)
            if not words:
                continue

            # Group words into lines by their rounded top position
            lines = []          # list of (y_top, line_text)
            cur_y   = None
            cur_words = []
            for w in words:
                y = round(float(w['top']))
                if cur_y is None or abs(y - cur_y) < 4:
                    cur_words.append(w['text'])
                    if cur_y is None:
                        cur_y = y
                else:
                    lines.append((cur_y, ' '.join(cur_words)))
                    cur_words = [w['text']]
                    cur_y = y
            if cur_words:
                lines.append((cur_y, ' '.join(cur_words)))

            if not lines:
                continue

            # Median inter-line gap → paragraph threshold = 1.8x median
            gaps = [lines[i+1][0] - lines[i][0] for i in range(len(lines) - 1)]
            if gaps:
                median_gap = sorted(gaps)[len(gaps) // 2]
                para_thresh = max(median_gap * 1.8, median_gap + 2)
            else:
                para_thresh = 9999

            # Assemble page text, injecting \n\n at paragraph breaks
            page_text = lines[0][1]
            for i in range(1, len(lines)):
                gap = lines[i][0] - lines[i-1][0]
                sep = '\n\n' if gap >= para_thresh else '\n'
                page_text += sep + lines[i][1]

            pages.append(page_text.strip())

    # Join pages with a double newline (no "Page Break" marker — just blank line)
    return '\n\n'.join(pages), len(pages)

def extract_txt(file_bytes):
    text = file_bytes.decode("utf-8", errors="replace")
    return text, 1

def extract_md(file_bytes):
    text = file_bytes.decode("utf-8", errors="replace")
    # strip markdown syntax for clean reading
    text = re.sub(r"#{1,6}\s+", "", text)
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    return text, 1

def extract_docx(file_bytes):
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs), 1
    except ImportError:
        return "(Install python-docx for .docx support)", 1

def extract_epub(file_bytes):
    try:
        import ebooklib
        from ebooklib import epub
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.chunks = []
                self._skip = False
            def handle_starttag(self, tag, attrs):
                if tag in ("script", "style"):
                    self._skip = True
            def handle_endtag(self, tag):
                if tag in ("script", "style"):
                    self._skip = False
            def handle_data(self, data):
                if not self._skip and data.strip():
                    self.chunks.append(data.strip())

        book = epub.read_epub(io.BytesIO(file_bytes))
        parts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            parser = TextExtractor()
            parser.feed(item.get_content().decode("utf-8", errors="replace"))
            chunk = " ".join(parser.chunks)
            if chunk.strip():
                parts.append(chunk)
        return "\n\n--- Page Break ---\n\n".join(parts), len(parts)
    except ImportError:
        return "(Install ebooklib for .epub support)", 1

def extract_rtf(file_bytes):
    text = file_bytes.decode("latin-1", errors="replace")
    # Basic RTF strip
    text = re.sub(r"\\[a-z]+\d* ?", " ", text)
    text = re.sub(r"[{}\\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(), 1

# ── routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/status")
def status():
    return jsonify({"ready": pipe is not None, "loading": is_loading})

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = file.filename.lower()
    file_bytes = file.read()

    try:
        if filename.endswith(".pdf"):
            text, pages = extract_pdf(file_bytes)
        elif filename.endswith(".epub"):
            text, pages = extract_epub(file_bytes)
        elif filename.endswith(".docx"):
            text, pages = extract_docx(file_bytes)
        elif filename.endswith(".md") or filename.endswith(".markdown"):
            text, pages = extract_md(file_bytes)
        elif filename.endswith(".rtf"):
            text, pages = extract_rtf(file_bytes)
        elif filename.endswith(".txt") or filename.endswith(".text"):
            text, pages = extract_txt(file_bytes)
        else:
            # fallback: try utf-8 text
            text, pages = extract_txt(file_bytes)
    except Exception as e:
        return jsonify({"error": f"Could not parse file: {e}"}), 500

    # Split into paragraph-based chunks for image-generation prompts.
    # CLIP handles up to 77 tokens (~55–65 words). We target 55 words minimum
    # per chunk but hard-cap at 100 words so CLIP never truncates silently.
    CLIP_TARGET = 55   # words — minimum before moving to next paragraph
    CLIP_MAX    = 100  # hard cap — truncate any chunk exceeding this
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    chunks = []
    i = 0
    while i < len(paragraphs):
        buf_words = paragraphs[i].split()
        i += 1
        # If the current paragraph is short, keep appending until target reached
        while i < len(paragraphs) and len(buf_words) < CLIP_TARGET:
            buf_words += paragraphs[i].split()
            i += 1
        # Hard-cap at CLIP_MAX words
        chunks.append(" ".join(buf_words[:CLIP_MAX]))

    return jsonify({
        "chunks": chunks,
        "display_text": text,
        "pages": pages,
        "total_chunks": len(chunks)
    })

@app.route("/generate", methods=["POST"])
def generate():
    if pipe is None:
        return jsonify({"error": "Model not ready yet"}), 503

    data = request.json
    # Keep prompt under ~77 CLIP tokens (~65 words is a safe ceiling)
    raw_prompt = data.get("prompt", "").strip()
    prompt_words = raw_prompt.split()[:65]
    prompt = " ".join(prompt_words)
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    print(f"\n[Vision Reader] Generating image for prompt:\n{prompt}\n", flush=True)

    with pipe_lock:
        image = pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    return jsonify({"image": img_b64})

@app.route("/fill-themes", methods=["POST"])
def fill_themes():
    """
    Use multilingual zero-shot-classification (xlm-roberta-large-xnli) to infer
    image themes from text. Works with Chinese, English, and any other language.

    Receives two contexts:
      - current_text: current page + next few chunks (high weight)
      - book_text:    sparse sample across the whole book (low weight, for era/place stability)

    For each theme field we score all candidate labels against both contexts,
    then blend: final_score = 0.65 * current + 0.35 * book
    This means the current page drives tone/style/palette while era/place
    lean on the whole-book picture.
    """
    data         = request.json or {}
    current_text = (data.get("current_text") or "").strip()
    book_text    = (data.get("book_text")    or "").strip()

    if not current_text and not book_text:
        return jsonify({"error": "No context provided"}), 400

    # ── Load model once, cache on function object ────────────────────────────────
    try:
        from transformers import pipeline as hf_pipeline

        if not hasattr(fill_themes, "_nlp"):
            print("[fill-themes] Loading xlm-roberta-large-xnli …")
            fill_themes._nlp = hf_pipeline(
                "zero-shot-classification",
                model="joeddav/xlm-roberta-large-xnli",
                device=0 if torch.backends.mps.is_available() else -1,
            )
            print("[fill-themes] Model ready.")

        clf = fill_themes._nlp

        def classify(text, labels, multi=False):
            """Run zero-shot classification, return {label: score} dict."""
            if not text or not text.strip():
                return {l: 0.0 for l in labels}
            try:
                # Truncate to ~512 tokens worth of chars to stay within model limits
                out = clf(text[:1800], candidate_labels=labels, multi_label=multi)
                # Pipeline may return a list (batched) or a single dict
                if isinstance(out, list):
                    out = out[0]
                return dict(zip(out["labels"], out["scores"]))
            except Exception as ex:
                print(f"[classify] failed: {ex}")
                return {l: 0.0 for l in labels}

        def blend(cur_scores, book_scores, cur_w=0.65, book_w=0.35):
            """Weighted blend of two score dicts, return best label."""
            # Union of all known labels
            labels = list({**cur_scores, **book_scores}.keys())
            if not labels:
                return ""
            blended = {
                l: cur_w  * cur_scores.get(l, 0.0)
                 + book_w * book_scores.get(l, 0.0)
                for l in labels
            }
            return max(blended, key=blended.get)

        # ── Candidate labels per field (language-neutral concepts) ───────────────

        ERA_LABELS = [
            "prehistoric or ancient times",
            "ancient civilisation, classical antiquity",
            "medieval era, middle ages",
            "Renaissance, early modern period",
            "17th or 18th century, Enlightenment",
            "19th century, Victorian or industrial era",
            "early 20th century, 1900s to 1940s",
            "mid 20th century, 1950s to 1970s",
            "late 20th century, 1980s to 1990s",
            "contemporary, 2000s to present",
            "near future",
            "far future, science fiction",
        ]
        PLACE_LABELS = [
            "Western Europe or North America",
            "Eastern Europe or Russia",
            "East Asia, China or Japan or Korea",
            "South or Southeast Asia",
            "Middle East or Central Asia",
            "Africa",
            "Latin America",
            "Oceania or Pacific",
            "interstellar space or alien world",
            "post-apocalyptic or dystopian wasteland",
            "fantasy or mythical realm",
            "digital or virtual world",
        ]
        CULTURE_LABELS = [
            "ancient Egyptian or Mesopotamian",
            "ancient Greek or Roman",
            "Celtic or Norse or Viking",
            "feudal Japanese or East Asian imperial",
            "imperial Chinese or dynastic",
            "South Asian, Hindu or Mughal",
            "Islamic or Ottoman",
            "Mesoamerican, Aztec or Maya or Inca",
            "African tribal or animist",
            "Byzantine or Orthodox Christian",
            "Gothic medieval, Catholic",
            "Baroque or Rococo, aristocratic European",
            "Victorian or colonial",
            "Art Deco or Jazz Age",
            "cyberpunk or futuristic dystopian",
            "steampunk or gaslamp",
            "space opera, galactic civilisation",
        ]
        STYLE_LABELS = [
            "oil painting, fine art",
            "ink brush painting, East Asian style",
            "watercolor painting",
            "pencil or charcoal sketch",
            "woodblock print or engraving",
            "documentary photography, realistic",
            "cinematic film still",
            "graphic novel or manga illustration",
            "fantasy concept art",
            "science fiction concept art",
            "impressionist painting",
            "film noir, black and white",
        ]
        TONE_LABELS = [
            "dark, ominous, threatening",
            "joyful, celebratory, bright",
            "romantic, tender, intimate",
            "epic, heroic, triumphant",
            "mysterious, enigmatic, shadowy",
            "melancholic, sorrowful, elegiac",
            "tense, suspenseful, dramatic",
            "whimsical, comedic, absurdist",
            "contemplative, philosophical, serene",
            "gothic, horrifying, supernatural",
            "action-packed, intense, violent",
            "nostalgic, wistful, dreamlike",
        ]
        PALETTE_LABELS = [
            "deep blacks and midnight blues, dark",
            "warm ambers and golds, golden hour",
            "icy whites and cool greys, cold",
            "lush greens and earthy browns, natural",
            "deep ocean blues and aquamarine",
            "crimson reds and fiery orange",
            "rich jewel tones, gold and burgundy, opulent",
            "muted greys and foggy tones, desaturated",
            "warm ochres and sandy tones, arid",
            "neon pinks and electric blues, cyberpunk",
            "soft pastels and pale rose, gentle",
            "deep space purples and starlight",
        ]

        # ── Classify both contexts ────────────────────────────────────────────────
        # Era and place: book-level context matters more (stability across book)
        # Tone, style, palette: current page matters more (page-by-page feel)

        cur_era     = classify(current_text, ERA_LABELS)
        book_era    = classify(book_text,    ERA_LABELS)
        era_label   = blend(cur_era, book_era, cur_w=0.40, book_w=0.60)

        cur_place   = classify(current_text, PLACE_LABELS)
        book_place  = classify(book_text,    PLACE_LABELS)
        place_label = blend(cur_place, book_place, cur_w=0.35, book_w=0.65)

        cur_cult    = classify(current_text, CULTURE_LABELS)
        book_cult   = classify(book_text,    CULTURE_LABELS)
        cult_label  = blend(cur_cult, book_cult, cur_w=0.50, book_w=0.50)

        cur_style   = classify(current_text, STYLE_LABELS)
        book_style  = classify(book_text,    STYLE_LABELS)
        style_label = blend(cur_style, book_style, cur_w=0.70, book_w=0.30)

        cur_tone    = classify(current_text, TONE_LABELS)
        book_tone   = classify(book_text,    TONE_LABELS)
        tone_label  = blend(cur_tone, book_tone, cur_w=0.80, book_w=0.20)

        cur_pal     = classify(current_text, PALETTE_LABELS)
        book_pal    = classify(book_text,    PALETTE_LABELS)
        pal_label   = blend(cur_pal, book_pal, cur_w=0.75, book_w=0.25)

        # Strip the verbose classifier descriptions down to clean short phrases
        def shorten(label):
            # Take everything before the first comma or parenthesis
            return re.split(r"[,\(]", label)[0].strip()

        return jsonify({
            "era":     shorten(era_label),
            "place":   shorten(place_label),
            "culture": shorten(cult_label),
            "style":   shorten(style_label),
            "tone":    shorten(tone_label),
            "palette": shorten(pal_label),
        })

    except Exception as e:
        print(f"[fill-themes] zero-shot classification failed: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, port=5050)
