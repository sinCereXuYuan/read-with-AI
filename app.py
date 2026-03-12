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
    # CLIP handles up to 77 tokens (~55–65 words). We use one paragraph per
    # chunk; if a paragraph is very short (< 40 words) we extend it by
    # appending the next paragraph(s) until we reach ~55 words or exceed it.
    # This avoids the CLIP truncation warning while keeping prompts coherent.
    CLIP_TARGET = 55   # words — comfortably under 77-token limit
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
        chunks.append(" ".join(buf_words))

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

if __name__ == "__main__":
    app.run(debug=False, port=5050)
