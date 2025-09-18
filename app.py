# app.py
from flask import Flask, render_template, request
import io
import os
import gc
import gdown
import joblib
import pdfplumber
import numpy as np
from sentence_transformers import util
import PyPDF2
from werkzeug.exceptions import RequestEntityTooLarge

# -------- CONFIG ----------
app = Flask(__name__)
# set max upload size (bytes) — here 10 MB (adjust as needed)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# -------- Helpers ----------
def download_and_load(file_id, local_path):
    """Download from Google Drive (if not exists) and load pickle file."""
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(local_path):
        gdown.download(url, local_path, quiet=False)
    return joblib.load(local_path)

def read_pdf_from_bytes(file_bytes):
    """
    Extract text from PDF bytes without saving to disk.
    Tries pdfplumber first, falls back to PyPDF2 if needed.
    """
    text_pages = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text_pages.append(txt)
        if text_pages:
            return "\n".join(text_pages)
    except Exception:
        pass

    # fallback to PyPDF2
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text_pages.append(txt)
        return "\n".join(text_pages)
    except Exception as e:
        return f"[ERROR] Could not extract PDF text: {e}"

# -------- Load models / data once ----------
judgment_texts = download_and_load("1vAA2spJ-AzHhBqs-gl6gL5_wDk22a5VW", "judgment_texts.pkl")
model = download_and_load("1-pje6HUuprf19yGIbJQA7MNwPNGqTkF0", "model.pkl")
case_names = download_and_load("1_IZQmTuucallXvQaeLM8P9q0co79_JD6", "case_names.pkl")
embeddings = download_and_load("1molCaZLasdsSMqqskRIcHnmnpQWfAupF", "embeddings.pkl")
modellog = download_and_load("1XALJYnXhZB9gXdjAgz8y_I852CpYt-eg", "modellog.pkl")

# -------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    predicted_category = None

    if request.method == "POST":
        input_text = request.form.get("text_input", "").strip()

        pdf_bytes = None
        if not input_text:
            # check for file in request
            uploaded = request.files.get("file")
            if not uploaded or uploaded.filename == "":
                return "❌ No input provided (paste text or upload a PDF).", 400

            # optional: strictly allow only .pdf extension
            if not uploaded.filename.lower().endswith(".pdf"):
                return "❌ Only PDF files are allowed.", 400

            # read file into memory (no saving)
            pdf_bytes = uploaded.read()
            if not pdf_bytes:
                return "❌ Uploaded file is empty.", 400

            # extract text from bytes
            input_text = read_pdf_from_bytes(pdf_bytes)

        # get text after "JUDGMENT" if present
        start_index = input_text.find("JUDGMENT")
        text = input_text[start_index:] if start_index != -1 else input_text

        # predicted category
        predicted_category = modellog.predict([text])[0]

        # semantic search - top 5
        query_embedding = model.encode(text, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, embeddings)[0]
        top_results = np.argsort(-cos_scores)[:5]

        results = []
        for idx in top_results:
            results.append({
                "case": case_names[idx],
                "score": float(cos_scores[idx]),
                "preview": judgment_texts[idx][:500]
            })

        # ********** PRIVACY: clean up large objects ASAP **********
        try:
            # remove sensitive text / bytes references
            del input_text
            del text
            if 'pdf_bytes' in locals():
                del pdf_bytes
            gc.collect()
        except Exception:
            pass

    return render_template("index.html", results=results, category=predicted_category)

# handle oversized uploads
@app.errorhandler(RequestEntityTooLarge)
def handle_over_max(e):
    return "Uploaded file is too large. Max size allowed is 10 MB.", 413

# -------- Run ----------
if __name__ == "__main__":
    app.run(debug=True)
