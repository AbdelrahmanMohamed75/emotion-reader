import json
import logging
import os
import re
import traceback

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "Artifacts")
MAX_CHARS = 500

EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

app = Flask(__name__, static_folder="static", static_url_path="/static")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("emotion")

_state = {}


def get_model():
    """Load the model once, on the first request (keeps startup fast)."""
    if not _state:
        if not os.path.exists(os.path.join(ART_DIR, "app_config.json")):
            raise FileNotFoundError(
                "Model files not found in the Artifacts folder. "
                "Copy the saved model files there first."
            )

        import torch
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizerFast,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = DistilBertTokenizerFast.from_pretrained(ART_DIR)
        model = DistilBertForSequenceClassification.from_pretrained(ART_DIR)
        model.to(device).eval()

        with open(os.path.join(ART_DIR, "label_names.json")) as f:
            labels = json.load(f)
        with open(os.path.join(ART_DIR, "app_config.json")) as f:
            cfg = json.load(f)

        _state.update(
            torch=torch,
            device=device,
            tokenizer=tokenizer,
            model=model,
            labels=labels,
            cfg=cfg,
        )
    return _state


def normalize(text):
    # Same cleaning used when the model was fine-tuned: no apostrophes, lowercase.
    text = text.lower().replace("'", "").replace("\u2019", "")
    return re.sub(r"\s+", " ", text).strip()


def predict_probs(text):
    s = get_model()
    if s["cfg"].get("normalize_text"):
        text = normalize(text)

    enc = s["tokenizer"](
        text,
        max_length=s["cfg"]["max_len"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(s["device"])

    with s["torch"].no_grad():
        logits = s["model"](**enc).logits
    probs = s["torch"].softmax(logits, dim=1)[0].cpu().tolist()
    return dict(zip(s["labels"], probs))


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/about")
def about():
    return send_from_directory(app.static_folder, "about.html")


@app.route("/api/metrics")
def metrics():
    path = os.path.join(ART_DIR, "metrics.json")
    if not os.path.exists(path):
        return jsonify(available=False)
    with open(path, encoding="utf-8") as f:
        return jsonify(available=True, **json.load(f))


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify(error="Write a sentence first."), 400
    if len(text) > MAX_CHARS:
        return jsonify(error=f"Keep it under {MAX_CHARS} characters."), 400

    try:
        probs = predict_probs(text)
    except Exception as e:  # show the REAL error instead of a blank "Internal Server Error"
        log.error("Prediction failed:\n%s", traceback.format_exc())
        return jsonify(error=f"{type(e).__name__}: {e}"), 500

    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_prob = ranked[0]
    return jsonify(
        label=top_label,
        emoji=EMOJIS.get(top_label, ""),
        confidence=top_prob,
        probabilities=[
            {"label": k, "emoji": EMOJIS.get(k, ""), "probability": v}
            for k, v in ranked
        ],
    )


@app.errorhandler(Exception)
def on_error(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    log.error("Unhandled error:\n%s", traceback.format_exc())
    return jsonify(error=f"{type(e).__name__}: {e}"), 500


# On the server (gunicorn) load the model once at startup instead of on the first visit.
if os.environ.get("PRELOAD_MODEL"):
    get_model()


if __name__ == "__main__":
    # Load the model at startup so any problem shows up here in the terminal.
    try:
        get_model()
        log.info("Model loaded OK.")
    except Exception:
        log.error("Model failed to load:\n%s", traceback.format_exc())
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), debug=False)
