import re

import streamlit as st

st.set_page_config(page_title="Emotion Reader", page_icon="😄")

MODEL_ID = "abdelrahmanemam10/emotion-reader-model"
# Same order the model was trained with (label_names.json)
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
EMOJIS = {"sadness": "😢", "joy": "😄", "love": "❤️", "anger": "😠", "fear": "😨", "surprise": "😲"}
MAX_LEN = 60

EXAMPLES = {
    "Happy day": "I am so happy today, everything is perfect!",
    "Not happy": "I am not happy at all right now",
    "Can't believe it": "I can't believe that just happened",
    "Not scared": "I am not scared anymore",
    "Missing someone": "I miss you so much it hurts",
    "Angry reply": "How dare you talk to me like that!",
}


@st.cache_resource(show_spinner="Loading the model. The first visit takes a minute...")
def load_model():
    import torch
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_ID)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_ID).eval()
    return torch, tokenizer, model


def normalize(text):
    # Same cleaning used when the model was fine-tuned: lowercase, no apostrophes.
    text = text.lower().replace("'", "").replace("\u2019", "")
    return re.sub(r"\s+", " ", text).strip()


def predict(text):
    torch, tokenizer, model = load_model()
    enc = tokenizer(
        normalize(text),
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=1)[0].tolist()
    return dict(zip(LABELS, probs))


def use_example(sentence):
    st.session_state["text"] = sentence
    st.session_state["run"] = True


def predictor():
    st.title("Find the emotion behind any sentence")
    st.write("Write one sentence in English. The model scores six emotions and shows which one leads.")

    st.text_area(
        "Your sentence",
        key="text",
        max_chars=500,
        height=120,
        placeholder="I can't believe how happy I am today",
    )
    go = st.button("Analyze emotion", type="primary", use_container_width=True)

    st.caption("Or try an example")
    cols = st.columns(3)
    for i, (label, sentence) in enumerate(EXAMPLES.items()):
        cols[i % 3].button(label, key=f"ex{i}", on_click=use_example, args=(sentence,), use_container_width=True)

    run = go or st.session_state.pop("run", False)
    if not run:
        return

    text = st.session_state.get("text", "").strip()
    if not text:
        st.warning("Write a sentence first.")
        return

    try:
        probs = predict(text)
    except Exception as e:
        st.error(f"Something went wrong: {type(e).__name__}: {e}")
        return

    top = max(probs, key=probs.get)
    st.divider()
    st.header(f"{EMOJIS[top]} {top.capitalize()}")
    st.caption(f"Leaning {probs[top] * 100:.1f}% toward this emotion")
    for name, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        st.progress(p, text=f"{EMOJIS[name]} {name.capitalize()}  {p * 100:.1f}%")
    st.caption("Confidence shows how strongly the model leans, not how likely it is to be right.")


pg = st.navigation(
    [
        st.Page(predictor, title="Predictor", icon="😄", default=True),
        st.Page("about_page.py", title="About and accuracy", icon="📊"),
    ]
)
pg.run()
