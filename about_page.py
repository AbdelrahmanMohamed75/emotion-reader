import json
import os

import streamlit as st

st.title("About this project")
st.write(
    "Emotion Reader takes one English sentence and tells you which of six emotions it expresses: "
    "sadness, joy, love, anger, fear or surprise. It also shows a score for every emotion, "
    "so you can see when the model is unsure."
)

st.header("What the project does")
st.write(
    "Written text carries feeling, but a computer cannot read it the way a person does. "
    "This project trains a language model to recognise the emotion in short texts and puts it behind "
    "a simple web page. Anyone can type a sentence and get an answer in a second, with no setup."
)

st.header("How it works")
st.markdown(
    """
1. **You type a sentence.** It can be up to 500 characters long.
2. **The text is cleaned.** It is lowercased and apostrophes are removed, exactly like the training data was.
3. **The tokenizer splits it into word pieces.** Each input is cut or padded to 60 tokens.
4. **A fine-tuned DistilBERT model scores the sentence.** DistilBERT is a smaller, faster version of BERT, trained here to pick one of six emotions.
5. **The scores become percentages.** A softmax step turns them into six numbers that add up to 100%.
"""
)

st.header("Why this model")
st.write(
    "Several models were trained on the same data (the dair-ai/emotion set of English tweets) "
    "and scored on the same 2,000 test sentences."
)
st.table(
    {
        "Model": [
            "SimpleRNN",
            "GRU (two layers)",
            "LSTM",
            "BiGRU",
            "BiGRU with GloVe word vectors",
            "DistilBERT, fine-tuned",
            "DistilBERT, fine-tuned with negation augmentation (used here)",
        ],
        "Test accuracy": ["34.7%", "8.0%", "88.8%", "91.8%", "92.7%", "92.5%", "92.3%"],
    }
)
st.write(
    "The best models are within about one percentage point of each other, too small a gap to pick a "
    "winner on accuracy alone. What set them apart was negation. Before augmentation, the fine-tuned "
    "DistilBERT got only 4 of 64 held-out negation sentences right (6%). After adding about 1,700 "
    "generated training sentences, it got 62 of 64 right (97%) while overall accuracy stayed at about 92%. "
    "The BiGRU with GloVe was not given this treatment and still read \"I am not happy at all right now\" as joy."
)
st.caption(
    "The 64 held-out sentences use words and sentence patterns that were not in the training data, "
    "but they are few and generated from templates. They show the fix works for that pattern, "
    "not that negation is solved."
)

st.header("Model accuracy")
st.write("These numbers come from testing the model on sentences it never saw during training.")
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Artifacts", "metrics.json")
try:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
except Exception:
    m = {}

if isinstance(m.get("accuracy"), (int, float)):
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{m['accuracy'] * 100:.1f}%")
    if isinstance(m.get("macro_f1"), (int, float)):
        c2.metric("Macro F1", f"{m['macro_f1'] * 100:.1f}%")
    info = ", ".join(str(x) for x in [m.get("dataset"), f"{m['test_size']:,} test sentences" if m.get("test_size") else ""] if x)
    if info:
        st.caption(info)
    per_class = m.get("per_class") or {}
    if per_class:
        st.table(
            {
                "Emotion": [k.capitalize() for k in per_class],
                "Precision": [f"{v['precision'] * 100:.1f}%" for v in per_class.values()],
                "Recall": [f"{v['recall'] * 100:.1f}%" for v in per_class.values()],
                "F1": [f"{v['f1'] * 100:.1f}%" for v in per_class.values()],
                "Sentences": [v["support"] for v in per_class.values()],
            }
        )
else:
    st.info("Evaluation results have not been added yet.")
st.caption(
    "Accuracy is the share of sentences labelled correctly. Precision tells how often a predicted emotion was right. "
    "Recall tells how many sentences of an emotion the model found. F1 combines the two, and macro F1 averages F1 "
    "over all six emotions equally."
)

st.header("Problems we ran into")
problems = [
    (
        "Negation",
        "A sentence like \"I am not happy at all\" contains a happy word but expresses the opposite. "
        "A test made the problem clear: on 64 held-out negation sentences the model was right only 4 times (6%). "
        "We generated about 1,700 extra training sentences from templates and re-trained. "
        "It now gets 62 of 64 right (97%), and overall test accuracy did not drop.",
    ),
    (
        "Fixing negation broke other sentences",
        "After the first fix, a sentence like \"I can't believe that just happened\" was read as joy or sadness, "
        "because the model had learned that \"can't\" and \"not\" point to sadness. We added counter-examples "
        "(joy, love and surprise sentences that contain \"can't\" or \"never\"), plus sentences such as "
        "\"I am not afraid anymore\", and re-trained from the original model. That sentence is now read as surprise.",
    ),
    (
        "Models that did not learn",
        "A plain SimpleRNN and a two-layer GRU scored 35% and 8% on the test set, so they failed to learn the task. "
        "We moved to bidirectional layers with early stopping, then to pretrained GloVe vectors and finally to a "
        "pretrained DistilBERT. All of these trained well.",
    ),
    (
        "Apostrophes and contractions",
        "\"can't\" and \"cant\" are split into different tokens, so text cleaned differently at training time and at "
        "prediction time gives worse results. The same cleaning (lowercase, no apostrophes) is applied before training "
        "and before every prediction.",
    ),
    (
        "Confidence is not correctness",
        "A model can be 95% sure and still be wrong, especially on sarcasm or mixed feelings. The page shows the score "
        "of all six emotions, not only the winner, and states that the percentage shows how strongly the model leans.",
    ),
    (
        "Model size and hosting",
        "The trained model file is about 270 MB, and PyTorch needs a fair amount of memory. Many free hosts cannot run it. "
        "The model is stored on the Hugging Face Hub and loaded when the app starts.",
    ),
]
for title, body in problems:
    with st.expander(title):
        st.write(body)

st.header("Limitations")
st.write(
    "The model reads English only and looks at one sentence at a time. It chooses from six emotions, so feelings "
    "outside that list, mixed emotions, and sarcasm will not always be recognised correctly. Some warm sentences "
    "still fail: \"I can't believe you remembered, this means everything to me\" is read as sadness."
)
