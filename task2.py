# task2_ner.py
import spacy
from spacy import displacy
import streamlit as st
from spacy.cli import download

# -------------------------------
# Step 0: Streamlit Page Config (MUST be first Streamlit command)
# -------------------------------
st.set_page_config(page_title="Named Entity Recognition", page_icon="🤖", layout="centered")

# -------------------------------
# Step 1: Load SpaCy Model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_model()

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🔍 Named Entity Recognition (NER) with SpaCy")
st.write("Paste any **customer query** below and see key entities highlighted.")

# -------------------------------
# Step 2: User Input
# -------------------------------
user_input = st.text_area("✍️ Enter customer query:", "Do you have the iPhone 15 in stock?")

if st.button("Extract Entities"):
    if user_input.strip():
        # Process text
        doc = nlp(user_input)

        # -------------------------------
        # Step 3: Extract Entities
        # -------------------------------
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        if entities:
            st.subheader("📌 Extracted Entities")
            for text, label in entities:
                st.success(f"**{text}** → {label}")

            # -------------------------------
            # Step 4: Visualize with displacy
            # -------------------------------
            st.subheader("🖼️ Visualization")
            html = displacy.render(doc, style="ent", jupyter=False)
            st.markdown(
                f"<div style='background-color:white; padding:10px; border-radius:10px'>{html}</div>",
                unsafe_allow_html=True
            )

        else:
            st.warning("No entities found. Try another query!")
    else:
        st.error("Please enter some text!")


