import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

model_path = r'/kaggle/working/model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

translation_pipeline = pipeline('translation', model=model, tokenizer=tokenizer)

st.title("Translation Model")

input_text = st.text_area("Enter text to translate:")

if st.button("Translate"):
    if input_text:
        translated_text = translation_pipeline(input_text)
        st.write("Translation:", translated_text[0]['translation_text'])
    else:
        st.write("Please enter some text.")