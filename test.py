
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Wybór modelu do tłumaczenia
model_name = "clui/opus-it-pl-v1"  # Model tłumaczenia z włoskiego na polski
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tytuł aplikacji
st.title("Song Translator")


# Wprowadzenie tekstu piosenki
input_text = st.text_area("Enter song lyrics:", height=300)

if st.button("Translate"):
    if input_text:
        # Podział tekstu na linie
        lines = input_text.split('\n')
        translated_lines = []
        
        # Tłumaczenie każdej linii
        for line in lines:
            if line.strip():  # Sprawdzenie, czy linia nie jest pusta
                # Tokenizacja
                inputs = tokenizer(line, return_tensors="pt", padding=True)
                
                # Generowanie tłumaczenia
                translated_outputs = model.generate(**inputs)
                
                # Dekodowanie przetłumaczonego tekstu
                translated_text = tokenizer.decode(translated_outputs[0], skip_special_tokens=True)
                
                translated_lines.append(translated_text)

        # Wyświetlanie przetłumaczonych linii w dwóch kolumnach
        col1, col2 = st.columns(2)  # Dwie kolumny
        
        with col1:
            st.subheader("Original Lines:")
            for original in lines:
                st.write(original)

        with col2:
            st.subheader("Translated Lines:")
            for translated in translated_lines:
                st.write(translated)
    else:
        st.warning("Please enter the song lyrics.")

