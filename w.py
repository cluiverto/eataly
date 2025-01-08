import streamlit as st
import easyocr
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage
from PIL import Image
import numpy as np


ocr = easyocr.Reader(['it'])
chat_model = ChatMistralAI(model="mistral-large-latest")

def extract_text_from_image(image):
    # Konwersja obrazu z PIL na tablicę numpy
    image_np = np.array(image)
    result = ocr.readtext(image_np)
    return " ".join([text[1] for text in result])

def translate_text(input_text):
    messages = [
        HumanMessage(content=f"Proszę przetłumaczyć poniższy tekst na polski: {input_text}")
    ]
    
    try:
        response = chat_model.invoke(messages)
        return response.content
    except Exception as e:
        return f"Wystąpił błąd podczas tłumaczenia: {str(e)}"

# Interfejs użytkownika Streamlit
st.title("Eataly")
uploaded_file = st.file_uploader("Prześlij obraz", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Odczytanie przesłanego pliku jako obrazu
    image = Image.open(uploaded_file)

    # Wyświetlenie przesłanego zdjęcia
    st.image(image, caption="Przesłane zdjęcie", use_column_width=True)

    if st.button("Przetłumacz"):
        # Ekstrakcja tekstu z obrazu po naciśnięciu przycisku
        extracted_text = extract_text_from_image(image)
        
        if extracted_text:
            translated_text = translate_text(extracted_text)
            st.write("Wyodrębniony tekst:", extracted_text)
            st.write("Przetłumaczony tekst:", translated_text)
        else:
            st.warning("Nie znaleziono tekstu do przetłumaczenia.")