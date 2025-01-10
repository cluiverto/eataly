import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from lyricsgenius import Genius
import os

# Wybór modelu do tłumaczenia
model_name = "clui/opus-it-pl-v1"  # Model tłumaczenia z włoskiego na polski
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tytuł aplikacji
st.title("Neural Notes")

# Ustawienia tła
st.markdown(
    """
    <style>
    body {
        background-color: #f0f4f8; /* Kolor tła podobny do DeepL */
        color: #333333; /* Kolor tekstu */
    }
    .stButton>button {
        background-color: #4CAF50; /* Zielony przycisk */
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Inicjalizacja Genius z tokenem
GENIUS_ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")  # Wczytanie tokena z .env
genius = Genius(GENIUS_ACCESS_TOKEN)

# Sekcja Genius Mode
st.subheader("Genius Mode")
st.write("")

# Wprowadzenie nazwy artysty
artist_name = st.text_input("Wprowadź nazwisko artysty:")

if st.button("Szukaj"):
    if artist_name:
        # Wyszukiwanie artysty i pobieranie jego utworów
        artist = genius.search_artist(artist_name, max_songs=3)
        
        if artist:
            # Wyświetlanie tytułów utworów
            song_titles = [song.title for song in artist.songs]
            selected_song = st.selectbox("Wybierz utwór:", song_titles)

            if selected_song:
                # Pobieranie tekstu wybranego utworu
                song = genius.search_song(selected_song, artist_name)
                lyrics = song.lyrics.splitlines()  # Podział tekstu na linie
                
                # Wyświetlanie tekstu piosenki w dwóch kolumnach
                col1, col2 = st.columns(2)  # Tworzenie dwóch kolumn
                
                with col1:
                    st.subheader("Oryginalne linie:")
                    for line in lyrics:
                        st.write(line)

                # Tłumaczenie każdej linii
                translated_lines = []
                for line in lyrics:
                    if line.strip():  # Sprawdzenie, czy linia nie jest pusta
                        inputs = tokenizer(line, return_tensors="pt", padding=True)
                        translated_outputs = model.generate(**inputs)
                        translated_text = tokenizer.decode(translated_outputs[0], skip_special_tokens=True)
                        translated_lines.append(translated_text)

                with col2:
                    st.subheader("Przetłumaczone linie:")
                    for translated in translated_lines:
                        st.write(translated)
        else:
            st.warning("Nie znaleziono utworów dla tego artysty.")
    else:
        st.warning("Proszę wpisać nazwisko artysty.")