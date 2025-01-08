import streamlit as st
from lyricsgenius import Genius
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Ładowanie zmiennych środowiskowych z pliku .env
load_dotenv()

# Inicjalizacja Genius z tokenem
GENIUS_ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")  # Wczytanie tokena z .env
genius = Genius(GENIUS_ACCESS_TOKEN)

# Wybór modelu do tłumaczenia
model_name = "clui/opus-it-pl-v1"  # Model tłumaczenia z włoskiego na polski
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Tytuł aplikacji
st.title("Song Lyrics Translator")

# Wprowadzenie nazwy artysty
artist_name = st.text_input("Enter the artist's name:")

if st.button("Search"):
    if artist_name:
        # Wyszukiwanie artysty i pobieranie jego utworów
        artist = genius.search_artist(artist_name, max_songs=3)
        
        if artist:
            # Wyświetlanie tytułów utworów
            song_titles = [song.title for song in artist.songs]
            selected_song = st.selectbox("Select a song:", song_titles)

            if selected_song:
                # Pobieranie tekstu wybranego utworu
                song = genius.search_song(selected_song, artist_name)
                lyrics = song.lyrics
                lyrics= lyrics[:200]
                
                # Wyświetlanie tekstu piosenki
                st.subheader("Original Lyrics:")
                st.write(lyrics)

                # Tłumaczenie tekstu
                inputs = tokenizer(lyrics, return_tensors="pt", padding=True)
                translated_outputs = model.generate(**inputs)
                translated_lyrics = tokenizer.decode(translated_outputs[0], skip_special_tokens=True)

                # Wyświetlanie przetłumaczonego tekstu
                st.subheader("Translated Lyrics:")
                st.write(translated_lyrics)
        else:
            st.warning("No songs found for this artist.")
    else:
        st.warning("Please enter an artist's name.")