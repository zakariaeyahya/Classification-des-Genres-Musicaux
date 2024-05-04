# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:09:49 2024

@author: salah
"""

import streamlit as st
import os
import librosa
import numpy as np
import json
import base64
from pydub import AudioSegment
from tensorflow.keras.models import load_model


# Application Streamlit
st.title("Prétraitement et Prédiction des fichiers audio")

uploaded_file = st.file_uploader("Choisissez un fichier audio", type=["mp3", "wav"])
# Charger les modèles
# Charger les modèles
model1 = load_model(r"D:\bureau\BD&AI 1\s4\ML\machine learning\xapp\model.h5")


# Liste des genres musicaux 
genres = [
    ('Amazigh (Ahidous)'),
    ('Chaâbi'),
    ('Gnawa'),
    ('Malhun'),
    ('Musique Andalouse'),
    ('Rap et Hip-Hop Marocain'),
    ('Raï'),
    ('Reggada'),
    ('Sufi')
]



# Fonction pour convertir MP3 en WAV
def convert_to_wav(mp3_file_path):
    audio_segment = AudioSegment.from_file(mp3_file_path, format="mp3")
    wav_file_path = os.path.splitext(mp3_file_path)[0] + ".wav"
    audio_segment.export(wav_file_path, format="wav")
    return wav_file_path


def extract_all_features(filename):
    y, sr = librosa.load(filename, duration=30)
    length = len(y)
    
    # Calculer les caractéristiques RMS
    rms_mean = librosa.feature.rms(y=y).mean()
    rms_var = librosa.feature.rms(y=y).var()
    
    # Calculer les caractéristiques du centroïde spectral
    spectral_centroid_mean = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_centroid_var = librosa.feature.spectral_centroid(y=y, sr=sr).var()
    
    # Calculer les caractéristiques de la largeur de bande spectrale
    spectral_bandwidth_mean_mean = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_bandwidth_mean_var = librosa.feature.spectral_bandwidth(y=y, sr=sr).var()
    
    # Calculer les caractéristiques du rolloff spectral
    rolloff_mean = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    rolloff_var = librosa.feature.spectral_rolloff(y=y, sr=sr).var()
    
    # Calculer les caractéristiques du taux de passage par zéro
    zero_crossing_rate_mean = librosa.feature.zero_crossing_rate(y=y).mean()
    zero_crossing_rate_var = librosa.feature.zero_crossing_rate(y=y).var()
    
    # Calculer les caractéristiques de l'harmonie
    y_harmonic = librosa.effects.harmonic(y)
    harmony_mean = np.mean(y_harmonic)
    harmony_var = np.var(y_harmonic)
    
    # Calculer les caractéristiques du percussif
    y_percussive = librosa.effects.percussive(y)
    percussive_mean = np.mean(y_percussive)
    percussive_var = np.var(y_percussive)
    
    # Calculer le tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Calculer les coefficients MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_vars = np.var(mfccs, axis=1)
    
    # Concaténer toutes les caractéristiques extraites
    feature = np.concatenate(([length], [rms_mean], [rms_var], [spectral_centroid_mean], [spectral_centroid_var],
                              [spectral_bandwidth_mean_mean], [spectral_bandwidth_mean_var], [rolloff_mean],
                              [rolloff_var], [zero_crossing_rate_mean], [zero_crossing_rate_var], [harmony_mean],
                              [harmony_var], [percussive_mean], [percussive_var], [tempo], mfcc_means, mfcc_vars))

    return feature



if uploaded_file:
    if not uploaded_file:
        st.warning("Veuillez télécharger un fichier audio.")
        

    # Chemin temporaire
    temp_path = os.path.join("/tmp", uploaded_file.name)

    # Sauvegarder le fichier téléchargé
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getvalue())

    # Convertir en WAV si nécessaire
    if uploaded_file.name.lower().endswith(".mp3"):
        temp_path = convert_to_wav(temp_path)

    # Afficher le fichier audio pour lecture
    st.audio(uploaded_file, format="audio/mp3")

    # Extraire les caractéristiques audio
    audio_features = extract_all_features(temp_path)
    
    # Ajuster la forme des caractéristiques audio
    audio_features = np.expand_dims(audio_features, axis=0)  # Ajoutez cette ligne

    # Prédire le genre musical à l'aide du modèle
    prediction = model1.predict(np.expand_dims(audio_features, axis=0))
    predicted_genre_index = np.argmax(prediction)
    predicted_genre = genres[predicted_genre_index]

    # Afficher le résultat
    st.write(f"Genre musical prédit : {predicted_genre}")
