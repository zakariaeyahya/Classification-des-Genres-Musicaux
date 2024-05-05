import json
import os
import math
import librosa
import numpy as np

DATASET_PATH = r'D:\bureau\BD&AI 1\s4\ML\machine learning\segments_audio'
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

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

def save_mfcc(dataset_path, json_path, num_segments=5):
    """Extracts MFCCs from music dataset and saves them into a json file along with genre labels.
    :param dataset_path (str): Path to dataset
    :param json_path (str): Path to json file used to save MFCCs
    :param num_segments (int): Number of segments we want to divide sample tracks into
    :return:
    """
    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "features": []
    }
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    # loop through all genre sub-folder
    for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):
        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:
            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = os.path.basename(dirpath)
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))
            # Get the index of the label in the mapping list
            label_id = data["mapping"].index(semantic_label)
            # process all audio files in genre sub-dir
            for f in filenames:
                # load audio file
                file_path = os.path.join(dirpath, f)
                # process all segments of audio file
                for d in range(num_segments):
                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment
                    # extract features
                    feature = extract_all_features(file_path)
                    # Ajouter les caractéristiques extraites à la liste des features
                    data["features"].append(feature.tolist())  # Convertir ndarray en list
                    data["labels"].append(label_id)  # Use the label_id obtained from mapping index
                    print("{}, segment:{}".format(file_path, d + 1))
    # save features to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def check_data_integrity(json_file):
    # Charger le fichier JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    # Vérifier la longueur des listes d'étiquettes et de caractéristiques MFCC
    num_mapping = len(data['mapping'])
    num_labels = len(data['labels'])
    num_features = len(data['features'])
    print(f"Nombre de catégories dans le mapping : {num_mapping}")
    print(f"Nombre d'étiquettes : {num_labels}")
    print(f"Nombre de vecteurs de caractéristiques : {num_features}")
    # Vérifier si le nombre d'étiquettes correspond au nombre de vecteurs de caractéristiques
    if num_labels != num_features:
        print("Erreur : Le nombre d'étiquettes ne correspond pas au nombre de vecteurs de caractéristiques")
    # Vérifier si toutes les étiquettes sont des entiers
    if not all(isinstance(label, int) for label in data['labels']):
        print("Erreur : Les étiquettes ne sont pas toutes des entiers")
    # Vérifier si les valeurs des étiquettes sont valides
    max_label = max(data['labels'])
    min_label = min(data['labels'])
    if min_label < 0 or max_label >= num_mapping:
        print("Erreur : Les valeurs des étiquettes ne sont pas valides")
    print("Vérification terminée.")

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
    check_data_integrity(JSON_PATH)
