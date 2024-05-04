# Prédiction des Genres de Musique Marocaine
#  Introduction
Ce projet vise à extraire et à classifier des caractéristiques audio à partir de fichiers audio. Deux scripts principaux sont utilisés : preprocessing_ml.py pour l'extraction de caractéristiques et la création d'un fichier JSON, et classification_ml.py pour la construction, l'entraînement et l'évaluation d'un modèle de classification.
# Fichier de Données
Le fichier de données comprend 8 types de musiques marocaines, avec 100 fichiers audio de 30 secondes pour chaque type :
"Amazigh (Ahidous)"
"Chaâbi"
"Gnawa"
"Malhun"
"Musique Andalouse"
"Rap et Hip-Hop Marocain"
"Raï"
"Reggada"
"Sufi" 
#  Script preprocessing_ml.py
Ce script est responsable de l'extraction de caractéristiques audio à partir de fichiers audio et de leur stockage dans un fichier JSON.

## Fonctions Principales
extract_all_features(filename): Extrait les caractéristiques audio d'un fichier spécifié.
save_mfcc(dataset_path, json_path, num_segments): Divise les fichiers audio en segments, extrait les caractéristiques de chaque segment et les sauvegarde dans un fichier JSON.
check_data_integrity(json_file): Vérifie l'intégrité des données extraites.
# Script classification_ml.py
Ce script implémente la construction, l'entraînement et l'évaluation d'un modèle de classification à l'aide des caractéristiques extraites.

## Fonctions Principales
load_data(data_path): Charge les données extraites à partir du fichier JSON.
plot_history(history): Trace les graphiques d'entraînement du modèle.
prepare_datasets(test_size, validation_size): Prépare les ensembles de données pour l'entraînement, la validation et les tests.
build_model(input_shape): Construit le modèle de classification.
predict(model, X, y): Effectue une prédiction sur un échantillon de données.
Sauvegarde et évaluation du modèle.
# Exécution
Pour exécuter ce projet, suivez ces étapes :

Exécutez le script preprocessing_ml.py pour extraire les caractéristiques audio et les sauvegarder.
Exécutez le script classification_ml.py pour construire, entraîner et évaluer le modèle de classification.
Assurez-vous d'avoir les dépendances requises installées avant d'exécuter les scripts.

# Conclusion
Ce projet offre une solution complète pour le prétraitement et la classification de données audio pour l'apprentissage machine. En utilisant les scripts fournis, vous pouvez extraire efficacement des caractéristiques audio et construire un modèle de classification pour analyser et catégoriser des données audio.
# Licence
Ce projet est sous licence MIT.

N'hésitez pas à explorer, expérimenter et contribuer au projet. !
