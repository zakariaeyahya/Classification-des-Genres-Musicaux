# Classification des Genres Musicaux par Apprentissage Automatique
Ce projet utilise l'apprentissage automatique pour classer les genres musicaux à partir de fichiers audio. Il comprend un script Python qui extrait des caractéristiques audio, normalise les données, construit un modèle neuronal profond avec TensorFlow, et offre une interface utilisateur graphique pour la classification en temps réel.

# Instructions d'utilisation
Installer les dépendances

Assurez-vous d'avoir les bibliothèques nécessaires installées en exécutant la commande suivante :

# Copy code
pip install pandas numpy matplotlib scipy librosa tensorflow scikit-learn joblib
Exécution du script principal

Exécutez le script principal audio_classification.py pour former le modèle et afficher les résultats.

bash
# Copy code
python audio_classification.py
Interface Utilisateur Graphique (GUI)

Pour utiliser l'interface utilisateur graphique pour la classification en temps réel, exécutez le script gui_classification.py :

bash
Copy code
python gui_classification.py
Fichiers Pré-Enregistrés

Le modèle a été sauvegardé sous le nom my_model.keras. Vous pouvez également trouver le convertisseur de classes sous le nom class.npy et le scaler sous le nom scaler.joblib.

# Structure du Projet
audio_classification.py: Le script principal pour le traitement audio, l'apprentissage automatique, et l'évaluation du modèle.

gui_classification.py: Script pour l'interface utilisateur graphique (GUI) permettant la classification en temps réel.

README.md: Documentation du projet.

Data/: Dossier contenant les données d'entraînement et les caractéristiques extraites.

models/: Dossier contenant les modèles entraînés et les paramètres nécessaires à la classification.
