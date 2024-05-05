# Prédiction des Genres de Musique Marocaine
![Banner landscape travel geometric blue (2)](https://github.com/zakariaeyahya/Classification-des-Genres-Musicaux/assets/155691167/da44433d-d07a-44e9-9ef0-9269dde4dbb0)
#  Introduction
Ce projet vise à classifier le genre de la musique en utilisant des modèles d'apprentissage profond. Il comprend la création et l'entraînement de modèles de Réseaux de Neurones Convolutifs (CNN) et de Réseaux de Neurones Récurrents avec Mémoire à Long Terme (RNN-LSTM) pour prédire le genre de la musique en fonction des caractéristiques audio extraites des fichiers audio. Le projet comprend également des scripts de prétraitement pour extraire les caractéristiques pertinentes des fichiers audio et une application Streamlit pour une prédiction interactive du genre musical.

# Fichier de Données (data)
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
# Fichiers Inclus
cnn_model.ipynb : Notebook Jupyter contenant le code pour la création et l'entraînement du modèle CNN en utilisant TensorFlow/Keras.
RNN-LSTM.ipynb : Notebook Jupyter contenant le code pour la création et l'entraînement du modèle RNN-LSTM en utilisant TensorFlow/Keras.
preprocessing_ml.py : Script Python pour extraire les caractéristiques audio des fichiers audio en prétraitement pour l'entraînement du modèle.
building_streamlit.py : Script Python pour créer une application Streamlit pour une prédiction interactive du genre musical.
 
# Utilisation
Pour exécuter ce projet, suivez ces étapes :

Prétraitement : Exécutez le script preprocessing_ml.py pour extraire les caractéristiques audio de votre ensemble de données de fichiers audio.
Entraînement du Modèle : Entraînez les modèles CNN et RNN-LSTM en utilisant les caractéristiques audio extraites. Cela peut être fait en exécutant les notebooks Jupyter respectifs (cnn_model.ipynb et RNN-LSTM.ipynb).
Application Streamlit : Déployez l'application Streamlit en exécutant le script building_streamlit.py. Les utilisateurs peuvent ensuite télécharger des fichiers audio vers l'application et obtenir des prédictions sur le genre de musique.
# Bibliothèques Utilisées
TensorFlow/Keras : Pour la création et l'entraînement des modèles d'apprentissage profond.
Librosa : Pour extraire les caractéristiques audio des fichiers audio.
Streamlit : Pour créer des applications web interactives.
PyDub : Pour la manipulation et la conversion des fichiers audio.

# Travaux Futurs
Les développements futurs pour ce projet pourraient inclure :

L'optimisation des modèles pour une meilleure précision.
L'expansion de l'ensemble de données pour inclure davantage de genres musicaux diversifiés.
L'incorporation d'autres architectures de deep learning pour la comparaison.
L'amélioration de l'interface utilisateur et des fonctionnalités de l'application Streamlit.

# Travaux Futurs
Les développements futurs pour ce projet pourraient inclure :

L'optimisation des modèles pour une meilleure précision.
L'expansion de l'ensemble de données pour inclure davantage de genres musicaux diversifiés.
L'incorporation d'autres architectures de deep learning pour la comparaison.
L'amélioration de l'interface utilisateur et des fonctionnalités de l'application Streamlit.

# Conclusion
Ce projet offre une solution complète pour le prétraitement et la classification de données audio pour l'apprentissage machine. En utilisant les scripts fournis, vous pouvez extraire efficacement des caractéristiques audio, construire un modèle de classification et créer une interface utilisateur interactive pour la prédiction des genres musicaux.
# Contributeurs
Salaheddine KAYOUH : Développeur et mainteneur du projet.
Yahya Zakariae : Développeur et mainteneur du projet.
#Contact
Pour toute question ou commentaire concernant ce projet, n'hésitez pas à contacter :
Yahya Zakariae : zakariae.yh@gmail.com ou www.linkedin.com/in/zakariae-yahya
 KAYOUH Salaheddine :salah.k2y07@gmail.com ou https://www.linkedin.com/in/salaheddine-kayouh-899b34235/

# Licence
Ce projet est sous licence MIT.

N'hésitez pas à explorer, expérimenter et contribuer au projet. !
