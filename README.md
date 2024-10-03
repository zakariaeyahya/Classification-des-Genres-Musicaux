<div align="center">
  <a href="#">
    <img src="https://github.com/zakariaeyahya/Classification-des-Genres-Musicaux/assets/155691167/da44433d-d07a-44e9-9ef0-9269dde4dbb0" alt="Banner" width="720">
  </a>

  <div id="user-content-toc">
    <ul>
      <summary><h1 style="display: inline-block;">Prédiction des Genres de Musique Marocaine</h1></summary>
    </ul>
  </div>
  
  <p>Classifier le genre de la musique marocaine en utilisant des modèles d'apprentissage profond</p>
    <a href="#" target="_blank">Live Preview</a>
    🛸
    <a href="#" target="_blank">Data on Kaggle</a>
    🌪️
    <a href="#" target="_blank">Request Feature</a>
</div>
<br>
<div align="center">
      <a href="#"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a>
      <img src="https://img.shields.io/github/stars/zakariaeyahya/Classification-des-Genres-Musicaux?color=blue&style=social"/>
      <a href="https://youtu.be/iYvwxq49_D0"><img src="https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white"/></a>
</div>

## 📝 Table of Contents

1. [ Introduction ](#introduction)
2. [ Fichier de Données ](#data)
3. [ Fichiers Inclus ](#included_files)
4. [ Utilisation ](#utilisation)
5. [ Bibliothèques Utilisées ](#bibliotheques)
6. [ Travaux Futurs ](#travaux_futurs)
7. [ Conclusion ](#conclusion)
8. [ Contributeurs ](#contributeurs)
9. [ Contact ](#contact)
10. [ Licence ](#licence)
<hr>

<a name="introduction"></a>
## 🔬 Introduction

<p>Ce projet vise à classifier le genre de la musique en utilisant des modèles d'apprentissage profond. Il comprend la création et l'entraînement de modèles de Réseaux de Neurones Convolutifs (CNN) et de Réseaux de Neurones Récurrents avec Mémoire à Long Terme (RNN-LSTM) pour prédire le genre de la musique en fonction des caractéristiques audio extraites des fichiers audio. Le projet comprend également des scripts de prétraitement pour extraire les caractéristiques pertinentes des fichiers audio et une application Streamlit pour une prédiction interactive du genre musical.</p>

<a name="data"></a>
## 🗃️ Fichier de Données

<p>Le fichier de données comprend 8 types de musiques marocaines, avec 100 fichiers audio de 30 secondes pour chaque type :</p>
<ul>
    <li>Amazigh (Ahidous)</li>
    <li>Chaâbi</li>
    <li>Gnawa</li>
    <li>Malhun</li>
    <li>Musique Andalouse</li>
    <li>Rap et Hip-Hop Marocain</li>
    <li>Raï</li>
    <li>Reggada</li>
    <li>Sufi</li>
</ul>

<a name="included_files"></a>
## 📂 Fichiers Inclus

<ul>
    <li><strong>cnn_model.ipynb</strong> : Notebook Jupyter pour la création et l'entraînement du modèle CNN en utilisant TensorFlow/Keras.</li>
    <li><strong>RNN-LSTM.ipynb</strong> : Notebook Jupyter pour la création et l'entraînement du modèle RNN-LSTM en utilisant TensorFlow/Keras.</li>
    <li><strong>preprocessing_ml.py</strong> : Script Python pour extraire les caractéristiques audio des fichiers audio.</li>
    <li><strong>building_streamlit.py</strong> : Script Python pour créer une application Streamlit pour une prédiction interactive du genre musical.</li>
</ul>

<a name="utilisation"></a>
## 🚀 Utilisation

<p>Pour exécuter ce projet, suivez ces étapes :</p>
<ol>
    <li><strong>Prétraitement</strong> : Exécutez le script <code>preprocessing_ml.py</code> pour extraire les caractéristiques audio de votre ensemble de données.</li>
    <li><strong>Entraînement du Modèle</strong> : Entraînez les modèles CNN et RNN-LSTM en utilisant les caractéristiques audio extraites. Cela peut être fait en exécutant les notebooks Jupyter respectifs (<code>cnn_model.ipynb</code> et <code>RNN-LSTM.ipynb</code>).</li>
    <li><strong>Application Streamlit</strong> : Déployez l'application Streamlit en exécutant le script <code>building_streamlit.py</code>. Les utilisateurs peuvent ensuite télécharger des fichiers audio et obtenir des prédictions sur le genre musical.</li>
</ol>

<a name="bibliotheques"></a>
## 📚 Bibliothèques Utilisées

<ul>
    <li><strong>TensorFlow/Keras</strong> : Pour la création et l'entraînement des modèles d'apprentissage profond.</li>
    <li><strong>Librosa</strong> : Pour extraire les caractéristiques audio des fichiers audio.</li>
    <li><strong>Streamlit</strong> : Pour créer des applications web interactives.</li>
    <li><strong>PyDub</strong> : Pour la manipulation et la conversion des fichiers audio.</li>
</ul>

<a name="travaux_futurs"></a>
## 🔮 Travaux Futurs

<ul>
    <li>Optimiser les modèles pour une meilleure précision.</li>
    <li>Étendre l'ensemble de données pour inclure davantage de genres musicaux diversifiés.</li>
    <li>Incorporer d'autres architectures de deep learning pour la comparaison.</li>
    <li>Améliorer l'interface utilisateur et les fonctionnalités de l'application Streamlit.</li>
</ul>

<a name="conclusion"></a>
## 🏁 Conclusion

<p>Ce projet offre une solution complète pour le prétraitement et la classification de données audio pour l'apprentissage machine. En utilisant les scripts fournis, vous pouvez extraire efficacement des caractéristiques audio, construire un modèle de classification et créer une interface utilisateur interactive pour la prédiction des genres musicaux.</p>

<a name="contributeurs"></a>
## 👥 Contributeurs

<ul>
    <li><strong>Salaheddine KAYOUH</strong> : Développeur et mainteneur du projet.</li>
    <li><strong>Yahya Zakariae</strong> : Développeur et mainteneur du projet.</li>
</ul>

<a name="contact"></a>
## 📬 Contact

<p>Pour toute question ou commentaire concernant ce projet, n'hésitez pas à contacter :</p>
<ul>
    <li><strong>Yahya Zakariae</strong> : <a href="mailto:zakariae.yh@gmail.com">zakariae.yh@gmail.com</a> ou <a href="https://www.linkedin.com/in/zakariae-yahya">LinkedIn</a></li>
    <li><strong>KAYOUH Salaheddine</strong> : <a href="mailto:salah.k2y07@gmail.com">salah.k2y07@gmail.com</a> ou <a href="https://www.linkedin.com/in/salaheddine-kayouh-899b34235/">LinkedIn</a></li>
</ul>

<a name="licence"></a>
## 📄 Licence

<p>Ce projet est sous licence MIT.</p>

<p>N'hésitez pas à explorer, expérimenter et contribuer au projet !</p>
