
---

<div align="center">
  <a href="#">
    <img src="https://github.com/zakariaeyahya/Classification-des-Genres-Musicaux/assets/155691167/da44433d-d07a-44e9-9ef0-9269dde4dbb0" alt="Banner" width="720">
  </a>
  <div id="user-content-toc">
    <ul>
      <summary><h1 style="display: inline-block;">Prediction of Moroccan Music Genres</h1></summary>
    </ul>
  </div>

  <p>Classifying Moroccan music genres using deep learning models</p>
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
1. [Introduction](#introduction)
2. [Dataset](#data)
3. [Included Files](#included_files)
4. [Usage](#utilisation)
5. [Libraries Used](#bibliotheques)
6. [Future Work](#travaux_futurs)
7. [Conclusion](#conclusion)
8. [Contributors](#contributeurs)
9. [Contact](#contact)
10. [License](#licence)
<hr>

<a name="introduction"></a>
## 🔬 Introduction
This project aims to classify music genres using deep learning models. It involves creating and training Convolutional Neural Network (CNN) and Long Short-Term Memory Recurrent Neural Network (RNN-LSTM) models to predict music genres based on audio features extracted from audio files. The project also includes preprocessing scripts to extract relevant audio features and a Streamlit application for interactive genre prediction.

<a name="data"></a>
## 🗃️ Dataset
The dataset includes 8 types of Moroccan music, with 100 audio files of 30 seconds each for every type:
- Amazigh (Ahidous)
- Chaâbi
- Gnawa
- Malhun
- Andalusian Music
- Moroccan Rap and Hip-Hop
- Raï
- Reggada
- Sufi

<a name="included_files"></a>
## 📂 Included Files
- **cnn_model.ipynb**: Jupyter Notebook for creating and training the CNN model using TensorFlow/Keras.
- **RNN-LSTM.ipynb**: Jupyter Notebook for creating and training the RNN-LSTM model using TensorFlow/Keras.
- **preprocessing_ml.py**: Python script to extract audio features from audio files.
- **building_streamlit.py**: Python script to create a Streamlit application for interactive music genre prediction.

<a name="utilisation"></a>
## 🚀 Usage
To run this project, follow these steps:
1. **Preprocessing**: Run the `preprocessing_ml.py` script to extract audio features from your dataset.
2. **Model Training**: Train the CNN and RNN-LSTM models using the extracted audio features. This can be done by running the respective Jupyter notebooks (`cnn_model.ipynb` and `RNN-LSTM.ipynb`).
3. **Streamlit Application**: Deploy the Streamlit application by running the `building_streamlit.py` script. Users can then upload audio files and get predictions on the music genre.

<a name="bibliotheques"></a>
## 📚 Libraries Used
- **TensorFlow/Keras**: For creating and training deep learning models.
- **Librosa**: For extracting audio features from audio files.
- **Streamlit**: For creating interactive web applications.
- **PyDub**: For audio file manipulation and conversion.

<a name="travaux_futurs"></a>
## 🔮 Future Work
- Optimize models for better accuracy.
- Expand the dataset to include more diverse music genres.
- Incorporate other deep learning architectures for comparison.
- Improve the user interface and features of the Streamlit application.

<a name="conclusion"></a>
## 🏁 Conclusion
This project provides a comprehensive solution for preprocessing and classifying audio data for machine learning. Using the provided scripts, you can efficiently extract audio features, build a classification model, and create an interactive user interface for music genre prediction.

<a name="contributeurs"></a>
## 👥 Contributors
- **Salaheddine KAYOUH**: Developer and maintainer of the project.
- **Yahya Zakariae**: Developer and maintainer of the project.

<a name="contact"></a>
## 📬 Contact
For any questions or comments regarding this project, feel free to contact:
- **Yahya Zakariae**: [zakariae.yh@gmail.com](mailto:zakariae.yh@gmail.com) or [LinkedIn](https://www.linkedin.com/in/zakariae-yahya)
- **KAYOUH Salaheddine**: [salah.k2y07@gmail.com](mailto:salah.k2y07@gmail.com) or [LinkedIn](https://www.linkedin.com/in/salaheddine-kayouh-899b34235/)

<a name="licence"></a>
## 📄 License
This project is licensed under the MIT License.

Feel free to explore, experiment, and contribute to the project
