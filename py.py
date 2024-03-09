import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle
import os
import librosa
import librosa.display
from IPython.display import display
from IPython.display import Audio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
df = pd.read_csv(r"D:\bureau\BD&AI 1\s4\ML\ai project\Data\features_3_sec.csv")
print(df.head())
df = df.drop('filename', axis=1)  # Correction du paramètre 'label' à 'filename'

audio_recording = r"D:\bureau\BD&AI 1\s4\ML\ai project\Data\genres_original\country\country.00005.wav"  # Correction du chemin
data, sr = librosa.load(audio_recording)
print(type(data), type(sr))
print(librosa.load(audio_recording, sr=45600))
stft = librosa.stft(data)
stft_db = librosa.amplitude_to_db(np.abs(stft))
plt.figure(figsize=(14, 6))
librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.show()
spectral_rolloff = librosa.feature.spectral_rolloff(y=data, sr=sr)[0]
plt.figure(figsize=(7, 6))
librosa.display.waveshow(data, sr=sr, alpha=0.4, color='b')
plt.show()
stft = librosa.stft(data)
stft_db = librosa.amplitude_to_db(np.abs(stft))
plt.figure(figsize=(7, 6))
librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
start=1000
end=1200
plt.figure(figsize=(7, 6))
plt.plot(data[start:end], color="#2B4F72")
plt.figure()
class_list =df.iloc[:,-1]
convertor=LabelEncoder()
y=convertor.fit_transform(class_list)
np.save("class.npy", convertor.classes_) 
print(y)
print(df.iloc[:,:-1])
from sklearn.preprocessing import StandardScaler
from joblib import dump

fit = StandardScaler()
x = fit.fit_transform(np.array(df.iloc[:, :-1], dtype=float))

# Enregistrement des paramètres du scaler
scaler_params = {
    'mean': fit.mean_,
    'scale': fit.scale_
}

# Enregistrement du scaler avec joblib
dump(fit, 'scaler.joblib')

# Enregistrement des paramètres du scaler avec numpy
np.save("scaler.npy", scaler_params)

print(x.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print("Forme de X_train :", X_train.shape)
print("Forme de X_test :", X_test.shape)
print("Forme de y_train :", y_train.shape)
print("Forme de y_test :", y_test.shape)
from keras.models import Sequential
def trainModel(model,epochs,optimizer) :
    batch_size=128
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=epochs,batch_size=batch_size)
def plotValidate(history):
    print("validation accuarcy",max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12,6))
    plt.show()
import keras as k
model=k.models.Sequential([
    k.layers.Dense(512,activation='relu',input_shape=(X_train.shape[1],)),
    k.layers.Dropout(0.2),
    k.layers.Dense(256,activation='relu'),
    k.layers.Dropout(0.2),
    k.layers.Dense(128,activation='relu'),
    k.layers.Dropout(0.2),
    k.layers.Dense(64,activation='relu'),
    k.layers.Dropout(0.2),
    k.layers.Dense(10,activation='softmax'),  
    ]
    )
print(model.summary())
model_history=trainModel(model=model, epochs=600, optimizer='adam')
print(model_history)
test_loss,test_acc=model.evaluate(X_test,y_test,batch_size=128)
print(" the test loss is : ",test_loss)
print("test test accuaracy is :",test_acc)
model.save('my_model.keras')
epochs = np.arange(1, 601)
plt.subplot(1, 2, 1)
plt.plot(epochs, model_history.history['loss'], color='r')
plt.title("model loss(sparse categorical crossentropy)")
plt.xlabel('Epochs')
plt.ylabel('loss')

plt.subplot(1, 2, 2)  # Change here
plt.plot(epochs, model_history.history['accuracy'], color='g')
plt.title("model accuracy")
plt.xlabel('Epochs')
plt.ylabel('accuracy')

plt.tight_layout()
plt.show()
epochs = np.arange(1, 601)

plt.subplot(1, 2, 1)
plt.plot(epochs, model_history.history['loss'], color='r')
plt.title("Model Loss (Sparse Categorical Crossentropy)")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.tight_layout()

plt.subplot(1, 2, 2)  # Change here
plt.plot(epochs, model_history.history['accuracy'], color='g')
plt.title("Model Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.tight_layout()

plt.show()

from sklearn.metrics import confusion_matrix
pred=model.predict(X_test)
preds=[]
for i in pred:
    out=np.argmax(i)
    preds.append(out)
cm=confusion_matrix(y_test,preds)
plt.imshow(cm,interpolation='nearest',cmap=plt.cm.Blues)
plt.title('confusion matrix')
classes=convertor.classes_
tick_marks=np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks,classes)
fmt='d'
thresh=cm.max()/2.
for i , j in np.ndindex(cm.shape):
    plt.text(j,i,format(cm[i,j],fmt),
             horizontalalignment="center",

             color="white" if cm[i,j] >thresh else "black"
             )
plt.tight_layout()
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()  
#validation 
model.input_shape
def predict(model,X,Y):
    X=X[np.newaxis,...]
    prediction=model.predict(X)
    print(prediction)
    predicted_index=np.argmax(prediction,axis=1)
    print("expected index:",Y)
    print("predicted index: ",predicted_index)
#testing 
X=X_test[150]
Y=y_test[150]
predict(model, X, Y)
import tkinter as tk
from tkinter import filedialog
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from joblib import load
import librosa
import numpy as np

# Charger le modèle entraîné
model = keras.models.load_model("my_model.keras")

# Charger le convertisseur de classes et le scaler
le = LabelEncoder()
le.classes_ = np.load("class.npy", allow_pickle=True)
fit = load('scaler.joblib')


# creat a function to extract features from audio file
def extract_feature(filename):
    y, sr = librosa.load(filename, duration=30)
    length = len(y)
    


    chroma_stft_mean = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    chroma_stft_var = librosa.feature.chroma_stft(y=y, sr=sr).var()
    rms_mean = librosa.feature.rms(y=y).mean()
    rms_var = librosa.feature.rms(y=y).var()
    spectral_centroid_mean = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_centroid_var = librosa.feature.spectral_centroid(y=y, sr=sr).var()
    spectral_bandwidth_mean_mean = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_bandwidth_mean_var = librosa.feature.spectral_bandwidth(y=y, sr=sr).var()
    rolloff_mean = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    rolloff_var = librosa.feature.spectral_rolloff(y=y, sr=sr).var()
    zero_crossing_rate_mean = librosa.feature.zero_crossing_rate(y=y).mean()
    zero_crossing_rate_var = librosa.feature.zero_crossing_rate(y=y).var()

    y_harmonic = librosa.effects.harmonic(y)
    harmony_mean = np.mean(y_harmonic)
    harmony_var = np.var(y_harmonic)

    y_percussive = librosa.effects.percussive(y)
    percussive_mean = np.mean(y_percussive)
    percussive_var = np.var(y_percussive)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc1_mean = mfccs[0].mean()
    mfcc1_var = mfccs[0].var()
    mfcc2_mean = mfccs[1].mean()
    mfcc2_var = mfccs[1].var()            
    mfcc3_mean=mfccs[2].mean()
    mfcc3_var=mfccs[2].var()
    mfcc4_mean=mfccs[3].mean()
    mfcc4_var=mfccs[4].var()
    mfcc5_mean=mfccs[4].mean()
    mfcc5_var=mfccs[4].var()
    mfcc6_mean=mfccs[5].mean()
    mfcc6_var=mfccs[5].var()
    mfcc7_mean=mfccs[6].mean()
    mfcc7_var=mfccs[6].var()
    mfcc8_mean=mfccs[7].mean()
    mfcc8_var=mfccs[7].var()
    mfcc9_mean=mfccs[8].mean()
    mfcc9_var=mfccs[8].var()
    mfcc10_mean=mfccs[9].mean()
    mfcc10_var=mfccs[9].var()
    mfcc11_mean=mfccs[10].mean()
    mfcc11_var=mfccs[10].var()
    mfcc12_mean=mfccs[11].mean()
    mfcc12_var=mfccs[11].var()
    mfcc13_mean=mfccs[12].mean()
    mfcc13_var=mfccs[12].var()
    mfcc14_mean=mfccs[13].mean()
    mfcc4_var=mfccs[13].var()
    mfcc15_mean=mfccs[14].mean()
    mfcc15_var=mfccs[14].var()
    mfcc16_mean=mfccs[15].mean()
    mfcc16_var=mfccs[15].var()
    mfcc17_mean=mfccs[16].mean()
    mfcc17_var=mfccs[16].var()
    mfcc18_mean=mfccs[17].mean()
    mfcc18_var=mfccs[17].var()
    mfcc19_mean=mfccs[18].mean()
    mfcc19_var=mfccs[18].var()
    mfcc20_mean=mfccs[19].mean()
    mfcc20_var=mfccs[19].var()
    feature = np.array([length, chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, spectral_centroid_mean,
                        spectral_centroid_var, spectral_bandwidth_mean_mean, spectral_bandwidth_mean_var, rolloff_mean, rolloff_var,
                        zero_crossing_rate_mean, zero_crossing_rate_var, harmony_mean, harmony_var, percussive_mean, percussive_var, tempo,
                        mfcc1_mean, mfcc1_var, mfcc2_mean, mfcc2_var, mfcc3_mean, mfcc3_var, mfcc4_mean, mfcc4_var, mfcc5_mean, mfcc5_var,
                        mfcc6_mean, mfcc6_var, mfcc7_mean, mfcc7_var, mfcc8_mean, mfcc8_var, mfcc9_mean, mfcc9_var, mfcc10_mean, mfcc10_var,
                        mfcc11_mean, mfcc11_var, mfcc12_mean, mfcc12_var, mfcc13_mean, mfcc13_var, mfcc14_mean, mfcc4_var, mfcc15_mean,
                        mfcc15_var, mfcc16_mean, mfcc16_var, mfcc17_mean, mfcc17_var, mfcc18_mean, mfcc18_var, mfcc19_mean, mfcc19_var,
                        mfcc20_mean, mfcc20_var])

    return feature

# Créez une fonction pour classer un fichier audio
def classify_audio():
    filename = filedialog.askopenfilename()
    if filename:
        X = extract_feature(filename)
        X = np.expand_dims(X, axis=0)
        X = fit.transform(X)
        y_pred = model.predict(X)
        predicted_index = le.classes_[np.argmax(y_pred, axis=1)]
        predicted_genre = predicted_index[0]

        result_label.config(text='The genre for your music is ' + predicted_genre)

# Interface graphique
window = tk.Tk()
window.geometry("500x200")
window.title('Audio Genre Classification')

load_button = tk.Button(window, text='Load Audio File', command=classify_audio)
load_button.pack(pady=20)

result_label = tk.Label(window, text='', font=("Helvetica", 20))
result_label.pack()

window.mainloop()