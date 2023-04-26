import json
import librosa
import numpy as np
import gradio as gr
import tensorflow as tf

# setting the file paths
model_path = "audio_clf_model"
encoding_path = "label_encodings.json"
examples_path = "example_audios"

# loading the files
model = tf.keras.models.load_model(model_path)
classes = json.load(open(encoding_path, "r"))
labels = [classes[str(i)] for i in range(len(classes))]

# Load the model
def pre_processor(audio_path):

  # load the audio file
  x, sample_rate = librosa.load(audio_path)
  
  # feature extracting (mfccs is an aduio feature)
  mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
  feature = mfccs
  
  return feature

def clsf(audio_path):
  
  # extracting the features
  features = pre_processor(audio_path)
  print(len(features))

  # batching the data
  sample = np.expand_dims(features, axis=0)

  # predicting
  preds = model.predict(sample).flatten()
  
  # results
  confidences = {labels[i]: np.round(float(preds[i]), 3) for i in range(len(labels))}

  return confidences

# GUI Component
demo_params = {
    "fn":clsf, 
    "inputs":gr.Audio(source="upload", type="filepath"),
    "outputs": "label",
    #live=True,
    "examples": examples_path
}
demo = gr.Interface(**demo_params)

# Launching the demo
if __name__ == "__main__":
    demo.launch()