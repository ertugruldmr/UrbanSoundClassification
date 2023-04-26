<div style="position: absolute; top: 0; right: 0;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

# Urban Sound Classification
 
## __Table Of Content__
- (A) [__Brief__](#brief)
  - [__Project__](#project)
  - [__Data__](#data)
  - [__Demo__](#demo) -> [Live Demo](https://huggingface.co/spaces/ErtugrulDemir/UrbanSoundClassification
)
  - [__Study__](#problemgoal-and-solving-approach) -> [Colab](https://colab.research.google.com/drive/1v6K0oUGeaKveOnO4cFAy-tfQxs29l0e5)
  - [__Results__](#results)
- (B) [__Detailed__](#Details)
  - [__Abstract__](#abstract)
  - [__Explanation of the study__](#explanation-of-the-study)
    - [__(A) Dependencies__](#a-dependencies)
    - [__(B) Dataset__](#b-dataset)
    - [__(C) Modelling__](#c-modelling)
    - [__(D) Deployment as web demo app__](#g-deployment-as-web-demo-app)
  - [__Licance__](#license)
  - [__Connection Links__](#connection-links)

## __Brief__ 

### __Project__ 
- This is a __Multi Variable Classification__ project on audio data  with  custom __deep learning model__. The project uses the  [__UrbanSound8K__](https://www.kaggle.com/datasets/chrisfilo/urbansound8k?datasetId=500970&sortBy=voteCount&select=UrbanSound8K.csv) to __classify__ the audio file into corresponding class.
- The __goal__ is build a deep learning model that accurately __classify__ the audio file into corresponding class. There are 10 classes.
- The performance of the model is evaluated using several __metrics__ loss.

#### __Overview__
- This project involves building a deep learning model to classify the audio files into one of the corresponding 10 classes. The dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes, the classes are: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music.  The models selected according to model tuning results, the progress optimized respectively the previous tune results. The project uses Python and several popular libraries such as Pandas, NumPy, tensorflow.

#### __Demo__

<div align="left">
  <table>
    <tr>
    <td>
        <a target="_blank" href="https://huggingface.co/spaces/ErtugrulDemir/UrbanSoundClassification
" height="30"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30">[Demo app] HF Space</a>
      </td>
      <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1UH7wd-05I7otci-GcXrQwAKbwX9WeaT7"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Demo app] Run in Colab</a>
      </td>
      <td>
        <a target="_blank" href="https://github.com/ertugruldmr/UrbanSoundClassification/blob/main/study.ipynb
"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">[Traning pipeline] source on GitHub</a>
      </td>
    <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1v6K0oUGeaKveOnO4cFAy-tfQxs29l0e5"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Traning pipeline] Run in Colab</a>
      </td>
    </tr>
  </table>
</div>


- Description
    -  __classify audio data__ into one of the corresponding 10 class.
    - __Usage__: 
      - upload or select an audio data then clikc the submit button for classification
- Embedded [Demo](https://huggingface.co/spaces/ErtugrulDemir/UrbanSoundClassification
) window from HuggingFace Space
    

<iframe
	src="https://huggingface.co/spaces/ErtugrulDemir/UrbanSoundClassification
"
	frameborder="0"
	width="850"
	height="450"
></iframe>

#### __Data__
- The [__UrbanSound8K__](https://www.kaggle.com/datasets/chrisfilo/urbansound8k?datasetId=500970&sortBy=voteCount&select=UrbanSound8K.csv) from kaggle dataset api.
- The dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes,
- This 10 classes are: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music.

#### Problem, Goal and Solving approach
- TThis is a __Multi Variable Classification__ problem  that uses the  [__UrbanSound8K__](https://www.kaggle.com/datasets/chrisfilo/urbansound8k?datasetId=500970&sortBy=voteCount&select=UrbanSound8K.csv)  to __classify the audio data__ into one of the 10 class.
- The __goal__ is build a deep learning  model that accurately __classify the audio data__ into one of the 10 class.
- __Solving approach__ is that using the supervised deep learning models. Basic Custom Fully Connected Classifier model is used for audio data classification.

#### Study
The project aimed classifying the audio data into one of the 10 classes using deep learning model architecture. The study includes following chapters.
- __(A) Dependencies__: Installations and imports of the libraries.
- __(B) Dataset__: Downloading and loading the dataset. Preparing the dataset from official website. Configurating the dataset performance and related pre-processes. 
- __(C) Preprocessing__: loading the data convenient form, analyzing the data, exploring the data, implementing feature extraction as mean of mfcc methodconfigurating the dataset object, batching, performance setting, visualizating, Implementing the audio data related processes.
- __(D) Modelling__:
  - Model Architecture
    - Custom Fully Connected Dense neural network model used to classify audio data into one of the 10 classes.
  - Training
    - Callbakcs and trainin params are setted. some of the callbacks are EarlyStopping, ModelCheckpoint, Tensorboard etc....  
    - training history
        <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/training_hist.png" style="width: 300px; height: 200px;">
        </div>
  - Saving the model
    - Saved the model as tensorflow saved model format.
- __(E) Deployment as web demo app__: Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.

#### results
- The final model is __Custom Fuylly Connected Dense Classifier Model__   because of the results and less complexity.
  -  Custom Fuylly Connected Dense Classifier Model
        <table><tr><th>Model Results </th><th></th></tr><tr><td>
  |   | loss  | val_loss  | accuracy  |  val_accuracy  |
  |---|-------|-------|-------|-------|
  |   |  0.7772 |0.5873| 0.7394|0.8168|
    </td></tr></table>

## Details

### Abstract
- [__UrbanSound8K__](https://www.kaggle.com/datasets/chrisfilo/urbansound8k?datasetId=500970&sortBy=voteCount&select=UrbanSound8K.csv)  is used to classify audio data into one of the 10 classes. The dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes. This 10 classes are: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music. The goal is build a deep learning model that accurately classify audio data one of the 10 classes using through custom fully connected dense classifer as deep learning algorithms via related training approachs such as  pretrained state of art models.The study includes creating the environment, getting the data, preprocessing the data, exploring the data, mormalizing the audio data, reforming the data to convenient notation, feature extraction as mfcc, configurating the dataset object, batching, performance setting, visualizating, modelling the data, saving the results, deployment as demo app. Training phase of the models implemented through tensorflow callbacks. After the custom model traininigs, transfer learning and fine tuning approaches are implemented. Selected the basic and more succesful when comparet between other models, final model is custom fully connected dense classifier model. __Custom Fully Connected Dense Classifier__ model  has __0.7772__ loss, __0.5873__ validation, __0.7394__ accuracy, __0.8168__ validation accuracy  other metrics are also found the results section. Created a demo at the demo app section and served on huggingface space.  

### File Structures

- File Structure Tree
```bash
├── demo_app
│   ├── app.py
│   ├── audio_clf_model
│   ├── example_audios
│   ├── label_encodings.json
│   └── requirements.txt
├── docs
│   └── images
├── env
│   ├── env_installation.md
│   └── requirements.txt
├── LICENSE
├── readme.md
└── study.ipynb
```
- Description of the files
  - demo_app/
    - Includes the demo web app files, it has the all the requirements in the folder so it can serve on anywhere.
  - demo_app/audio_clf_model:
    - Custom Fully Connected Dense Classifier Model for classifying the audio data into one of the 10 classes.
  - demo_app/example_audios
    - It includes test case files. 
  - demo_app/label_encodings
    - It is the encoding map of the target classes.
  - docs/
    - Includes the documents about results and presentations
  - env/
    - It includes the training environmet related files. these are required when you run the study.ipynb file.
  - LICENSE.txt
    - It is the pure apache 2.0 licence. It isn't edited.
  - readme.md
    - It includes all the explanations about the project
  - study.ipynb
    - It is all the studies about solving the problem which reason of the dataset existance. 
  - requirements.txt
    - It includes the library dependencies of the study.   

### Explanation of the Study
#### __(A) Dependencies__:
  - There is a third-part installation as kaggle dataset api the other requirements will be alread installed while creating the environment. Just follow the code order to satisfy requirements. You can create an environment via env/requirements.txt. Create a virtual environment then use hte following code. It is enough to satisfy the requirements for runing the study.ipynb which training pipeline.
  - Dataset can download from kaggle dataset.
#### __(B) Dataset__: 
  - Downloading the [__UrbanSound8K__](https://www.kaggle.com/datasets/chrisfilo/urbansound8k?datasetId=500970&sortBy=voteCount&select=UrbanSound8K.csv) via kaggle dataset api. 
  - The dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes,
  - This 10 classes are: air_conditioner, car_horn, children_playing, dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, and street_music.
  - Example data
      <div style="display:flex; justify-content: center; align-items:center;">
        <img src="docs/images/example_data.png" style="width: 300px; height: 200px;">
      </div>
  - Target Distirbutions
      <div style="display:flex; justify-content: center; align-items:center;">
        <img src="docs/images/target_distribution.png" style="width: 300px; height: 200px;">
      </div>

#### __(C) Modelling__: 
  - The processes are below:
    - Model Architecture
      - Custom Fully Connected Dense neural network model used to classify audio data into one of the 10 classes.
      - Architecture
        <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/AudioClassifierArchitecture.png" style="width: 300px; height: 700px;">
        </div>
      - Training
        - Callbakcs and trainin params are setted. some of the callbacks are EarlyStopping, ModelCheckpoint, Tensorboard etc....  
        - training history
          <div style="display:flex; justify-content: center; align-items:center;">
            <img src="docs/images/training_hist.png" style="width: 300px; height: 200px;">
          </div>
    - Saving the model
      - Saved the model as tensorflow saved model format.
  - __(E) Deployment as web demo app__: Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.

  #### results
- The final model is __Custom Fuylly Connected Dense Classifier Model__   because of the results and less complexity.
  -  Custom Fuylly Connected Dense Classifier Model
        <table><tr><th>Model Results </th><th></th></tr><tr><td>
  |   | loss  | val_loss  | accuracy  |  val_accuracy  |
  |---|-------|-------|-------|-------|
  |   |  0.7772 |0.5873| 0.7394|0.8168|
    </td></tr></table>
    - Saving the project and demo studies.
      - trained model __Custom Fuylly Connected Dense Classifier Model__ as tensorflow (keras) saved_model format.

#### __(D) Deployment as web demo app__: 
  - Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.
  - Desciption
    - Project goal is classifiying the audio data into one of the 10 classes.
    - Usage: upload or select audio data then use the button to predict.
  - Demo
    - The demo app in the demo_app folder as an individual project. All the requirements and dependencies are in there. You can run it anywhere if you install the requirements.txt.
    - You can find the live demo as huggingface space in this [demo link](https://huggingface.co/spaces/ErtugrulDemir/UrbanSoundClassification
) as full web page or you can also us the [embedded demo widget](#demo)  in this document.  
    
## License
- This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

<h1 style="text-align: center;">Connection Links</h1>

<div style="text-align: center;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://www.hackerrank.com/ertugrulbusiness"><img src="https://hrcdn.net/fcore/assets/work/header/hackerrank_logo-21e2867566.svg" height="30"></a>
    <a href="https://app.patika.dev/ertugruldmr"><img src="https://app.patika.dev/staticFiles/newPatikaLogo.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

