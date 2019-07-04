# ChangeDetectionBaseline

Baseline approach to Change Detection using deep learning and Siamese CNNs

## Setup:

Start with installing the prerequisite python libraries. We worked with the following versions: 

```
tensorflow              1.12.0
Keras                   2.2.4
Keras-Applications      1.0.7
Keras-Preprocessing     1.0.5
numpy                   1.16.0
opencv-python-headless  4.0.0.21
scikit-image            0.14.2
scikit-learn            0.20.2
albumentations         0.2.0
image-classifiers      0.2.0
imageio                2.5.0
imageio-ffmpeg         0.2.0
seaborn                0.9.0
segmentation-models    0.2.0
tqdm                   4.29.1
```

Donwload the dataset and place it into a folder specified in Settings.py. 

Then run "python main.py" with the required arguments (such as encoder type, number of epochs or the batch size).

Run this to see the help:
```
python3 main.py --help
```
