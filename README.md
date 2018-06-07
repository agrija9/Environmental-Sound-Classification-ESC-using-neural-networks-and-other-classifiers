# Environmental-Sound-Classification-ESC-using-neural-networks-and-other-classifiers

- Audio feature extraction and classification with the [ECS-10 data set](https://github.com/karoldvl/ESC-50) audio dataset 
- ECS-10 audio data is included. It consists of 10 classes of different environmental sounds (sea waves, kids playing, etc.)
- The main goal is to compare classification accuracies for the 6 tested classifiers. 

## Dependencies
- Anaconda 2 with Python 2.7. (Python 3.6 not tested yet)
- Librosa (audio loading, audio visualization and feature extraction)
- Sci-kit learn
- Keras (Theano backend)
- Numpy, Matplotlib
- Pandas (data visualization)

## Jupyter Notebook
A Jupyter Notebook (Python 2.7 Kernel) is added to illustrate the workflow. 

The scripts for feature extraction and classification have been added as 
```.py``` files and are all loaded in the Jupyter Notebook sequentally.

Running ```feature_extraction.py``` creates a numpy array for features (```feature.npy```) and one for labels (```label.npy```).
These files will be saved in the current directory.

### Audio features extracted
- MFCC
- Chroma
- Mel spectrogram
- Tonal centroid feature
- Spectral contrast

### Classifiers implemented
- Support Vector Machine (SVM)
- Random Forest (RF)
- Naive Bayes (NB)
- Convolutional Neural Network (CNN)
- Multilayer Perceptron (MLP)
- Recurrent Neural Network (RNN)

### Accuracies obtained
**Note**: Direct comparison between classifiers can't be donde yet since their parameters haven't been tuned to optimize
accuracy yet. Out of 400 audio samples, the test set consisted on the 33% of this.
- SVM: 81.7%
- RF: 80%
- NB: 69.7%
- CNN: 71.25% (100 epochs)
- MLP: 63.125 (100 epochs)
- RNN: 66% (100 epochs)

### Approaches to improve accuracy
- Compute other features: MFCC + ZCR features improve classification 
[accuracy](https://workshop2016.iwslt.org/downloads/IWSLT_2016_paper_3.pdf)
for speech, noise and music labels. See if it also works for the 10 classes.
- Tune optimization hyperparameters (for every classifier): Weight initialization, decaying learning rate.
- Data scaling and feature normalization (MFCC)
