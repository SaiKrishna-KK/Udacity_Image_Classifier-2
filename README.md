# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program 2023. In this project, I have first developed code for an image classifier built with PyTorch, then convert it into a command line application.

# Flower Image Classifier

This is a project to train a neural network on the Oxford Flowers 102 dataset to classify images of flowers into their respective species. To get detail commprehension of the project please the `Image Classifier Project.ipynb.`

The project consists of two parts:

- `train.py`: a Python script that trains a neural network on the Oxford Flowers 102 dataset and saves the trained model as a checkpoint
- `predict.py`: a Python script that uses the trained model to predict the species of a flower in an input image

## Getting Started

To get started with this project, you should first clone this repository to your local machine:

```bash
git clone https://github.com/<username>/flower-image-classifier.git
```
Once you have cloned the repository, you will need to install the required Python packages:


## Training the Model

To train the model, run the train.py script:

```bash
python train.py <data_dir> --save_dir <save_dir> --arch <arch> --learning_rate <learning_rate> --hidden_units <hidden_units> --epochs <epochs> --gpu
```
- `<data_dir>` is the directory where the Oxford Flowers 102 dataset is stored
- `<save_dir>` is the directory where the trained model checkpoint will be saved
- `<arch>` is the architecture of the neural network to be used for training (default is `vgg16`)
- `<learning_rate>` is the learning rate to be used for training (default is `0.001`)
- `<hidden_units>` is the number of hidden units in the classifier layer of the neural network (default is `4096`)
- `<epochs>` is the number of epochs to train for (default is `5`)
- `--gpu` is an optional flag to train the model on GPU

## Using the Trained Model for Prediction
To use the trained model for prediction, run the `predict.py` script:

```bash
python predict.py <input> <checkpoint> --top_k <top_k> --category_names <category_names> --gpu
```

- `<input>` is the input image file for which to predict the species of the flower
- `<checkpoint>` is the checkpoint file for the trained model
- `--top_k` is the number of most likely classes to return (default is `5`)
- `--category_names` is a JSON file that maps the class values to category names (default is `cat_to_name.json`)
- `--gpu` is an optional flag to use GPU for prediction

## Acknowledgments
This project is part of the Udacity AI Programming with Python Nanodegree Program. The Oxford Flowers 102 dataset is provided by the Visual Geometry Group at the University of Oxford.