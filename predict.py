import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import json
from PIL import Image
from torch.autograd import Variable
import torchvision.models as models
import torch
from torch import nn, optim


parser = argparse.ArgumentParser(
    description = 'Parser | predict.py'
)

parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str)
parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = parser.parse_args()
pathImage = args.input
n_out = args.top_k
gpu = args.gpu
device_name = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
path = args.checkpoint #path to checkpoint

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    classifier = nn.Sequential(nn.Linear(25088, 4096),
                           nn.ReLU(),
                           nn.Dropout(0.5),
                           nn.Linear(4096, 102),
                           nn.LogSoftmax(dim=1))
    model.classifier = classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Resize the image
    image = image.resize((256, 256))
    
    # Crop out the center 224x224 portion of the image
    image = image.crop((16, 16, 240, 240))
    
    # Convert to numpy array
    np_image = np.array(image)
    
    # Scale to values between 0 and 1
    np_image = np_image / 255
    
    # Normalize the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Load and process the image
    with Image.open(image_path) as image:
        processed_image = process_image(image)

    # Convert to tensor and unsqueeze to create a batch of size 1
    image_tensor = torch.from_numpy(np.array(processed_image)).type(torch.FloatTensor)
    image_tensor = image_tensor.unsqueeze(0)

    # Move image tensor to device
    
    image_tensor = image_tensor.to(device)

    # Move model to device
    model.to(device)

    # Turn off gradients
    with torch.no_grad():

        # Get predictions
        logps = model.forward(image_tensor)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class = [idx_to_class[tc] for tc in top_class[0].tolist()]

    return top_p[0].tolist(), top_class



model = load_checkpoint(path)
with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)

probs, classes = predict(pathImage, model, n_out, device_name)

for i in range(len(probs)):
    print(f"{cat_to_name[str(classes[i])]} with a probability of {probs[i]}")


print("Prediction Completed!")