import numpy as np
import json
from collections import OrderedDict 
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms, models
import torchvision.models as models
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import argparse

arch = {"vgg16":25088,
        "densenet121":1024}

parser = argparse.ArgumentParser(
    description = 'Parser | train.py'
)
parser.add_argument('data_dir', action="store", default="./flowers/")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
parser.add_argument('--arch', action="store", default="vgg16")
parser.add_argument('--learning_rate', action="store", type=float,default=0.01)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=512)
parser.add_argument('--epochs', action="store", default=3, type=int)
parser.add_argument('--gpu', action="store", default="gpu")


args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
lr = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
gpu = args.gpu
device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

data_dir = 'flowers'
train_dir, valid_dir, test_dir = data_dir + '/train', data_dir + '/valid', data_dir + '/test'



# Define transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
dataloaders = {'train': trainloader, 'valid': validloader, 'test': testloader}


# Load a pre-trained VGG16 model
model = models.vgg16(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Define a new, untrained feed-forward network as a classifier
classifier = nn.Sequential(nn.Linear(25088, 4096),
                           nn.ReLU(),
                           nn.Dropout(0.5),
                           nn.Linear(4096, 102),
                           nn.LogSoftmax(dim=1))

# Replace the pre-trained classifier with our new classifier
model.classifier = classifier

# Define the loss function and optimizer
criterion, optimizer = nn.NLLLoss(), optim.Adam(model.classifier.parameters(), lr=lr)

# Move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the classifier layers using backpropagation

steps, running_loss, print_every= 0,0,5

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        # Reset the optimizer gradients
        optimizer.zero_grad()
        # Forward pass and backward pass
        logps = model.forward(inputs)
        loss = criterion(logps, labels) 
        loss.backward()
        optimizer.step()
        # Update running loss
        running_loss += loss.item()
        # Print training statistics
        if (steps % print_every) == 0:
            # Switch to evaluation mode and turn off gradients
            model.eval()
            with torch.no_grad():
                validation_loss, accuracy = 0, 0
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    validation_loss += criterion(logps, labels)
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Training loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
            # Switch back to training mode and reset running loss
            running_loss = 0
            model.train()


test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

# Set the model to evaluation mode and turn off gradients
model.eval()
with torch.no_grad():
    test_loss, accuracy = 0, 0
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        test_loss += criterion(logps, labels)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {accuracy/len(testloader):.3f}")


checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'class_to_idx': train_dataset.class_to_idx,
    'classifier': model.classifier,
    'epochs': epochs
}

torch.save(checkpoint, 'checkpoint.pth')
print("Checkpoint Saved!")





