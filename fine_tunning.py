
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import seaborn as sn
import pandas as pd
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torchvision.transforms as T
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import time
import copy

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# %%
print(torch. __version__)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
train_dir = '/home/equipeia/Desktop/TCC_MBA/dataset/Dataset_BUSI_with_GT/train'
test_dir = '/home/equipeia/Desktop/TCC_MBA/dataset/Dataset_BUSI_with_GT/test'
train_benign_dir = '/home/equipeia/Desktop/TCC_MBA/dataset/Dataset_BUSI_with_GT/train/benign'
train_malignant_dir = '/home/equipeia/Desktop/TCC_MBA/dataset/Dataset_BUSI_with_GT/train/malignant'
train_normal_dir = '/home/equipeia/Desktop/TCC_MBA/dataset/Dataset_BUSI_with_GT/train/normal'
test_benign_dir = '/home/equipeia/Desktop/TCC_MBA/dataset/Dataset_BUSI_with_GT/test/benign'
test_malignant_dir = '/home/equipeia/Desktop/TCC_MBA/dataset/Dataset_BUSI_with_GT/test/malignant'
test_normal_dir = '/home/equipeia/Desktop/TCC_MBA/dataset/Dataset_BUSI_with_GT/test/normal'

# %%
benign_cancer = torchvision.io.read_image('/home/equipeia/Desktop/TCC_MBA/dataset/Dataset_BUSI_with_GT/train/benign/benign (1).png')
print("This is benign breast cancer")
T.ToPILImage()(benign_cancer)


# %%
malignant_example = torchvision.io.read_image('/home/equipeia/Desktop/TCC_MBA/dataset/Dataset_BUSI_with_GT/train/malignant/malignant (1).png')
print("This is malignant breast cancer")
T.ToPILImage()(malignant_example)


# %%
normal_example = torchvision.io.read_image('/home/equipeia/Desktop/TCC_MBA/dataset/Dataset_BUSI_with_GT/train/normal/normal (1).png')
print("This is normal breast without cancer")
T.ToPILImage()(normal_example)


# %% [markdown]
# ## 2. Preprocessing Data
# 
# Resnet only accepts inputs 224 x 224

# %%
# Create transform function
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
    transforms.RandomResizedCrop(224),
    #transforms.RandomHorizontalFlip(), # data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
])
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),   #must same as here
     transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# %%
train_dataset = datasets.ImageFolder(train_dir, transforms_train)
test_dataset = datasets.ImageFolder(test_dir, transforms_test)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# %% [markdown]
# ### Data Stats

# %%
print('Train dataset size:', len(train_dataset))
print('Test dataset size:', len(test_dataset))
class_names = train_dataset.classes
print('Class names:', class_names)

# %% [markdown]
# ### Data after preprocessing

# %%
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 60
plt.rcParams.update({'font.size': 20})

def imshow(input, title):
    # torch.Tensor => numpy
    input = input.numpy().transpose((1, 2, 0))
    # undo image normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # display images
    plt.imshow(input)
    plt.title(title)
    plt.show()
# load a batch of train image
iterator = iter(train_dataloader)
# visualize a batch of train image
inputs, classes = next(iterator)
out = torchvision.utils.make_grid(inputs[:4])
imshow(out, title=[class_names[x] for x in classes[:4]])

# %% [markdown]
# ## Loading base model

# %%
model = models.resnet50(pretrained=True)

# %%
model

# %%
num_features = model.fc.in_features 
print('Number of features from pre-trained model', num_features)

# %% [markdown]
# ### Customizing base model

# %% [markdown]
# #### Defining functions:

# %%
def plot_accuracy(train_accuracy: list, test_accuracy: list, num_epochs: int) -> None:
    """
    Plot training and testing accuracy over epochs.

    Parameters:
    train_accuracy (list): List of training accuracies over epochs.
    test_accuracy (list): List of testing accuracies over epochs.
    num_epochs (int): Number of epochs to plot.

    Returns:
    None
    """
    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(1, num_epochs + 1), train_accuracy, '-o')
    plt.plot(np.arange(1, num_epochs + 1), test_accuracy, '-o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'])
    plt.title('Train vs Test Accuracy over time')
    plt.show()

# %%
def plot_errors(train_error: list, test_error: list, num_epochs: int) -> None:
    """
    Plot training and testing errors over epochs.

    Parameters:
    train_accuracy (list): List of training errors over epochs.
    test_accuracy (list): List of testing errors over epochs.
    num_epochs (int): Number of epochs to plot.

    Returns:
    None
    """
    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(1, num_epochs + 1), train_error, '-o')
    plt.plot(np.arange(1, num_epochs + 1), test_error, '-o')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend(['Train', 'Test'])
    plt.title('Train vs Test Error over time')
    plt.show()

# %%
def evaluate_model(model: nn.Module, test_dataloader: DataLoader, device: torch.device) -> None:
    """
    Evaluate the model performance on the test set and visualize the results.

    Parameters:
    model (nn.Module): The trained neural network model.
    test_dataloader (DataLoader): DataLoader for the test data.
    device (torch.device): Device to perform the computations (CPU or GPU).

    Returns:
    None
    """
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)  # Feed Network
            outputs = (torch.max(torch.exp(outputs), 1)[1]).data.cpu().numpy()
            y_pred.extend(outputs)  # Save Prediction
            labels = labels.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

    # Visualization and result
    classes = test_dataloader.dataset.classes  # Assuming test_dataset has an attribute 'classes'
    
    print("Accuracy on Test set: ", accuracy_score(y_true, y_pred))
    print('Confusion matrix: \n', confusion_matrix(y_true, y_pred))
    print('Classification report: \n', classification_report(y_true, y_pred))
    
    # Plot confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(7, 7))
    plt.title("Confusion matrix for Skin Cancer classification")
    sn.heatmap(df_cm, annot=True)
    plt.show()

# %%
def do_fine_tune(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    num_epochs: int = 30
) -> None:
    """
    Fine-tune the given model using the provided training and testing data loaders, loss criterion, and optimizer.

    Parameters:
    model (nn.Module): The neural network model to be fine-tuned.
    train_dataloader (DataLoader): DataLoader for the training data.
    test_dataloader (DataLoader): DataLoader for the testing data.
    criterion (nn.Module): Loss function to use during training.
    optimizer (optim.Optimizer): Optimizer to update the model weights.
    device (torch.device): Device to perform the computations (CPU or GPU).
    num_epochs (int): Number of epochs to train the model (default is 30).

    Returns:
    None
    """
    
    train_loss = []
    train_accuracy = []
    test_loss = []
    test_accuracy = []

    start_time = time.time()  # For showing time
    
    for epoch in range(num_epochs):  # Loop for every epoch
        print(f"Epoch {epoch} running")  # Printing message
        
        # Training Phase
        model.train()  # Training model
        running_loss = 0.0  # Set loss to 0
        running_corrects = 0
        
        for inputs, labels in train_dataloader:  # Load a batch data of images
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()  # Clear gradients for optimized tensors
            outputs = model(inputs)  # Forward inputs and get output
            _, preds = torch.max(outputs, 1)  # Get predicted class
            loss = criterion(outputs, labels)
            loss.backward()  # Get loss value and update the network weights
            optimizer.step()
            
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data).item()
        
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = running_corrects / len(train_dataloader.dataset) * 100.0
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_acc)
        
        # Print progress
        print(f'[Train #{epoch+1}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}% Time: {time.time() - start_time:.4f}s')
        
        # Testing Phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).item()
        
        epoch_loss = running_loss / len(test_dataloader.dataset)
        epoch_acc = running_corrects / len(test_dataloader.dataset) * 100.0
        test_loss.append(epoch_loss)
        test_accuracy.append(epoch_acc)
        
        # Print progress
        print(f'[Test #{epoch+1}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}% Time: {time.time() - start_time:.4f}s')
        
    plot_accuracy(train_accuracy, test_accuracy, num_epochs)
    evaluate_model(model, test_dataloader, device)



# %%
def save_model (save_path):
    torch.save(model.state_dict(), save_path)

# %% [markdown]
# ### Experimentation:

# %% [markdown]
# ##### Scenario 1: Trainning only the FC Layer, with 3 outputs. 
# **Optimizer**: SGD -> learning rate: 0.0001, momentum=0.9
# 
# **Loss**: CrossEntropyLoss
# 
# **Epochs**: 30
# 
# **Data Augmentation?** No

# %%
# Add a fully-connected layer for classification
model.fc = nn.Linear(num_features, 3) #classes: normal, benignant, malignant
model = model.to(device)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# %%
# Set the random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# %%
do_fine_tune(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=30
)

# %%
save_path = 'scenerio1_acc0.69'
save_model(save_path)

# %% [markdown]
# ##### Scenario 2: Trainning only the FC Layer, with 3 outputs. 
# **Optimizer**: SGD -> learning rate: 0.0001, momentum=0.9
# 
# **Loss**: CrossEntropyLoss
# 
# **Epochs**: 60
# 
# **Data Augmentation?** No

# %%
# Add a fully-connected layer for classification
model.fc = nn.Linear(num_features, 3) #classes: normal, benignant, malignant
model = model.to(device)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

# %%
# Set the random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# %%
do_fine_tune(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=60
)


