import logging
import numpy as np
import os
import matplotlib.pyplot as plt

from PIL import Image
from keras.layers import Dense
from keras.models import Model, Sequential, load_model
from keras.optimizers import SGD
from keras.applications import resnet50
from keras.utils import to_categorical
from keras.layers import GlobalAveragePooling2D
from sklearn.preprocessing import LabelEncoder


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os

def get_labels_from_directory(base_path):
    # Initialize empty lists to store file paths and labels
    X_train, y_train, X_test, y_test = [], [], [], []

    # Get training data and labels
    train_dir = os.path.join(base_path, 'train')
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            X_train.append(file_path)
            y_train.append(class_name)

    # Get testing data and labels
    test_dir = os.path.join(base_path, 'test')
    for class_name in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_name)
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            X_test.append(file_path)
            y_test.append(class_name)

    return X_train, y_train, X_test, y_test


def resizeImages(train_images, test_images, width, height):
    logging.info("Resizing Breast Ultrassound images")

    X_Train = []
    for i in range(0, len(train_images)):
        X_Train.append(np.array(Image.open(train_images[i]).resize(size=(width,height))))
    resized_train = np.array(X_Train)

    X_Test = []
    for i in range(0, len(test_images)):
        X_Test.append(np.array(Image.open(test_images[i]).resize(size=(width,height))))
    resized_test = np.array(X_Test)

    return resized_train, resized_test


def fine_tune(num_classes, num_epochs, batch_size, X_train , Y_train, X_test, Y_test):
    base_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for layer in base_model.layers[:-int(7)]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)  
    x = Dense(1024, activation='relu')(x)  
    predictions = Dense(num_classes, activation='softmax')(x)  #

    model = Model(inputs=base_model.input, outputs=predictions)
    optimizer = SGD(learning_rate=0.01, momentum=0.0001, decay=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training the model ...")
    
    batches = list(range(0, len(Y_train), batch_size))
    perm = np.random.permutation(len(Y_train))
    
    errLoss = []
    accLoss = []
    errLoss.append(1)
    accLoss.append(0)

    for e in range(0, num_epochs):
        for b in batches:
            if b + batch_size < len(Y_train):
                x = X_train[perm[b : b + batch_size]]
                y = Y_train[perm[b : b + batch_size]]
            else:
                x = X_train[perm[b : ]]
                y = Y_train[perm[b : ]]
            
            loss = model.train_on_batch(x, y)

    print("\tEpoch %i. [Error, Accuracy]: %.15f, %.15f " % (e+1, loss[0], loss[1]))
    errLoss.append(loss[0])
    accLoss.append(loss[1])
    
    print("Testing the model ...")
    acc = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print("\tTop-1 Accuracy: %f" % acc[1])

if __name__ == '__main__':
    base_path = '/home/equipeia/Desktop/TCC_MBA/dataset/Dataset_BUSI_with_GT'
    num_classes = 3 
    num_epochs = 30  
    batch_size = 32  
    
    X_train, y_train, X_test, y_test = get_labels_from_directory(base_path)
    X_train, X_test = resizeImages(X_train, X_test, 224, 224)
    
    print('\tTraining set shape: ', X_train.shape)
    print('\tTesting set shape: ', X_test.shape)
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    logging.info("Fine Tunning ResNet50:")
    fine_tune(num_classes, num_epochs, batch_size, X_train, y_train, X_test, y_test)