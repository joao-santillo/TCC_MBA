import argparse
import numpy as np
import logging
import os

from PIL import Image
from keras.models import Model
from keras.applications import resnet50
from keras.layers import GlobalAveragePooling2D
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
base_path = '/home/equipeia/Desktop/TCC_MBA/dataset/Dataset_BUSI_with_GT'

def split_data(class_name, base_path, test_size=0.2, stratify=True):
    """Divide os dados em conjuntos de treino e teste."""
    logging.info(f"Executing Train/Test split with test_size = {test_size} for class {class_name}")
    
    source_dir = os.path.join(base_path, class_name)
    files = [os.path.join(source_dir, file) for file in os.listdir(source_dir) if file.endswith('.png')]

    y = [class_name] * len(files)

    if stratify:
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=42, stratify=y)
    else:
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
    
    logging.info(f"Train split with {len(train_files)} examples for class {class_name}")
    logging.info(f"Test split with {len(test_files)} examples for class {class_name}")

    return train_files, test_files

def resize_images(image_paths, width, height):
    """Redimensiona as imagens para a largura e altura especificadas."""
    logging.info("Resizing Breast Ultrassound images")
    resized_images = [np.array(Image.open(img).resize((width, height))) for img in image_paths]
    return np.array(resized_images)

def feature_extraction_cnn(X_train, X_test, include_top):
    """Extrai características usando o modelo ResNet50 pré-treinado."""
    logging.info("Loading the ResNet50-ImageNet model")
    
    if not include_top:
        model = resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(500, 500, 3))
        model = Model(inputs=model.input, outputs=GlobalAveragePooling2D()(model.output))
    else:
        model = resnet50.ResNet50(include_top=True, weights='imagenet', classes=1000)
        model = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
    
    model.summary()
    
    X_train = model.predict(X_train)
    X_test = model.predict(X_test)
    
    logging.info(f"Features training shape: {X_train.shape}")
    logging.info(f"Features testing shape: {X_test.shape}")
    
    return X_train, X_test

def dimension_reduction(X_train, X_test, method, n_components):
    """Reduz a dimensionalidade das características usando PCA ou IncrementalPCA."""
    logging.info("Dimensionality reduction with PCA ...")
    
    if method == 'I-PCA':
        reducer = IncrementalPCA(n_components=n_components)
    else:
        reducer = PCA(n_components=n_components)
    
    X_train = reducer.fit_transform(X_train)
    X_test = reducer.transform(X_test)
    
    logging.info(f"Features training shape: {X_train.shape}")
    logging.info(f"Features testing shape: {X_test.shape}")
    
    return X_train, X_test

def classification_svm(X_train, y_train, X_test, y_test):
    """Realiza a classificação usando SVM e calcula métricas de desempenho."""
    logging.info("Classification with Linear SVM ...")
    
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    acc = balanced_accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    logging.info(f"Accuracy Linear SVM: {acc:.4f}")
    logging.info(f"Precision Linear SVM: {precision:.4f}")
    logging.info(f"F1-Score Linear SVM: {f1:.4f}")
    logging.info(f"Recall Linear SVM: {recall:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario', help="Insert the scenario to execute", type=int)
    args = parser.parse_args()

    scenarios = {
        1: {
            'test_size': 0.2,
            'stratify': False,
            'resize': (224, 224),
            'include_top': True,
            'pca': False
        },
        2: {
            'test_size': 0.2,
            'stratify': True,
            'resize': (500, 500),
            'include_top': False,
            'pca': False
        },
        3: {
            'test_size': 0.2,
            'stratify': True,
            'resize': (500, 500),
            'include_top': False,
            'pca': True,
            'n_components': 512
        }
    }

    config = scenarios[args.scenario]
    logging.info(f"Executing scenario {args.scenario}")

    # Carregar e dividir os dados
    x_train_benign, x_test_benign = split_data('benign', base_path, config['test_size'], config['stratify'])
    x_train_malignant, x_test_malignant = split_data('malignant', base_path, config['test_size'], config['stratify'])
    x_train_normal, x_test_normal = split_data('normal', base_path, config['test_size'], config['stratify'])

    x_train = x_train_benign + x_train_malignant + x_train_normal
    x_test = x_test_benign + x_test_malignant + x_test_normal

    y_train = ['benign'] * len(x_train_benign) + ['malignant'] * len(x_train_malignant) + ['normal'] * len(x_train_normal)
    y_test = ['benign'] * len(x_test_benign) + ['malignant'] * len(x_test_malignant) + ['normal'] * len(x_test_normal)

    # Redimensionar imagens
    resized_x_train = resize_images(x_train, *config['resize'])
    resized_x_test = resize_images(x_test, *config['resize'])

    # Extração de características
    features_Xtrain, features_Xtest = feature_extraction_cnn(resized_x_train, resized_x_test, config['include_top'])

    # Redução de dimensionalidade
    if config['pca']:
        features_Xtrain, features_Xtest = dimension_reduction(features_Xtrain, features_Xtest, "PCA", config['n_components'])

    # Classificação SVM
    classification_svm(features_Xtrain, y_train, features_Xtest, y_test)
    logging.info("SVM classification done")
