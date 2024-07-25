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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
base_path = 'dataset/Dataset_BUSI_with_GT'

#--------------------------------------------------------------------------------------------------------------------------------------------------------------

def split_data(class_name, base_path, test_size=0.2, stratify=True):    
    logging.info(f"Executing Train/Test split with test_size = {test_size} for class {class_name}")

    source_dir = os.path.join(base_path, class_name)
    files = [os.path.join(source_dir, file) for file in os.listdir(source_dir)]

    '''
    Como temos um desbalanceamento de classes, está sendo utilizando o parâmetro stratify do train_test_split.
    Se não utilizarmos este parâmetro, os labels de treinamento favorecerão a classe majoritária, o que compromete o treinamento da CNN. 
    Esta técnica evita a criação de vieses e mantém a integridade dos dados.

    Divisão de classes do dataset utilizado:

    Case	Number of images
    Benign	487
    Malignant	210
    Normal	133
    Total	780
    '''

    y = [class_name] * len(files)

    if stratify == True:
        logging.info(f"Stratified train-test split")
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=42, stratify=y)
    else:
        logging.info(f"Non-stratified train-test split")
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
    
    logging.info(f"Train split with {len(train_files)} examples for class {class_name}")
    logging.info(f"Test split with {len(test_files)} examples for class {class_name}")

    return train_files, test_files

def resizeImages(train_images, test_images, width, height):
    logging.info("Resizing Breast Ultrassound images")

    X = []
    for i in range(0, len(train_images)):
        X.append(np.array(Image.open(train_images[i]).resize(size=(width,height))))
    resized_train = np.array(X)

    X = []
    for i in range(0, len(test_images)):
        X.append(np.array(Image.open(test_images[i]).resize(size=(width,height))))
    resized_test = np.array(X)

    return resized_train, resized_test


def featureExtractionCNN(Xtrain, Xtest, include_top):
    logging.info("Loading the ResNet50-ImageNet model")

    '''É vantajoso jogar o topo fora, para não ser necessário redimensionar as imagens de entrada. Ao redimensionar imagens de entrada, 
    algumas características importantes são perdidas.'''
    
    if include_top == False:
        logging.info("ResNet50 Top not included")
        model = resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(500, 500, 3))
        model = Model(inputs=model.input, outputs=GlobalAveragePooling2D()(model.output))  
        
    else:
        logging.info("ResNet50 Top included")
        model = resnet50.ResNet50(include_top=True, weights='imagenet', classes=1000)
        model = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
    
    model.summary()
    
    Xtrain = model.predict(Xtrain)
    Xtest = model.predict(Xtest)

    '''prediction = np.array(model.predict(Xtrain))
    Xtrain = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]))
    
    prediction = np.array(model.predict(Xtest))
    Xtest = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]))'''    

    print('\t\tFeatures training shape: ', Xtrain.shape)
    print('\t\tFeatures testing shape: ', Xtest.shape)
    return Xtrain, Xtest

#CROSS-VALIDATION NÃO SE APLICA MUITO BEM PARA REDES NEURAIS, SE APLICA PARA ML COMUM.
'''def crossValidation(Xtrain, Ytrain):
    print("\tCross-validation with K-NN ...")
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    #kf = KFold(n_splits=5, shuffle=True)

    knn = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(knn, Xtrain, np.ravel(Ytrain, order='C'), cv=kf)
    print('\t\tAccuracy K-NN: %0.4f +/- %0.4f' % (scores.mean(), scores.std()))'''

'''UTILIZAR O I-PCA PARA RODAR SEM O TOPO, POIS A DIMENSIONALIDADE ESTARÁ AUMENTADA EM 49x PARA RESNET-50.
UTILIZAR O PCA JOGA TODOS OS DADOS NA MEMÓRIA DE UMA SÓ VEZ, FAZER O TESTE PARA VER SE O PC AGUENTA.'''

def dimensionReduction(Xtrain, Xtest, method, n_components):
    print("\tDimensionality reduction with PCA ...")
    if method == 'I-PCA':
        ipca = IncrementalPCA(n_components=n_components)
        Xtrain = ipca.fit_transform(Xtrain)
        Xtest = ipca.transform(Xtest)
    else:
        pca = PCA(n_components=n_components)
        Xtrain = pca.fit_transform(Xtrain)
        Xtest = pca.transform(Xtest)
    print('\t\tFeatures training shape: ', Xtrain.shape)
    print('\t\tFeatures testing shape: ', Xtest.shape)
    return Xtrain, Xtest

'''KNN, RANDOM-FOREST, NAIVE-BAYES, LOGISTIC-REGRESSION
GARANTIA DE SEPARABILIDADE DAS CLASSES
PROVA MATEMATICA DE QUE A CLASSIFICAÇÃO NÃO ESTÁ ENVIESADA, PERMITINDO GENERALIZAÇÃO POSTERIOR.'''

def classificationSVM(Xtrain, Ytrain, Xtest, Ytest):
    print("\tClassification with Linear SVM ...")
    svm = SVC(kernel='linear')
    svm.fit(Xtrain, np.ravel(Ytrain, order='C'))
    result = svm.predict(Xtest)
    
    acc = balanced_accuracy_score(result, np.ravel(Ytest, order='C'))
    precision = precision_score(result, np.ravel(Ytest, order='C'), average='weighted')
    f1 = f1_score(result, np.ravel(Ytest, order='C'), average='weighted')
    recall = recall_score(result, np.ravel(Ytest, order='C'), average='weighted')

    print("\t\tAccuracy Linear SVM: %0.4f" % acc)
    print("\t\tPrecision Linear SVM: %0.4f" % precision)
    print("\t\tF1-Score Linear SVM: %0.4f" % f1)
    print("\t\tRecall Linear SVM: %0.4f" % recall)

#NÃO É NECESSÁRIO:
#PRECISO TER CARACTERÍSTICAS DE ALTO NÍVEL PARA A EXPLICABILIDADE E ESSE PROCESSO DESCONSIDERA ALGUMAS CARACTERÍSTICAS.
'''
    def multiFeatureExtractionCNN(Xtrain, Xtest):
    print("\tLoading the ResNet50-ImageNet model ...")
    model = resnet50.ResNet50(include_top=True, weights='imagenet', input_shape=(224, 224, 3), classes=1000)

    modelGlobal = Model(inputs=model.input, outputs=model.get_layer(name='avg_pool').output)
    modelLocal = Model(inputs=model.input, outputs=model.get_layer(name='activation_4').output)
    
    prediction = np.array(modelGlobal.predict(Xtrain))
    XtrainGlobal = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]*prediction.shape[2]*prediction.shape[3]))
    
    prediction = np.array(modelGlobal.predict(Xtest))
    XtestGlobal = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]*prediction.shape[2]*prediction.shape[3]))

    prediction = np.array(modelLocal.predict(Xtrain))
    XtrainLocal = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]*prediction.shape[2]*prediction.shape[3]))
    
    prediction = np.array(modelLocal.predict(Xtest))
    XtestLocal = np.reshape(prediction, (prediction.shape[0], prediction.shape[1]*prediction.shape[2]*prediction.shape[3]))
    
    Xtrain = np.concatenate((XtrainGlobal, XtrainLocal), axis=1)
    Xtest = np.concatenate((XtestGlobal, XtestLocal), axis=1)

    print('\t\tFeatures fusion training shape: ', Xtrain.shape)
    print('\t\tFeatures fusion testing shape: ', Xtest.shape)
    return Xtrain, Xtest
    
'''
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scenario', help="Insert the scenario to execute", type=int)
    args = parser.parse_args()

    '''Cenário 1
        Test_size = 20%
        Estratificação manual
        Com redimensionamento para 224x224
        ResNet50
        Topo da ResNet50 mantido
        Classificação SVM
        Métricas Acurácia, Precisão, F1-Score, Revocação (balanceados)
    '''
    
    '''
    A compressão de imagens me fez perder X%, usar a arquitetura original não é válido. A comparação de resultados do cenário 1 com o cenário 2 é justificativa
    para jogar o topo fora, já que esse percentual é significativo para uma aplicação de diagnóstico médico.
    '''

    '''Cenário 2
        Test_size = 20%
        Estratificação manual
        Sem redimensionamento, imagens 500x500
        ResNet50
        Topo da ResNet50 descartado
        Camada de Average Pooling
        Sem PCA
        Classificação SVM
        Métricas Acurácia, Precisão, F1-Score, Revocação (balanceados)
    '''

    '''
    Três objetivos diferentes no baseline. 75% das componentes são irrelevantes. Analisar métricas para definir percentual de redução.
    '''

    '''Cenário 3
        Test_size = 20%
        Estratificação manual
        Sem redimensionamento, imagens 500x500
        ResNet50
        Topo da ResNet50 descartado
        Camada de Average Pooling
        Aplicação do PCA com 512 componentes
        Classificação SVM
        Métricas Acurácia, Precisão, F1-Score, Revocação (balanceados)
    '''
    
    '''Os resultados estão concisos. As métricas variam de 0 a 1 e todas estão com valores bem próximos, mostrando consistência.'''

    '''
    1) Rodada com fine-tunning tradicional, com aumento de dados, peso de classes.
    2) Em cima do melhor resultado, comparar o fine-tunning com a extração de características.
    3) Em cima do melhor resultado de fine-tunning, aplicar o Poly-CAM+

    '''

    '''
    Metodologia:
    1) Seção para explicar os dados (balanceamento, resolução, particionamento treino-teste, imagens de exemplo para demonstar características das classes)
    2) Seção para descrever o fluxo de execução: resultados divididos em três partes: extração de características, transferência de aprendizado com fine-tunning 
    (técnicas como aumento de dados e variação dos hiperparâmetros), explicabilidade. Detalhar as métricas utilizadas.
    3) Resultados da extração de caracterísicas com gráfico, tabela. Explicar as três configurações. Descrever o resultado e interpretar.
    4) Fine-tunning
    5) Explicablidade
    
    '''

    if (args.scenario == 1):
        
        test_size = 0.2

        logging.info("Splitting data")
        
    
        x_train_benign, x_test_benign = split_data('benign', base_path, test_size, stratify=False)
        x_train_malignant, x_test_malignant = split_data('malignant', base_path, test_size, stratify=False)
        x_train_normal, x_test_normal = split_data('normal', base_path, test_size, stratify=False)

        x_train = x_train_benign + x_train_malignant + x_train_normal
        x_test = x_test_benign + x_test_malignant + x_test_normal

        y_train = ['benign'] * len(x_train_benign) + ['malignant'] * len(x_train_malignant) + ['normal'] * len(x_train_normal)
        y_test = ['benign'] * len(x_test_benign) + ['malignant'] * len(x_test_malignant) + ['normal'] * len(x_test_normal)
        logging.info(f"Train/test split executed with test_size = {test_size}")

        resized_x_train, resized_x_test, = resizeImages(x_train, x_test, 224, 224)
        logging.info("Train/test resized")

        features_Xtrain, features_Xtest = featureExtractionCNN(resized_x_train, resized_x_test, include_top=True)
        logging.info("Features extracted")

        classificationSVM(features_Xtrain, y_train, features_Xtest, y_test)
        logging.info("SVM classification done")
        
    elif (args.scenario == 2):
        
        test_size = 0.2

        logging.info("Splitting data")
        
        x_train_benign, x_test_benign = split_data('benign', base_path, test_size, stratify=True)
        x_train_malignant, x_test_malignant = split_data('malignant', base_path, test_size, stratify=True)
        x_train_normal, x_test_normal = split_data('normal', base_path, test_size, stratify=True)

        x_train = x_train_benign + x_train_malignant + x_train_normal
        x_test = x_test_benign + x_test_malignant + x_test_normal

        y_train = ['benign'] * len(x_train_benign) + ['malignant'] * len(x_train_malignant) + ['normal'] * len(x_train_normal)
        y_test = ['benign'] * len(x_test_benign) + ['malignant'] * len(x_test_malignant) + ['normal'] * len(x_test_normal)

        logging.info(f"Train/test split executed with test_size = {test_size}")

        resized_x_train, resized_x_test, = resizeImages(x_train, x_test, 500, 500)

        features_Xtrain, features_Xtest = featureExtractionCNN(resized_x_train, resized_x_test, include_top=False)
        logging.info("Features extracted")

        #x_train, x_test = dimensionReduction(features_Xtrain, features_Xtest, "PCA", n_components = 256)

        classificationSVM(features_Xtrain, y_train, features_Xtest, y_test)
        logging.info("SVM classification done")

        

    elif (args.scenario == 3):
        
        test_size = 0.2

        logging.info("Splitting data")
        
        x_train_benign, x_test_benign = split_data('benign', base_path, test_size, stratify=True)
        x_train_malignant, x_test_malignant = split_data('malignant', base_path, test_size, stratify=True)
        x_train_normal, x_test_normal = split_data('normal', base_path, test_size, stratify=True)

        x_train = x_train_benign + x_train_malignant + x_train_normal
        x_test = x_test_benign + x_test_malignant + x_test_normal

        y_train = ['benign'] * len(x_train_benign) + ['malignant'] * len(x_train_malignant) + ['normal'] * len(x_train_normal)
        y_test = ['benign'] * len(x_test_benign) + ['malignant'] * len(x_test_malignant) + ['normal'] * len(x_test_normal)

        logging.info(f"Train/test split executed with test_size = {test_size}")

        resized_x_train, resized_x_test, = resizeImages(x_train, x_test, 500, 500)

        features_Xtrain, features_Xtest = featureExtractionCNN(resized_x_train, resized_x_test, include_top=False)
        logging.info("Features extracted")

        x_train_PCA, x_test_PCA = dimensionReduction(features_Xtrain, features_Xtest, "PCA", n_components = 512)

        classificationSVM(x_train_PCA, y_train, x_test_PCA, y_test)
        logging.info("SVM classification done")