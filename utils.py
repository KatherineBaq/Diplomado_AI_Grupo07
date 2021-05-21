import os
import pandas as pd
import numpy as np
import cv2
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

CATEGORIES = ['COVID19','NORMAL','PNEUMONIA']
img_size=224

# create a dataframe with info about file_name and label.
def metainfo_df(path):
    """
    Function to create a dataframe with filesname and labels
    """
    meta_info = []
    for i,folder in enumerate(os.listdir(path)):
        n_class = i
        for img in os.listdir(os.path.join(path,folder)):
            meta_info.append({
                'filename':img,
                'label':str(n_class)
            })
    return pd.DataFrame(meta_info)


def sampled_df(df):
    """
    Create a sample based on a dataframe with same number of images by class
    """
    classes = df['label'].unique()
    sample_size = min(df['label'].value_counts())
    df_sampled = []
    for elem in classes:
        df_temp = df[df['label'] == elem]
        df_sampled.append(df_temp.sample(sample_size))
    return pd.concat(df_sampled)




def load_data(data_path):
    """"
    Load all data from directory
    """
    train_data = []
    for category in CATEGORIES:
        path = os.path.join(data_path,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img))
            if len(img_array.shape)==3:
                img_new = cv2.resize(img_array,(img_size,img_size))
                train_data.append([img_new, class_num])
            else:
                print(img.shape)
    return train_data


def images_to_arrays(data):
    """
    Turn the images loaded into arrays
    """
    X = []
    y = []
    for img,label in data:
        X.append(img)
        y.append(label)
    # turn X into an array:
    X = np.array(X).reshape(-1,img_size,img_size,3)
    y = np.array(y)
    return X,y



def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')