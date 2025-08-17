#!/usr/bin/env python3
"""
This is the second file to be run for the NSL-KDD dataset. 
This file depends on the output of the first file which is the preprocessing file.

"""

import tensorflow as tf
from tensorflow import keras

# from tensorflow.keras import layers, regularizers

import os
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


import time 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from datetime import datetime

import os
import  sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

sys.path.append(parent_dir)

import utils 
import utils.utility as utility
import shap
import utils.dnn_binaryClassification as dnnbin
import utils.dnn_multiClassification as dnnmulti
import utils.dnn_explanations as dnnexp
import utils.ml_utility as mu

# ## Data processing and exploration

def HDMLFSNSLKDD_02():
    t_start = time.time()

    data_dir = 'path/to/HDMLFS/data_folder/'
    datasets_full = data_dir + '01fulltraintest.csv'

    # Loading the datasets into dataframe
    raw_df = pd.read_csv(datasets_full)

    pd.set_option('display.max_columns', None) 

    pd.set_option('display.max_rows', 100) 

    raw_df.attack.value_counts()

    label = ['attack']

    r2l = ['ftp_write', 'xlock', 'xsnoop', 'guess_passwd', 'imap', 'named', 'warezmaster',
        'multihop', 'sendmail', 'snmpguess', 'snmpgetattack', 'spy', 'warezclient', 'worm', 'phf']

    dos = ['pod', 'smurf', 'apache2', 'teardrop', 'back', 'land', 
        'mailbomb', 'neptune', 'udpstorm', 'processtable']

    probe = ['ipsweep', 'portsweep', 'mscan', 'saint', 'nmap', 'satan']

    u2r = ['loadmodule', 'buffer_overflow', 'perl', 'xterm', 'httptunnel', 
        'rootkit', 'ps', 'sqlattack']

    # Create a mapping dictionary for replacement
    replace_dict = {**{item: 'r2l' for item in r2l},
                    **{item: 'dos' for item in dos},
                    **{item: 'probe' for item in probe},
                    **{item: 'u2r' for item in u2r}}

    raw_df['attack'] = raw_df['attack'].replace(replace_dict)

    raw_df['Threat'] = np.where(raw_df['attack']=='normal', 0, 1)

    mostService = 6

    topmostservice = raw_df['service'].value_counts().nlargest(mostService).index

    raw_df['service'] = raw_df['service'].where(raw_df['service'].isin(topmostservice), 'others')

    cleaned_df = raw_df.copy()

    # Removing zero-variance columns
    removed_columns = ["rerror_rate",  "num_root",  "dst_host_srv_serror_rate", "dst_host_srv_rerror_rate",
                    "serror_rate",  "srv_serror_rate",  "dst_host_same_srv_rate",  "srv_rerror_rate", "num_outbound_cmds"]

    cleaned_df = cleaned_df.drop(removed_columns, axis=1)

    cleaned_df2 = cleaned_df.copy()

    all_columns = list(cleaned_df.columns)

    labels = ['attack', 'Threat']

    dataset_type = ['set_type']

    categorical_features = ['protocol_type', 'service', 'flag']

    numerical_features = list(set(all_columns)-set(categorical_features)-set(dataset_type)-set(labels)) 

    # Initialize LabelEncoder
    le = LabelEncoder()

    # Fit and transform the 'attack' column
    cleaned_df['attack'] = le.fit_transform(cleaned_df['attack'])

    y = cleaned_df[label[0]]

    label_names = list(le.classes_)

    n_classes = len(le.classes_)

    # Hot encoding the categorical data
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32)
    encoded_categorical = encoder.fit_transform(cleaned_df[categorical_features])

    encoded_feature_names = encoder.get_feature_names_out(categorical_features)

    df_ohe = pd.DataFrame(encoded_categorical, columns=encoded_feature_names)

    df_full = pd.concat([df_ohe, cleaned_df[numerical_features]], axis=1)

    df_full_X_train, df_full_X_test, y_train, y_test = train_test_split(df_full, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()

    df_full_X_train[numerical_features] = scaler.fit_transform(df_full_X_train[numerical_features])

    df_full_X_test[numerical_features] = scaler.transform(df_full_X_test[numerical_features])

    # #### Decision Trees Classifiers

    t1 = time.time()

    report_df_dt, modelDT, y_predDT = mu.ml_multiClReport(mu.dt, df_full_X_train,
                                                        y_train, df_full_X_test,
                                                        y_test, label_names)

    t2 = time.time()
    print(f'{(t2-t1):.2f}')

    print(f'Decision Tree Metrics in 4dp \n: {report_df_dt}')

    print(f'\nDecision Tree ML training takes: {(t2-t1):.2f}\n')


    # mu.ml_multiConfxMtrx(y_test, y_predDT, label_names)

    # #### Random Forest Classifier

    t1 = time.time()

    report_df_rf, modelRF, y_predRF = mu.ml_multiClReport(mu.rf, df_full_X_train, y_train, df_full_X_test, y_test, label_names)

    t2 = time.time()

    print(f'Random Forest Metrics in 4dp\n: {report_df_rf}')

    print(f'\nRandom Forest ML training takes: {(t2-t1):.2f}\n')


    # #### Logistic Regression Classifier

    t1 = time.time()

    report_df_lr, modelLR, y_predLR = mu.ml_multiClReport(mu.lr, df_full_X_train,
                                                        y_train, df_full_X_test,
                                                        y_test, label_names)

    t2 = time.time()


    print(f'Logistic Regression Metrics in 4dp: \n: {report_df_lr}')

    print(f'\nLogistic Regression ML training takes: {(t2-t1):.2f}\n')

    # #### Naive Bayes Classifier

    t1 = time.time()

    report_df_nb, modelNB, y_predNB = mu.ml_multiClReport(mu.nb, df_full_X_train,
                                                        y_train, df_full_X_test,
                                                        y_test, label_names)


    t2 = time.time()

    print(f'Naive Bayes Metrics in 4dp: \n: {report_df_nb}')

    print(f'\nNaive Bayes ML training takes: {(t2-t1):.2f}\n')

    # #### XGBoost Classifier

    t1 = time.time()

    report_df_xgb, modelXGB, y_predXGB = mu.ml_multiClReport(mu.xgb, df_full_X_train, y_train, df_full_X_test, y_test, label_names)

    t2 = time.time()

    print(f'XGBoost Metrics in 4dp: \n: {report_df_xgb}')

    print(f'\nXGBoost ML training takes: {(t2-t1):.2f}\n')

    t_stop = time.time()
    print(f'The script took a total of: {(t_stop-t_start):.2f}s')

# --------------------------------------------------
def main():

    with open("output_results_nslkdd02.txt", "w") as f:
        sys.stdout = f
        
        now = datetime.now()

        print(f'The script is starting . . . . . {now.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        HDMLFSNSLKDD_02()        
    
        print(f'\n\nThe End')
# --------------------------------------------------
if __name__ == '__main__':
    main()