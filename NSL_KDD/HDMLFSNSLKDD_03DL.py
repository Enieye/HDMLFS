#!/usr/bin/env python3
"""
This is the third file to be run for the NSL-KDD dataset. 
It is the file that produces the results of the deep learning models of DNN, CNN and ResNet
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


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder

from tensorflow.keras import layers, models, callbacks, regularizers
import tensorflow.keras.backend as K
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.model_selection import KFold  # Corrected import

import scipy.stats as stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.compose import ColumnTransformer
from tensorflow.keras.metrics import Precision, Recall, AUC

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
def HDMLFSNSLKDD_03():
    t_start = time.time()
    data_dir = 'path/to/HDMLFS/data_folder/'

    datasets_full = data_dir + '01fulltraintest.csv'

    # Loading tdatasets into dataframe

    df = pd.read_csv(datasets_full)


    pd.set_option('display.max_columns', None)

    pd.set_option('display.max_rows', 100)

    # Reducing the memory usage
    from utils.reduce_memory import optimize_memory
    df = optimize_memory(df)

    df2 = df.copy()


    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    label = ['attack']

    labels = ['attack', 'Threat']

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

    df['attack'] = df['attack'].replace(replace_dict)

    # Creating a binary column for attack
    df = utility.getBin(df, label)

    label_names = list(df[label[0]].value_counts().index.sort_values())

    num_classes = len(label_names)

    constant_zero_columns = df.columns[(df == 0).all()]
    constant_zero_columns = list(constant_zero_columns)


    # Removing features that have constant_zero_columns
    df = df.drop(constant_zero_columns, axis=1)

    # Getting Binary columns

    binary_columns = [c for c
                        in list(df.drop(labels, axis=1))
                        if df[c].nunique() == 2]


    categorical_columns = ['protocol_type', 'service', 'flag']

    numerical_columns = list(set(df.columns) - set(categorical_columns)- set(binary_columns) - set(labels))

    # Correlation
    # utility.plot_correlation_heatmap_no_numbers(df[numerical_columns])


    # Checking for correlation between the numeric variables
    corr_columns = utility.find_correlated_columns(df[numerical_columns])

    all_keys = {key for d in corr_columns for key in d.keys()}

    all_corr_columns = list(all_keys)
    distinct_values = set()

    for d in corr_columns:
        for value in d.values():
            # Check if the value is a list
            if isinstance(value, list):
                # Add each element of the list as a separate item in the set
                distinct_values.update(value)
            else:
                # Otherwise, just add the value to the set
                distinct_values.add(value)

    common_key_values = all_keys.intersection(distinct_values)

    # numeric columns to be removed because of corrlation
    removed_columns = list(distinct_values - common_key_values)

    # Now remove removed_columns from the df and numerical columns
    df = df.drop(removed_columns, axis=1)

    numerical_columns2 = list(set(numerical_columns)-set(removed_columns))

    # Encoding the Categoirical

    # First reducing the cardinality of all the categorical features
    for feature in categorical_columns:
        print(f"{feature} : {df[feature].nunique()} unique categories")
        if df[feature].nunique()>6:
            df[feature] = np.where(df[feature].isin(df[feature].value_counts().head(5).index), df[feature], 'others')

    # Hot encoding the categorical data
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32, handle_unknown="ignore")
    encoded_categorical_train = encoder.fit_transform(df[categorical_columns])

    encoded_feature_names = encoder.get_feature_names_out(categorical_columns)
    df_ohe = pd.DataFrame(encoded_categorical_train, columns=encoded_feature_names)

    df_full = pd.concat([df_ohe, df[numerical_columns2], df[binary_columns], df[labels]], axis=1)

    # Now, label encoding the 'attack' feature
    le = LabelEncoder()
    df_full[labels[0]] = le.fit_transform(df_full[labels[0]])

    train_df, test_df, y_train, y_test = train_test_split(df_full.drop('set_type', axis=1), df_full[labels[0]], test_size=0.2, random_state=42)

    minmax_scaler = MinMaxScaler()
    train_df[numerical_columns2] = minmax_scaler.fit_transform(train_df[numerical_columns2])
    test_df[numerical_columns2] = minmax_scaler.transform(test_df[numerical_columns2])


    X_train, y_train,  X_valid, y_valid, X_test, y_test, feature_names, input_shape = dnnmulti.generateDataSet(train_df, test_df,
                                                                                                            labels, True)

    t1 = time.time()

    report_df_dnn, history_dnn, model_dnn, y_true_dnn, y_pred_dnn = dnnmulti.trainDNNMultiClass(X_train, y_train, X_valid, y_valid, X_test, y_test, num_classes,
                        label_names, 128, dropout_rate=0.1, l1_reg=1e-3, model_name="Feedforward_NN")

    t2 = time.time()

    print(f'DNN Metrics in 4dp: \n: {report_df_dnn}')

    print(f'\nDNN DL training takes: {(t2-t1):.2f}\n')

    t1 = time.time()
    report_df_cnn, history_cnn, model_cnn, y_true_cnn, y_pred_cnn = dnnmulti.trainCNNMultiClass(X_train, y_train, X_valid, y_valid, X_test, y_test, num_classes,
                        label_names, batch_size=128, dropout_rate=1e-8, l1_reg=1e-8, model_name="CNN")
    t2 = time.time()


    print(f'CNN Metrics in 4dp: \n: {report_df_cnn}')

    print(f'\nCNN DL training takes: {(t2-t1):.2f}\n')

    t1 = time.time()
    report_df_resnet, history_resnet, model_resnet, y_true_resnet, y_pred_resnet = dnnmulti.trainResNetMultiClass(X_train, y_train, X_valid, y_valid, X_test, y_test, num_classes,
                            label_names, 128, dropout_rate=1e-12, l1_reg=1e-12, model_name="Resnet")
    t2 = time.time()


    print(f'ResNet Metrics in 4dp: \n: {report_df_resnet}')

    print(f'\nResNet DL training takes: {(t2-t1):.2f}\n')

    # dnnmulti.dnn_multiConfxMtrx(y_true_dnn, y_pred_dnn, label_names)


    # dnnmulti.dnn_multiConfxMtrx(y_true_cnn, y_pred_cnn, label_names)

    # dnnmulti.dnn_multiConfxMtrx(y_true_resnet, y_pred_resnet, label_names)


    # Getting the feature importance
    feature_importancesDNN = dnnexp.compute_intgrd_explanations(model_dnn, X_test, feature_names, 5000)

    top25IGNSLKDDFeats = list(feature_importancesDNN.Features)[:25]

    # dnnexp.plotImportance(feature_importancesDNN[:20], figsize=(10,6))

    shap_imp_dnn, shap_values_dnn, X_sample_dnn = dnnexp.shap_importance_multi(model_dnn, X_test, feature_names, 500)

    top25SHAPNSLKDDFeats = list(shap_imp_dnn.Features)[:25]

    # dnnexp.create_shap_bar_multi(shap_values_dnn, X_sample_dnn, feature_names, max_display=20)


    # dnnexp.create_shap_waterfall_multi(shap_values_dnn, X_sample_dnn, feature_names, instance_idx=0, max_display=20)


    # dnnexp.create_shap_summary_aggregated(shap_values_dnn, X_sample_dnn, feature_names)

    # dnnexp.create_shap_waterfall_aggregated(shap_values_dnn, X_sample_dnn, feature_names)

    selected_featuresIGShap = dnnexp.select_IGShapFeatures(feature_importancesDNN, shap_imp_dnn, thresh=0.5)

    X_train_sf, y_train_sf,  X_valid_sf, y_valid_sf, X_test_sf, y_test_sf, feature_names_sf, input_shape_sf = dnnmulti.getSFDataSet(train_df, test_df,
                                                                            selected_featuresIGShap+[labels[1]], label, True)


    t1 = time.time()
    report_df_dnn_sf, history_dnn2, model_dnn2, y_true_dnn2, y_pred_dnn2 = dnnmulti.trainDNNMultiClass(X_train_sf, y_train_sf, X_valid_sf, y_valid_sf, X_test_sf, y_test_sf, num_classes,
                        label_names, 128, dropout_rate=1e-5, l1_reg=1e-5, model_name="Feedforward_NN")

    t2 = time.time()

    print(f'DNN_SF Metrics in 4dp: \n: {report_df_dnn_sf}')

    print(f'\nDNN_SF DL training takes: {(t2-t1):.2f}\n')

    # dnnmulti.dnn_multiConfxMtrx(y_true_dnn2, y_pred_dnn2, label_names)

    t1 = time.time()

    report_cnn_sf, history_cnn_sf, model_cnn_sf, y_true_cnn_sf, y_pred_cnn = dnnmulti.trainCNNMultiClass(X_train_sf, y_train_sf, X_valid_sf, y_valid_sf, X_test_sf, y_test_sf,
                                                                                                        num_classes, label_names, 128, dropout_rate=1e-8, l1_reg=1e-8, model_name="cnn_withFS")

    t2 = time.time()



    print(f'CNN_SF Metrics in 4dp: \n: {report_cnn_sf}')

    print(f'\nCNN_SF DL training takes: {(t2-t1):.2f}\n')

    # dnnmulti.dnn_multiConfxMtrx(y_true_cnn_sf, y_pred_cnn, label_names)

    t1 = time.time()

    report_resnet_sf, history_resnet_sf, model_resnet_sf, y_true_resnet_sf, y_pred_resnet = dnnmulti.trainResNetMultiClass(X_train_sf, y_train_sf, X_valid_sf, y_valid_sf, X_test_sf, y_test_sf, num_classes,
                                                                                                                        label_names, 128, dropout_rate=1e-8, l1_reg=1e-8, model_name="Resnet_withFS")

    t2 = time.time()

    print(f'ResNet_SF Metrics in 4dp: \n: {report_resnet_sf}')

    print(f'\nResNet_SF DL training takes: {(t2-t1):.2f}\n')

    # dnnmulti.dnn_multiConfxMtrx(y_true_resnet_sf, y_pred_resnet, label_names)

    # dnnmulti.plotModel(history_dnn2)

    # dnnmulti.plotModel(history_cnn_sf, 1)

    # dnnmulti.plotModel(history_resnet_sf, 1)

    # #### Performing Ablation Studies (abs)


    # 1. Ablation Studies
    # i. ABS IG

    # Considering the top 25 encoded features from IG



    X_train_absIG, y_train_absIG,  X_valid_absIG, y_valid_absIG, X_test_absIG, y_test_absIG, feature_names_absIG, input_shape_absIG = dnnmulti.getSFDataSet(train_df, test_df,
                                                                            top25IGNSLKDDFeats, label, True)



    # a. DNN ABS IG

    t1 = time.time()


    report_df_dnn_absIG, history_dnn_absIG, model_dnn_absIG, y_true_dnn_absIG, y_pred_dnn_absIG = dnnmulti.trainDNNMultiClass(X_train_absIG, y_train_absIG, X_valid_absIG, y_valid_absIG, X_test_absIG, y_test_absIG, num_classes,
                        label_names, 128, dropout_rate=1e-2, l1_reg=1e-2, model_name="Feedforward_NNabs")

    t2 = time.time()



    print(f'DNN_IG Metrics in 4dp: \n: {report_df_dnn_absIG}')

    print(f'\nDNN_IG training takes: {(t2-t1):.2f}\n')



    t1 = time.time()

    report_df_cnn_absIG, history_cnn_absIG, model_cnn_absIG, y_true_cnn_absIG, y_pred_cnn_absIG = dnnmulti.trainCNNMultiClass(X_train_absIG, y_train_absIG, X_valid_absIG, y_valid_absIG, X_test_absIG, y_test_absIG, num_classes,
                        label_names, 128, dropout_rate=1e-2, l1_reg=1e-2, model_name="Feedforward_NNabs")

    t2 = time.time()


    print(f'CNN_IG Metrics in 4dp: \n: {report_df_cnn_absIG}')

    print(f'\nCNN_IG training takes: {(t2-t1):.2f}\n')

    # c. ResNet ABS IG

    t1 = time.time()

    report_df_resnet_absIG, history_resnet_absIG, model_resnet_absIG, y_true_resnet_absIG, y_pred_resnet_absIG = dnnmulti.trainResNetMultiClass(X_train_absIG, y_train_absIG, X_valid_absIG, y_valid_absIG, X_test_absIG, y_test_absIG, num_classes,
                        label_names, 128, dropout_rate=1e-2, l1_reg=1e-2, model_name="Feedforward_NNabs")

    t2 = time.time()




    print(f'ResNet_IG Metrics in 4dp: \n: {report_df_resnet_absIG}')

    print(f'\nResNet_IG training takes: {(t2-t1):.2f}\n')

    # i. ABS SHAP

    # Considering the top 25 encoded features from SHAP

    X_train_absSHAP, y_train_absSHAP,  X_valid_absSHAP, y_valid_absSHAP, X_test_absSHAP, y_test_absSHAP, feature_names_absSHAP, input_shape_absSHAP = dnnmulti.getSFDataSet(train_df, test_df,
                                                                            top25SHAPNSLKDDFeats, label, True)


    # # b. DNN ABS SHAP

    t1 = time.time()


    report_df_dnn_absSHAP, history_dnn_absSHAP, model_dnn_absSHAP, y_true_dnn_absSHAP, y_pred_dnn_absSHAP = dnnmulti.trainDNNMultiClass(X_train_absSHAP, y_train_absSHAP, X_valid_absSHAP, y_valid_absSHAP, X_test_absSHAP, y_test_absSHAP, num_classes,
                        label_names, 128, dropout_rate=1e-2, l1_reg=1e-2, model_name="Feedforward_NNabs")

    t2 = time.time()


    print(f'DNN_SHAP Metrics in 4dp: \n: {report_df_dnn_absSHAP}')

    print(f'\nDNN_SHAP training takes: {(t2-t1):.2f}\n')

    # b. CNN ABS SHAP
    t1 = time.time()


    report_df_cnn_absSHAP, history_cnn_absSHAP, model_cnn_absSHAP, y_true_cnn_absSHAP, y_pred_cnn_absSHAP = dnnmulti.trainCNNMultiClass(X_train_absSHAP, y_train_absSHAP, X_valid_absSHAP, y_valid_absSHAP, X_test_absSHAP, y_test_absSHAP, num_classes,
                        label_names, 128, dropout_rate=1e-2, l1_reg=1e-2, model_name="Feedforward_NNabs")

    t2 = time.time()

    print(f'CNN_SHAP Metrics in 4dp: \n: {report_df_cnn_absSHAP}')

    print(f'\nCNN_SHAP training takes: {(t2-t1):.2f}\n')


    # c ResNet ABS SHAP
    t1 = time.time()

    report_df_resnet_absSHAP, history_resnet_absSHAP, model_resnet_absSHAP, y_true_resnet_absSHAP, y_pred_resnet_absSHAP = dnnmulti.trainResNetMultiClass(X_train_absSHAP, y_train_absSHAP, X_valid_absSHAP, y_valid_absSHAP, X_test_absSHAP, y_test_absSHAP, num_classes,
                        label_names, 128, dropout_rate=1e-2, l1_reg=1e-2, model_name="Feedforward_NNabs")

    t2 = time.time()

    print(f'ResNet_SHAP Metrics in 4dp: \n: {report_df_resnet_absSHAP}')

    print(f'\nResNet_SHAP training takes: {(t2-t1):.2f}\n')

    t_stop = time.time()
    print(f'The script took a total of: {(t_stop-t_start):.2f}s')

def main():

    with open("output_results_nslkdd03.txt", "w") as f:
        sys.stdout = f

        HDMLFSNSLKDD_03()

        print(f'The End')
# --------------------------------------------------
if __name__ == '__main__':
    main()