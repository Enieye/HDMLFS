#!/usr/bin/env python3

import pandas as pd
import numpy as np
# import utility

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.base import clone
import time
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, auc, recall_score, precision_score, f1_score

from sklearn.preprocessing import StandardScaler
from datetime import datetime

import os
import  sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

sys.path.append(parent_dir)

import utils.utility as utility


def HDMLFSNSLKDD_01():
    t_start = time.time()

    data_dir = 'path/to/HDMLFS/data_folder'

    dataset_train_paths = data_dir + 'KDDTrain+.txt'
    dataset_test_paths = data_dir + 'KDDTest+.txt'

    #Loading -datasets into dataframe
    df_train = pd.read_csv(dataset_train_paths, header=None)
    df_test = pd.read_csv(dataset_test_paths, header=None)

    df_train.head(2)

    df_test.head(2)

    df_train.shape, df_test.shape

    #Reset column names for training set
    df_train.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
    'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
    'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'attack', 'outcome']


    df_train.head(2)

    #Reset column names for test set
    df_test.columns = df_train.columns

    df_test.head(2)

    df_train.outcome.nunique()

    df_test.outcome.nunique()

    # Combining the training set and test set for the purpose of preprocessing
    # Before concating, add a column to each of the training and test set to distinguish them later
    df_train[ 'set_type'] = 0
    df_test['set_type'] = 1


    # concatenating them
    df = pd.concat((df_train.iloc[:,:], df_test.iloc[:,:]))

    # Resetting the index
    df = df.reset_index(drop=True)

    df_sf = utility.preprocess_dataset(df, 'attack')

    # Saving the combined data into a file
    df.to_csv(data_dir+'preprocessed_featureselection/01fulltraintest3.csv', index=False)

    t_stop = time.time()
    print(f'The script took a total of: {(t_stop-t_start):.2f}s')

# --------------------------------------------------
def main():

    with open("output_results_nslkdd01.txt", "w") as f:
        sys.stdout = f
        
        now = datetime.now()

        print(f'The script is starting . . . . . {now.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        HDMLFSNSLKDD_01()

        print(f'\n\nThe End')
# --------------------------------------------------
if __name__ == '__main__':
    main()
