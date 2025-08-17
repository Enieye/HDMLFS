import pandas as pd
import numpy as np
import copy

from sklearn.base import clone
import time
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, auc, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler


# Define preprocessing functions
def remove_zero_entries(df):

    # Remove rows with all zero entries
    df2 = df.loc[~(df == 0).all(axis=1)]
    # Remove columns with all zero entries
    df2 = df2.loc[:, ~(df2 == 0).all(axis=0)]
    # List of removed features due to zero values
    zero_value_features = list(set(list(df.columns))-set(list(df2.columns)))
    print(f'The features that were removed for having zero values are {zero_value_features}')
    return df2, zero_value_features


def remove_zero_variance(df):

    # Identify numerical columns with zero variance
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    zero_variance_columns = [col for col in numerical_columns if df[col].var() == 0]
    
    # Identify categorical columns with only one unique value
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    single_category_columns = [col for col in categorical_columns if df[col].nunique() == 1]
    
    # Combine columns to be removed
    zero_variance_features = zero_variance_columns + single_category_columns
    print(f'The features that were removed for having zero variance are {zero_variance_features}')

    # Remove identified columns
    df2 = df.drop(columns=zero_variance_features)
    
    return df2, zero_variance_features

def uncommonAttack(df, activity_column, tempCol=None):
    if tempCol is not None:
        uniqueTrain = list(df[df[tempCol]==0][activity_column].unique())
        uniqueTest = list(df[df[tempCol]==1][activity_column].unique())
        uncommonListAttack = list(set(uniqueTrain).symmetric_difference(set(uniqueTest)))
        return df[~df[activity_column].isin(uncommonListAttack)]
    else:
        return df  

def exclude_rare_activity_types(df, activity_column, min_instances=200):

    # Count instances of each activity type
    activity_counts = df[activity_column].value_counts()
    # Filter out activity types with fewer than min_instances
    valid_activities = activity_counts[activity_counts >= min_instances].index
    common_attack_list = list(valid_activities)
    uncommon_attack_list = list(set(list(activity_counts.index)) - set(list(valid_activities)))
    print(f'Common attack types are: {common_attack_list}')
    print(f'The number of uncommon attacks types are {len(uncommon_attack_list)}. \n And they are : {uncommon_attack_list}')
    df2 = df[df[activity_column].isin(valid_activities)]
    print(f'The number of rows removed because of uncommon attacks is {df.shape[0] - df2.shape[0]}')
    return df2

def find_correlated_columns(df, threshold=0.9):

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated_columns_dicts = []
    
    for column in upper.columns:
        correlated_columns = list(upper.index[upper[column] > threshold])
        if correlated_columns:
            correlated_columns_dicts.append({column: correlated_columns})
    
    return correlated_columns_dicts

def feature_selection(correlations):

    # Convert list of dictionaries to a single dictionary
    correlation_dict = {k: v for d in correlations for k, v in d.items()}
    
    # Initialize selected and removed feature sets
    selected_features = set()
    removed_features = set()

    # Make a deep copy of the correlations to avoid mutation
    correlations_copy = copy.deepcopy(correlation_dict)
    
    # Group correlations to handle complex dependencies
    while correlations_copy:
        # Find the feature with the most correlations (for more robust selection)
        max_corr_feature = max(correlations_copy, key=lambda k: len(correlations_copy[k]))
        corr_features = correlations_copy.pop(max_corr_feature)
        
        # Add the most connected feature to selected_features
        selected_features.add(max_corr_feature)
        
        # Add correlated features to removed_features
        removed_features.update(corr_features)
        
        # Remove selected and removed features from other correlation entries
        correlations_copy = {k: [f for f in v if f not in selected_features and f not in removed_features]
                             for k, v in correlations_copy.items() if k not in removed_features}

    # Convert sets to lists
    selected_features = list(selected_features)
    removed_features = list(removed_features)
    
    return selected_features, removed_features

def preprocess_dataset(df, activity_column, tempCol=None, min_instances=200, corr_threshold=0.9):

    # df2, zero_values_features = remove_zero_entries(df)

    df2, zero_variance_features = remove_zero_variance(df)

    df_filtered = uncommonAttack(df2, activity_column, tempCol) 

    df3 = exclude_rare_activity_types(df_filtered, activity_column, min_instances)
    
    # Remove the activity column to compute correlations on features only
    features_df = df3.drop(columns=[activity_column])
    
    # Find correlated columns
    correlated_columns = find_correlated_columns(features_df, corr_threshold)
    print(f"Correlated columns (threshold > {corr_threshold}):", correlated_columns)
    
    # Perform feature selection
    selected_features, removed_features = feature_selection(correlated_columns)
    print(f"Selected features due to correlation: {selected_features}")
    print(f"Removed features due to correlation: {removed_features}")
    
    removed_corr_and_zero_value_features = zero_variance_features + removed_features
    print(f'Features removed due to zero values and correlation {removed_corr_and_zero_value_features}')
    
    # Removing from the dataframe the features that were not selected
    preprocessed_df = df3.drop(removed_features, axis=1)
    
    return preprocessed_df

def df_metrics(base_model, X_train, y_train, X_test, y_test):
       
    model = clone(base_model)
    t1 = time.time()
    model.fit(X_train, y_train)
    t2 = time.time()
    t_diff = t2 - t1
    y_pred = model.predict(X_test)
    model_name = type(model).__name__
    train_accuracy = round(model.score(X_train, y_train), 4)
    test_accuracy = round(accuracy_score(y_test, y_pred), 4)
    auc = round(roc_auc_score(y_true=y_test, y_score=y_pred),4)
    precision = round(precision_score(y_true=y_test, y_pred=y_pred), 4)
    recall = round(recall_score(y_true=y_test, y_pred=y_pred), 4)
    f1 = round(f1_score(y_true=y_test, y_pred=y_pred), 4)
    df = pd.DataFrame(
            data=[[
                model_name,
                train_accuracy,
                test_accuracy, 
                auc,
                precision,
                recall,
                f1,
                t_diff]],
            columns = ['Model', 'Train_Accuracy', 'Test_Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'Time(s)']
        )
    confusion_mat =  confusion_matrix(y_test, y_pred)
    return df, confusion_mat, model


def df_metrics2(base_model, X_train, y_train, X_test, y_test):
    
   
    model = clone(base_model)
    t1 = time.time()
    model.fit(X_train, y_train)
    t2 = time.time()
    t_diff = t2 - t1
    y_pred = model.predict(X_test)
    model_name = type(model).__name__
    train_accuracy = round(model.score(X_train, y_train), 4)
    test_accuracy = round(accuracy_score(y_test, y_pred), 4)
    auc = round(roc_auc_score(y_true=y_test, y_score=y_pred),4)
    precision = round(precision_score(y_true=y_test, y_pred=y_pred), 4)
    recall = round(recall_score(y_true=y_test, y_pred=y_pred), 4)
    f1 = round(f1_score(y_true=y_test, y_pred=y_pred), 4)
    df = pd.DataFrame(
            data=[[
                model_name,
                train_accuracy,
                test_accuracy, 
                auc,
                precision,
                recall,
                f1,
                t_diff]],
            columns = ['Model', 'Train_Accuracy', 'Test_Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'Time(s)']
        )
    confusion_mat =  confusion_matrix(y_test, y_pred)
    return df, confusion_mat, model

# Preprocessing the data to give Xtrain, ytrain, Xtext, ytest
def dataModel(df, label, tempCol=None):
    numeric_features = []
    scaler = StandardScaler()
    if tempCol is not None:
        numeric_features = df.drop(tempCol, axis=1).dtypes[df.drop(tempCol, axis=1).dtypes!='object'].index
        scaler.fit(df[df[tempCol]==0][numeric_features])
        df[numeric_features] = scaler.transform(df[numeric_features])
        df2 = df.copy()
        df2[label] = np.where(df2[label] == 'normal', 0, 1)
        df3 = pd.get_dummies(df2)
        X = df3.drop([label, tempCol], axis=1)
        y = df3[label]
        n_train = len(df3[df3[tempCol]==0])
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_test = X[n_train:]
        y_test = y[n_train:]
    else:
        pass

    return (X_train, y_train, X_test, y_test)


import seaborn as sns
import matplotlib.pyplot as plt

# ***********************************************************************************************************

def plot_correlation_heatmap(df):

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Compute the correlation matrix and round values to 2 decimal places
    corr_matrix = numeric_df.corr().round(2)
    
    # Set up the matplotlib figure with a blue background
    plt.figure(figsize=(16, 14), facecolor='lightblue')
    
    # Draw the heatmap with rounded values and adjust the font size
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Blues', center=0, cbar_kws={"shrink": .8}, annot_kws={"size": 12})
    
    # Add title and labels
    plt.title('Correlation Heatmap', fontsize=20)
    plt.show()

    import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ***********************************************************************************************************
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_heatmap2(df, figsize=(16, 16), cmap='coolwarm', annot=True):

    # Compute the correlation matrix
    corr_matrix = df.corr()
    
    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=annot,
        cmap=cmap,
        fmt=".2f",
        linewidths=0.5,
        vmin=-1,  # Correlation ranges from -1 to 1
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
    )
    
    # Add titles and labels
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming 'df' is your pandas DataFrame with numerical features
# plot_correlation_heatmap(df)
# ***********************************************************************************************************



def plot_correlation_heatmap_no_numbers(df, figsize=(12, 10), cmap='coolwarm'):

    # Compute the correlation matrix
    corr_matrix = df.corr()
    
    # Create the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=False,  # Do not show the correlation values
        cmap=cmap,
        linewidths=0.5,
        vmin=-1,  # Correlation ranges from -1 to 1
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"}
    )
    
    # Add titles and labels
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

# Example usage:
# Assuming 'df' is your pandas DataFrame with numerical features
# plot_correlation_heatmap_no_numbers(df)


def getBin(df, label):
    benign = df[label].value_counts().index[0]
    label = label
    temp = df[label]
    df['Threat'] = np.where(df[label]== benign, 0, 1)
    df = df.drop(label, axis=1)    
    df = pd.concat([df, temp], axis=1)
    return df
