from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1
import matplotlib.pyplot as plt

from tf_explain.core.integrated_gradients import IntegratedGradients
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

import shap

def compute_intgrd_explanations(model, X_test, feature_names, n_samples=5000):

    # Convert input to a TensorFlow tensor
    X_sample = X_test[:n_samples]  

    X_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)

    # Use IntegratedGradients for tabular data
    explainer = IntegratedGradients()

    # Generate explanations
    interpolated_images = IntegratedGradients.generate_interpolations(
        np.array(X_tensor), n_steps=50
    )

    # Get integrated gradients
    gradients = IntegratedGradients.get_integrated_gradients(
        interpolated_images, model, class_index=0, n_steps=50
    )

    # Compute feature importances by averaging absolute gradients
    feature_importances = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()

    # Create a DataFrame of feature importances
    feature_importances_df = pd.DataFrame({
        "Features": feature_names,
        "Importance": feature_importances
    })

    # Sort the DataFrame by 'Importance' in descending order
    feature_importances_df = feature_importances_df.sort_values(by="Importance", ascending=False)

    return feature_importances_df

# # Usage 
# feature_importancesDNN = compute_deeplift_explanations_tabular(modelDNN, X_test, feature_names, n_samples)

def plotImportance(feature_importances_df, figsize=(10, 6)):    

    # Plot the sorted feature importances
    plt.figure(figsize=figsize)
    plt.barh(
        feature_importances_df["Features"], 
        feature_importances_df["Importance"], 
        color="skyblue"
    )
    plt.title("Integrated Gradient Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.gca().invert_yaxis()  
    plt.show()


def shap_importance_multi(model, X_test, feature_names, n_samples=500):

    # Subset test data
    X_sample = np.array(X_test[:n_samples])  

    # Create SHAP GradientExplainer
    explainer = shap.GradientExplainer(model, X_sample)

    # Compute SHAP values
    shap_values = explainer.shap_values(X_sample)  

    # Process multi-class vs binary-class case
    if isinstance(shap_values, list):  
        shap_array = np.array(shap_values)  
        avg_shap_values = np.mean(np.abs(shap_array), axis=(0, 1))  
    else:  
        avg_shap_values = np.mean(np.abs(shap_values), axis=0) 

    # Ensure avg_shap_values is 1D
    avg_shap_values = avg_shap_values.mean(axis=1) if avg_shap_values.ndim == 2 else avg_shap_values

   
    print(f"Feature names length: {len(feature_names)}")
    print(f"SHAP values shape after aggregation: {avg_shap_values.shape}")  

    # Ensure feature_names and avg_shap_values have the same length
    if len(avg_shap_values) != len(feature_names):
        raise ValueError(f"Feature name length ({len(feature_names)}) and SHAP importance length ({len(avg_shap_values)}) mismatch!")

    # Create DataFrame
    feature_importances_df = pd.DataFrame({
        "Features": feature_names,
        "Importance_shap": avg_shap_values
    }).sort_values(by="Importance_shap", ascending=False)

    return feature_importances_df, shap_values, X_sample

# ******************************************************************************************
# Usage
# shap_imp_dnn, shap_values_dnn, X_sample_dnn = shap_importance_multi(model_dnn, X_test, numerical_columns2, 500)

def create_shap_bar_multi(shap_values, X_sample, feature_names, max_display=30):

    if isinstance(X_sample, np.ndarray):
        X_sample = pd.DataFrame(X_sample, columns=feature_names)

    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    if len(shap_values.shape) == 3:
        print("Detected multi-class SHAP values.")
        n_samples, n_features, n_classes = shap_values.shape

        shap_values_list = [shap_values[:, :, class_idx] for class_idx in range(n_classes)]

        # Compute mean absolute SHAP values across classes
        mean_abs_shap_values = np.mean(np.abs(shap_values), axis=2)  

        # Bar Plot with max_display adjusted
        print("Generating SHAP bar plot...")
        shap.summary_plot(
            mean_abs_shap_values,  
            X_sample,
            feature_names=feature_names,
            plot_type="bar",
            max_display=max_display  
        )
    else:
        raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

# Usage
# create_shap_bar_multi(shap_values_dnn, X_sample_dnn, numerical_columns2, max_display=20)

# **********************************************************************************************
def create_shap_waterfall_multi(shap_values, X_sample, feature_names, instance_idx=0, max_display=20):

    # Ensure X_sample is a pandas DataFrame
    if isinstance(X_sample, np.ndarray):
        X_sample = pd.DataFrame(X_sample, columns=feature_names)

    # Ensure feature_names is a simple list (not an array)
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    # Check if multi-class (SHAP values is 3D: (n_samples, n_features, n_classes))
    if len(shap_values.shape) == 3:
        print(f"Generating SHAP waterfall plot for instance {instance_idx}...")

        n_samples, n_features, n_classes = shap_values.shape

        # Convert 3D array into a list of (n_samples, n_features) arrays (one per class)
        shap_values_list = [shap_values[:, :, class_idx] for class_idx in range(n_classes)]

        instance_data = X_sample.iloc[instance_idx]  # 
        for class_idx in range(n_classes):
            instance_shap_values = shap_values_list[class_idx][instance_idx]  
            base_value = np.mean(shap_values_list[class_idx])  

            # Limit the number of features displayed in the waterfall plot
            if len(instance_shap_values) > max_display:
                instance_shap_values = instance_shap_values[:max_display]
                instance_data = instance_data[:max_display]
                feature_names_sub = feature_names[:max_display]
            else:
                feature_names_sub = feature_names

            shap.waterfall_plot(
                shap.Explanation(
                    values=instance_shap_values,
                    base_values=base_value,
                    data=instance_data.values,  
                    feature_names=feature_names_sub
                )
            )
    else:
        raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

# # Usage
# create_shap_waterfall_multi(shap_values_dnn, X_sample_dnn, numerical_columns2, instance_idx=0, max_display=20)

# **********************************************************************************************
import shap
import numpy as np
import pandas as pd

def create_shap_summary_multi(shap_values, X_sample, feature_names, max_display=30):

    # Ensure X_sample is a pandas DataFrame
    if isinstance(X_sample, np.ndarray):
        X_sample = pd.DataFrame(X_sample, columns=feature_names)

    # Ensure feature_names is a simple list (not an array)
    if isinstance(feature_names, np.ndarray):
        feature_names = feature_names.tolist()

    # Check if SHAP values are for multi-class classification (3D array: (n_samples, n_features, n_classes))
    if len(shap_values.shape) == 3:
        print("Detected multi-class SHAP values.")

        n_samples, n_features, n_classes = shap_values.shape

        # Convert 3D array into a list of (n_samples, n_features) arrays (one per class)
        shap_values_list = [shap_values[:, :, class_idx] for class_idx in range(n_classes)]

        # Summary Plot (beeswarm) for each class
        print("Generating SHAP summary plot for multi-class classification...")
        for class_idx in range(n_classes):
            print(f"Class {class_idx}:")
            shap.summary_plot(
                shap_values_list[class_idx],
                X_sample,
                feature_names=feature_names,
                max_display=max_display  
            )
    else:
        raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

# # Usage
# create_shap_summary_multi(shap_values_dnn, X_sample_dnn, numerical_columns2, max_display=30)

# **********************************************************************************************
def create_shap_summary_aggregated(shap_values, X_sample, feature_names):

    # Aggregate SHAP values across classes
    shap_values_mean = np.mean(shap_values, axis=2)  # Shape: (n_samples, n_features)

    # Generate summary plot
    shap.summary_plot(shap_values_mean, X_sample, feature_names=feature_names)

# Usage
# create_shap_summary_aggregated(shap_values_dnn, X_sample_dnn, numerical_columns2)


# **********************************************************************************************
def create_shap_waterfall_aggregated(shap_values, X_sample, feature_names, instance_idx=0, max_features=20):

    # Select the SHAP values for the instance across all classes
    instance_shap_values = shap_values[instance_idx]  
    # Aggregate SHAP values across classes (mean impact per feature)
    aggregated_shap_values = np.mean(instance_shap_values, axis=1)  
    # Select instance feature values
    instance_data = X_sample[instance_idx]  

    # Compute absolute SHAP importance and sort by magnitude
    top_indices = np.argsort(np.abs(aggregated_shap_values))[::-1][:max_features]
    top_features = np.array(feature_names)[top_indices]
    top_shap_values = aggregated_shap_values[top_indices]
    top_instance_data = instance_data[top_indices]

    # Compute base value (mean prediction)
    base_value = np.mean(aggregated_shap_values)  

    # Create SHAP explanation object
    shap_exp = shap.Explanation(
        values=top_shap_values,
        base_values=base_value,
        data=top_instance_data,
        feature_names=top_features
    )

    # Plot the waterfall
    shap.waterfall_plot(shap_exp)

# # Usage
# create_shap_waterfall_aggregated(shap_values_dnn, X_sample_dnn, numerical_columns2)


# **********************************************************************************************
# Selecting features from Integratec Gradient and Feature Importances 
def select_IGShapFeatures(feature_importancesDNN, shap_imp_dnn, thresh=0.5):
    df_IGShap = pd.merge(feature_importancesDNN, shap_imp_dnn, on='Features', how='inner')
    
    # Ensure at least half of the features are selected
    max_thresh = int(df_IGShap.shape[0] * 0.5)
    thresholdIGShap = max(int(df_IGShap.shape[0] * thresh), max_thresh)  
    df_IGShap['IG Rank'] = df_IGShap['Importance'].rank(ascending=False)
    df_IGShap['Shap Rank'] = df_IGShap['Importance_shap'].rank(ascending=False)
    df_IGShap['Selected'] = (df_IGShap['IG Rank'] <= thresholdIGShap) & (df_IGShap['Shap Rank'] <= thresholdIGShap)

    selected_features = df_IGShap[df_IGShap['Selected']]['Features'].tolist()

    # Ensure at least half of the features are selected
    if len(selected_features) < max_thresh:
        additional_features = df_IGShap[~df_IGShap['Selected']].nlargest(max_thresh - len(selected_features), ['IG Rank', 'Shap Rank'])['Features'].tolist()
        selected_features.extend(additional_features)
    return selected_features

# Usage 
# sf_igshapDnn = select_IGShapFeatures(feature_importancesDNN, shap_imp_dnn, thresh=0.5, max_thresh=30)

# **********************************************************************************************
