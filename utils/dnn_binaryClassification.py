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


# *****************************************************************
# Define a residual block
# *****************************************************************
def residual_block(x, units, l1_lambda=0.01):
    shortcut = x
    # First dense layer with regularization
    x = Dense(units, kernel_regularizer=l1(l1_lambda))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = Dense(units)(x)
    x = BatchNormalization()(x)
    
    # Second dense layer with L1 regularization
    x = Dense(units, kernel_regularizer=l1(l1_lambda))(x)
    x = BatchNormalization()(x)
    
    # Add the shortcut connection
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

# *****************************************************************
# Custom Residual Network for structured data
# *****************************************************************
def create_resnet_for_structured_data(input_shape, l1_lambda=0.01):
    inputs = Input(shape=input_shape)
    
    # Initial Dense layer
    x = Dense(32, kernel_regularizer=l1(l1_lambda))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Add multiple residual blocks
    for _ in range(3):  
        x = residual_block(x, 32)
    
    # Final Dense layer for binary classification
    output = Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = Model(inputs=inputs, outputs=output)
    
    return model


def plotModel(history, trunc=0):
    # Getting the loss and metrics 
    loss = history.history['loss'][trunc:]
    val_loss = history.history['val_loss'][trunc:]
    acc = history.history['accuracy'][trunc:]
    val_acc = history.history['val_accuracy'][trunc:]
    
    # Getting the epochs
    epochs = range(1, len(loss) + 1)  
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # For loss
    ax1.plot(epochs, loss, 'bo', label='Training loss')
    ax1.plot(epochs, val_loss, 'r', label='Validation loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # For accuracy
    ax2.plot(epochs, acc, 'bo', label='Training accuracy')
    ax2.plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()  
    plt.show()



# Define the callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',     
    patience=20,            
    restore_best_weights=True  
)

model_checkpoint = ModelCheckpoint(
    'best_model.keras',        
    monitor='val_loss',      
    save_best_only=True     #
)


#**************************************************************************************
# The refactored functions
#**************************************************************************************
# Defining modular functions
def build_modelFC(input_shape):
    """Builds a feedforward neural network model."""
    model = keras.Sequential([
        layers.Dense(256, activation="relu", input_shape=input_shape),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    return model

def compile_model(model, lr_rate=0.001):
    """Compiles the model with specified optimizer and metrics."""
    optimizer = keras.optimizers.Adam(learning_rate=lr_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )

def fit_model(model, X_train, y_train, X_valid, y_valid, epochs=20, batch_size=32):
    """Fits the model with early stopping callback."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[early_stopping]
    )
    return history


def evaluate_model(model, history, X_test, y_test, model_name="Feedforward_NN"):
    """Evaluates the model on the test set and returns metrics as a dataframe."""
    eval_metrics = model.evaluate(X_test, y_test, verbose=0)
    metrics_dict = {
        'model_name': model_name,
        'train_accuracy': round(history.history['accuracy'][-1], 4),  
        'val_accuracy': round(history.history['val_accuracy'][-1], 4),  # 
        'test_accuracy': round(eval_metrics[1], 4),
        'test_precision': round(eval_metrics[2], 4),
        'test_recall': round(eval_metrics[3], 4),
        'test_f1_score': round(2 * (eval_metrics[2] * eval_metrics[3]) / (eval_metrics[2] + eval_metrics[3]), 4) if eval_metrics[2] + eval_metrics[3] > 0 else 0.0,
        'test_auc': round(eval_metrics[4], 4)
    }
    return pd.DataFrame([metrics_dict])


def compute_deeplift_explanations_tabular(model, X_sample, feature_names):

    X_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)

    explainer = IntegratedGradients()

    # Generate explanations
    interpolated_images = IntegratedGradients.generate_interpolations(
        np.array(X_tensor), n_steps=50
    )

    gradients = IntegratedGradients.get_integrated_gradients(
        interpolated_images, model, class_index=0, n_steps=50
    )

    feature_importances = tf.reduce_mean(tf.abs(gradients), axis=0).numpy()

    feature_importances_df = pd.DataFrame({
        "Features": feature_names,
        "Importance": feature_importances
    })

    feature_importances_df = feature_importances_df.sort_values(by="Importance", ascending=False)

    return feature_importances_df

def plotImportance(feature_importances_df):    

    # Plot the sorted feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_importances_df["Features"], 
        feature_importances_df["Importance"], 
        color="skyblue"
    )
    plt.title("Sorted Feature Importances from DeepLIFT (Tabular)")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.gca().invert_yaxis()  
    plt.show()


# ***********************************************************************************
# Computing the shap feature importances and shap_values, 
import shap
import pandas as pd

def compute_shap_feature_importance(model, X_train, X_test, background_size, feature_names):

    # Randomly select a subset of the training data as background data
    X_train_sampled = X_train[np.random.choice(X_train.shape[0], background_size, replace=False)]

    explainer = shap.DeepExplainer(model, X_train_sampled)

    X_test_sampled = X_test[np.random.choice(X_test.shape[0], background_size, replace=False)]

    # Compute SHAP values
    shap_values = explainer.shap_values(X_test_sampled)

    shap_values_squeezed = np.squeeze(shap_values)

    feature_importances = np.mean(np.abs(shap_values_squeezed), axis=0)

    importance_df = pd.DataFrame({
        'Features': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Plot the SHAP summary plot
    shap.summary_plot(shap_values_squeezed, X_test_sampled, feature_names=feature_names)

    return importance_df, shap_values_squeezed, X_test_sampled

# # Usage 
# shap_importance, shap_values, X_test_sampled = compute_shap_feature_importance(modelFC, X_train, X_test, 500, feature_names)

# ***********************************************************************************
# Generating waterfall_plot and bar_plot from the return values from above
def create_shap_plots(shap_importance_df, shap_values, X_sample, feature_names, instance_idx=0):

    # Bar Plot
    print("Generating SHAP bar plot...")
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar"
    )

    # Waterfall Plot (for a single instance)
    print("Generating SHAP waterfall plot...")
    instance_shap_values = shap_values[instance_idx]  # 
    instance_data = X_sample[instance_idx]  # 
    base_value = np.mean(instance_shap_values)  

    shap.waterfall_plot(
        shap.Explanation(
            values=instance_shap_values,
            base_values=base_value,  
            data=instance_data,
            feature_names=feature_names
        )
    )

# # Usage
# create_shap_plots(shap_importance, shap_values, X_test_sampled, feature_names, instance_idx=0)

# ****************************Getting n_top_features for SHAP feature importance **********************

# def get_cross_validated_features(shap_importance_df, X_train, y_train, top_n_values, n_splits=5):
def get_cross_validated_features(importance_df, X_train, y_train, top_n_values, n_splits=5):

    results = []

    # Perform cross-validation for different top_n values
    for top_n in top_n_values:
        top_features = importance_df['Features'].iloc[:top_n].values
        top_n_indices = [i for i, feature in enumerate(importance_df['Features']) if feature in top_features]
        
        X_train_top_n = X_train[:, top_n_indices]
        # X_test_top_n = X_test[:, top_n_indices]

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []

        for train_idx, valid_idx in kf.split(X_train_top_n):
            # Split data into training and validation sets
            X_fold_train, X_fold_valid = X_train_top_n[train_idx], X_train_top_n[valid_idx]
            y_fold_train, y_fold_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            
            input_shape = (X_fold_train.shape[1],)
            model = build_modelFC(input_shape)
            compile_model(model)
            history = fit_model(model, X_fold_train, y_fold_train, X_fold_valid, y_fold_valid, 10, 64)

            eval_metrics = model.evaluate(X_fold_valid, y_fold_valid, verbose=0)
            accuracy = eval_metrics[1]
            precision = eval_metrics[2]
            recall = eval_metrics[3]
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
            fold_metrics.append((accuracy, precision, recall, f1_score))
        
        # Compute mean and standard deviation for metrics across folds
        accuracies, precisions, recalls, f1_scores = zip(*fold_metrics)
        results.append({
            'top_n': top_n,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_precision': np.mean(precisions),
            'std_precision': np.std(precisions),
            'mean_recall': np.mean(recalls),
            'std_recall': np.std(recalls),
            'mean_f1_score': np.mean(f1_scores),
            'std_f1_score': np.std(f1_scores),
        })

    results_df = pd.DataFrame(results)
    return results_df


# *** Plotting the cross_val_results 
def plot_cross_val_results(cross_val_results):

    cross_val_results = cross_val_results.set_index('top_n')

    metrics = ['mean_accuracy', 'mean_precision', 'mean_recall', 'mean_f1_score']
    stds = ['std_accuracy', 'std_precision', 'std_recall', 'std_f1_score']

    plt.figure(figsize=(10, 6))
    for metric, std in zip(metrics, stds):
        plt.errorbar(
            cross_val_results.index,
            cross_val_results[metric],
            yerr=cross_val_results[std],
            label=metric.capitalize(),
            marker='o',
            linestyle='-'
        )

    plt.title("Cross-Validation Metrics vs Top N Features")
    plt.xlabel("Top N Features")
    plt.ylabel("Metric Value")
    plt.legend(title="Metrics")
    plt.grid(True)
    plt.show()

def get_best_featuresAndModel(X_train, y_train, X_valid, y_valid, X_test, y_test, feature_names, importance_df, n_best):
    bestFeatures = list(importance_df[:n_best]['Features'])

    if len(feature_names) != X_train.shape[1]:
        raise ValueError("The number of feature names does not match the number of columns in X_train.")

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_valid_df = pd.DataFrame(X_valid, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    X_train_FS = X_train_df[bestFeatures]
    X_valid_FS = X_valid_df[bestFeatures]
    X_test_FS = X_test_df[bestFeatures]

    X_trainvalid = pd.concat([X_train_FS, X_valid_FS], ignore_index=True)
    y_trainvalid = pd.concat([y_train, y_valid], ignore_index=True)


    modelFinal = build_modelFC((n_best,))
    compile_model(modelFinal, lr_rate=0.001)
    history = modelFinal.fit(X_trainvalid, y_trainvalid, epochs=20, batch_size=32)

    eval_metrics = modelFinal.evaluate(X_test_FS, y_test)
    metrics_dict = {
        'train_accuracy': round(history.history['accuracy'][-1], 4),  
        'test_accuracy': round(eval_metrics[1], 4),
        'test_precision': round(eval_metrics[2], 4),
        'test_recall': round(eval_metrics[3], 4),
        'test_f1_score': round(2 * (eval_metrics[2] * eval_metrics[3]) / (eval_metrics[2] + eval_metrics[3]), 4) if eval_metrics[2] + eval_metrics[3] > 0 else 0.0,
        'test_auc': round(eval_metrics[4], 4)
    }

    modelFinal_df = pd.DataFrame([metrics_dict])

    return modelFinal_df, history, bestFeatures

def get_best_ModelIGShap(X_train, y_train, X_valid, y_valid, X_test, y_test, feature_names, selected_features):
    bestFeatures = selected_features
    if len(feature_names) != X_train.shape[1]:
        raise ValueError("The number of feature names does not match the number of columns in X_train.")

    # Create the dataframe
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_valid_df = pd.DataFrame(X_valid, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    X_train_FS = X_train_df[bestFeatures]
    X_valid_FS = X_valid_df[bestFeatures]
    X_test_FS = X_test_df[bestFeatures]

    X_trainvalid = pd.concat([X_train_FS, X_valid_FS], ignore_index=True)
    y_trainvalid = pd.concat([y_train, y_valid], ignore_index=True)

    modelFinal = build_modelFC(((len(bestFeatures)),))
    compile_model(modelFinal, lr_rate=0.001)
    history = modelFinal.fit(X_trainvalid, y_trainvalid, epochs=20, batch_size=32)

    eval_metrics = modelFinal.evaluate(X_test_FS, y_test)
    metrics_dict = {
        'train_accuracy': round(history.history['accuracy'][-1], 4),  
        'test_accuracy': round(eval_metrics[1], 4),
        'test_precision': round(eval_metrics[2], 4),
        'test_recall': round(eval_metrics[3], 4),
        'test_f1_score': round(2 * (eval_metrics[2] * eval_metrics[3]) / (eval_metrics[2] + eval_metrics[3]), 4) if eval_metrics[2] + eval_metrics[3] > 0 else 0.0,
        'test_auc': round(eval_metrics[4], 4)
    }

    modelFinal_df = pd.DataFrame([metrics_dict])

    return modelFinal_df, history, bestFeatures

# ## **** Usage: **** 
# modelFinalIGShap_df, history_finalIGShap = get_best_ModelIGShap(X_train, y_train, X_valid, y_valid,
#                                                                 X_test, y_test, feature_names, selected_features)

# Focal loss implementation
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss_value = -alpha_t * tf.pow(1 - p_t, gamma) * tf.math.log(tf.clip_by_value(p_t, 1e-7, 1.0))
        return tf.reduce_mean(focal_loss_value)
    # use this in the compile_fl function below
    return focal_loss_fixed

## Usage in 
def compile_fl(model, lr_rate=0.001):
    optimizer = keras.optimizers.Adam(learning_rate=lr_rate)
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )


### CNN (with focal loss)
# Defining modular functions
def build_cnn(input_shape):
    model = keras.Sequential([
        layers.Conv1D(64, kernel_size=3, activation="relu", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv1D(32, kernel_size=3, activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(1, activation="sigmoid")
    ])
    return model


def build_resnet50_1d(input_shape):
    """Builds a ResNet-50 inspired Conv1D model."""
    inputs = layers.Input(shape=input_shape)

    # Initial Conv Layer
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(x)

    def residual_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = layers.Conv1D(filters, kernel_size=kernel_size, strides=stride, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters, kernel_size=kernel_size, strides=1, padding="same")(x)
        x = layers.BatchNormalization()(x)

        # Adjust shortcut dimensions if needed
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, kernel_size=1, strides=stride, padding="same")(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        return x

    for filters, strides in zip([64, 128, 256, 512], [1, 2, 2, 2]):
        x = residual_block(x, filters, stride=strides)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    return model

def get_best_trainvalid_features(X_train, y_train, X_valid, y_valid, X_test, y_test, feature_names, selected_features):
    bestFeatures = selected_features
    if len(feature_names) != X_train.shape[1]:
        raise ValueError("The number of feature names does not match the number of columns in X_train.")

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_valid_df = pd.DataFrame(X_valid, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    X_train_FS = X_train_df[bestFeatures]
    X_valid_FS = X_valid_df[bestFeatures]
    X_test_FS = X_test_df[bestFeatures]

    X_trainvalid = pd.concat([X_train_FS, X_valid_FS], ignore_index=True)
    y_trainvalid = pd.concat([y_train, y_valid], ignore_index=True)

    return X_trainvalid, y_trainvalid, X_test_FS, y_test