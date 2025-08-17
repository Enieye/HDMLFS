# For the Deep Learning Model
# Updated Modular Functions for Multi-Class Classification
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras import Sequential, layers, regularizers

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder



def build_multi(input_shape, num_classes, dropout_rate=0.1, l1_reg=1e-6):
    model = Sequential([
        layers.Dense(256, activation="relu", input_shape=input_shape,
                     kernel_regularizer=regularizers.l1(l1_reg)),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l1(l1_reg)),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation="softmax")  
    ])
    return model


# ********************************************************************************************
def compile_multi(model, lr_rate=0.001):
    optimizer = Adam(learning_rate=lr_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',  
        metrics=['accuracy']
    )
# ********************************************************************************************
def fit_multi(model, X_train, y_train, X_valid, y_valid, epochs=20, batch_size=64):
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stopping]
    )
    return history, model

# ********************************************************************************************
def evaluate_multi(model, X_test, y_test, label_names, model_name="Feedforward_NN"):

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names))

    # Return report as a dataframe for further analysis
    return pd.DataFrame(report).transpose().round(4), y_true, y_pred

# ********************************************************************************************

def dnn_multiConfxMtrx(y_true, y_pred, label_names):  
    cm=confusion_matrix(y_true, y_pred)
    f,ax=plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax, cmap=sns.color_palette("coolwarm", as_cmap=True),
            xticklabels=label_names, yticklabels=label_names    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


# **************************************************************************************************
def prepData(df, numerical_columns, removed_columns, labels, test_size=0.1):
    df2  = df.copy()
    numerical_columns2 = list(set(numerical_columns) - set(removed_columns))
    df2 = df2.drop(removed_columns, axis=1)
    train_df, test_df = train_test_split(df2, test_size=test_size, random_state=2, shuffle=True, stratify=df2[labels[0]])

    le = LabelEncoder()
    train_df[labels[0]] = le.fit_transform(train_df[labels[0]])
    test_df[labels[0]] = le.transform(test_df[labels[0]])

    minmax_scaler = MinMaxScaler()
    train_df[numerical_columns2] = minmax_scaler.fit_transform(train_df[numerical_columns2])
    test_df[numerical_columns2] = minmax_scaler.transform(test_df[numerical_columns2])
    
    return train_df, test_df, numerical_columns2

# **************************************************************************************************

from sklearn.utils import resample
import pandas as pd

def resample_to_fraction(df, label_col, frac=0.1):

    max_class_size = df[label_col].value_counts().max()
    target_size = int(frac * max_class_size)
    
    resampled_data = []
    
    for category in df[label_col].unique():
        category_data = df[df[label_col] == category]
        
        if len(category_data) < target_size:
            
            category_data = resample(category_data,
                                     replace=True,
                                     n_samples=target_size,
                                     random_state=42)
        elif len(category_data) > target_size:
            
            category_data = resample(category_data,
                                     replace=False,
                                     n_samples=target_size,
                                     random_state=42)
        
        resampled_data.append(category_data)
    
    return pd.concat(resampled_data)

# **************************************************************************************************

# # # Usage: 
def generateDataSet(df_train, df_test, labels, validSet=False, test_size=0.1):

    num_classes = df_train[labels[0]].nunique()
    X_train = df_train.drop(labels, axis=1).copy()
    y_train = df_train[[labels[0]]].copy()
    X_test = df_test.drop(labels, axis=1).copy()
    y_test = df_test[[labels[0]]].copy()

    feature_names = list(X_train.columns)
    input_shape = (X_train.shape[1],)


    if validSet == True:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=test_size, random_state=42)

        X_train = X_train.values
        X_valid = X_valid.values
        X_test = X_test.values

        y_train = to_categorical(y_train, num_classes)
        y_valid = to_categorical(y_valid, num_classes)
        y_test = to_categorical(y_test, num_classes)

    else:
               
        X_train = X_train.values
        X_valid = None
        X_test = X_test.values

        y_train = to_categorical(y_train, num_classes)
        y_valid = None
        y_test = to_categorical(y_test, num_classes) 

    return X_train, y_train, X_valid, y_valid, X_test, y_test, feature_names, input_shape 

# ********************************************************************************************
   
def getSFDataSet(df_train, df_test, selected_features, labels, validSet=False, test_size=0.1):
    num_classes = df_train[labels[0]].nunique()
    
    X_train = df_train[selected_features].copy()
    y_train = df_train[[labels[0]]].copy()
    X_test = df_test[selected_features].copy()
    y_test = df_test[[labels[0]]].copy()

    feature_names = list(X_train.columns)
    input_shape = (X_train.shape[1],)

    if validSet == True:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=test_size, random_state=42)
        
        X_train = X_train.values
        X_valid = X_valid.values
        X_test = X_test.values

        y_train = to_categorical(y_train, num_classes)
        y_valid = to_categorical(y_valid, num_classes)
        y_test = to_categorical(y_test, num_classes)

    else:
               
        X_train = X_train.values
        X_valid = None
        X_test = X_test.values

        y_train = to_categorical(y_train, num_classes)
        y_valid = None
        y_test = to_categorical(y_test, num_classes) 

    return X_train, y_train, X_valid, y_valid, X_test, y_test, feature_names, input_shape

# ********************************************************************************************
def trainDNNMultiClass(X_train, y_train, X_valid, y_valid, X_test, y_test, num_classes, 
                       label_names, batch_size, epochs=20, dropout_rate=0.1, l1_reg=1e-6, model_name="Feedforward_NN"):
    input_shape = (X_train.shape[1], ) 
    model = build_multi(input_shape, num_classes, dropout_rate=dropout_rate, l1_reg=l1_reg)
    compile_multi(model, lr_rate=0.001)
    history, model = fit_multi(model, X_train, y_train, X_valid, y_valid, epochs=epochs, batch_size=batch_size)
    report_df, y_true, y_pred = evaluate_multi(model, X_test, y_test, label_names, model_name=model_name)
    return report_df, history, model, y_true, y_pred

# ********************************************************************************************

# CNN

def build_multi_cnn(input_shape, num_classes, dropout_rate=0.1, l1_reg=1e-6):
    model = Sequential([
        layers.Conv1D(64, kernel_size=3, activation="relu", input_shape=input_shape,
                      kernel_regularizer=regularizers.l1(l1_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Conv1D(32, kernel_size=3, activation="relu",
                      kernel_regularizer=regularizers.l1(l1_reg)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.GlobalAveragePooling1D(),
        layers.Dense(num_classes, activation="softmax") 
    ])
    return model


# ********************************************************************************************
  
def trainCNNMultiClass(X_train, y_train, X_valid, y_valid, X_test, y_test, num_classes, 
                       label_names, batch_size=128, epochs=20, dropout_rate=0.1, l1_reg=1e-6, model_name="CNN"):
    input_shape = (X_train.shape[1], 1)  
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_valid_cnn = np.expand_dims(X_valid, axis=-1)
    X_test_cnn = np.expand_dims(X_test, axis=-1)

    model = build_multi_cnn(input_shape, num_classes, dropout_rate=dropout_rate, l1_reg=l1_reg )
    compile_multi(model, lr_rate=0.001)
    history, model = fit_multi(model, X_train_cnn, y_train, X_valid_cnn, y_valid, epochs=epochs, batch_size=batch_size)
    report_df, y_true, y_pred = evaluate_multi(model, X_test_cnn, y_test, label_names, model_name=model_name)
    return report_df, history, model, y_true, y_pred

# ********************************************************************************************

def build_resnet_multi(input_shape, num_classes, dropout_rate=0.1, l1_reg=1e-6):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=7, strides=2, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l1(l1_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(x)
    x = layers.Dropout(dropout_rate)(x)

    def residual_block(x, filters, kernel_size=3, stride=1):
        shortcut = x
        x = layers.Conv1D(filters, kernel_size=kernel_size, strides=stride, padding="same", activation="relu",
                          kernel_regularizer=regularizers.l1(l1_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters, kernel_size=kernel_size, strides=1, padding="same",
                          kernel_regularizer=regularizers.l1(l1_reg))(x)
        x = layers.BatchNormalization()(x)
        
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, kernel_size=1, strides=stride, padding="same",
                                     kernel_regularizer=regularizers.l1(l1_reg))(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        return x

    for filters, strides in zip([64, 128, 256], [1, 2, 2]):
        x = residual_block(x, filters, stride=strides)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)  

    model = keras.Model(inputs, outputs)
    return model

# ********************************************************************************************
def build_resnet_multi_extra(input_shape, num_classes, dropout_rate=1e-9, l1_reg=1e-9):
    inputs = layers.Input(shape=input_shape)

    # Initial Conv Layer
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding="same", activation="relu",
                      kernel_regularizer=regularizers.l1(l1_reg))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding="same")(x)
    x = layers.Dropout(dropout_rate)(x)

    def residual_block(x, filters, kernel_size=3, stride=2):
        shortcut = x
        x = layers.Conv1D(filters, kernel_size=kernel_size, strides=stride, padding="same", activation="relu",
                          kernel_regularizer=regularizers.l1(l1_reg))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters, kernel_size=kernel_size, strides=1, padding="same",
                          kernel_regularizer=regularizers.l1(l1_reg))(x)
        x = layers.BatchNormalization()(x)
        
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, kernel_size=1, strides=stride, padding="same",
                                     kernel_regularizer=regularizers.l1(l1_reg))(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        x = layers.Dropout(dropout_rate)(x)
        return x

    for filters, strides in zip([64, 128, 256, 512], [1, 2, 2, 2]):
        x = residual_block(x, filters, stride=strides)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)  # Multi-class output

    model = keras.Model(inputs, outputs)
    return model

# ********************************************************************************************

def trainResNetMultiClass(X_train, y_train, X_valid, y_valid, X_test, y_test, num_classes, 
                          label_names, batch_size, epochs=20, dropout_rate=0.1, l1_reg=1e-6, model_name="Resnet"):

    input_shape = (X_train.shape[1], 1) 
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_valid_cnn = np.expand_dims(X_valid, axis=-1)
    X_test_cnn = np.expand_dims(X_test, axis=-1)

    model = build_resnet_multi(input_shape, num_classes, dropout_rate=dropout_rate, l1_reg=l1_reg)

    compile_multi(model, lr_rate=0.001)
    history, model = fit_multi(model, X_train_cnn, y_train, X_valid_cnn, y_valid, epochs=epochs, batch_size=batch_size)
    report_df, y_true, y_pred = evaluate_multi(model, X_test_cnn, y_test, label_names, model_name=model_name)
    return report_df, history, model, y_true, y_pred

# ********************************************************************************************
def trainResNetMultiClassExtra(X_train, y_train, X_valid, y_valid, X_test, y_test, num_classes, 
                          label_names, batch_size, epochs=20, dropout_rate=1e-9, l1_reg=1e-9, model_name="Resnet"):

    input_shape = (X_train.shape[1], 1)  
    X_train_cnn = np.expand_dims(X_train, axis=-1)
    X_valid_cnn = np.expand_dims(X_valid, axis=-1)
    X_test_cnn = np.expand_dims(X_test, axis=-1)

    model = build_resnet_multi(input_shape, num_classes, dropout_rate=dropout_rate, l1_reg=l1_reg)
    compile_multi(model, lr_rate=0.001)
    history, model = fit_multi(model, X_train_cnn, y_train, X_valid_cnn, y_valid, epochs=epochs, batch_size=batch_size)
    report_df, y_true, y_pred = evaluate_multi(model, X_test_cnn, y_test, label_names, model_name=model_name)
    return report_df, history, model, y_true, y_pred
# ********************************************************************************************

def plotModel(history, trunc=0):
    # Getting the loss and metrics 
    loss = history.history['loss'][trunc:]
    val_loss = history.history['val_loss'][trunc:]
    acc = history.history['accuracy'][trunc:]
    val_acc = history.history['val_accuracy'][trunc:]
    
    # Getting the epochs
    epochs = range(1, len(loss) + 1)  #
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