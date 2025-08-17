from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1
import matplotlib.pyplot as plt

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
    # x = Dense(64)(inputs) # Prior
    x = Dense(32, kernel_regularizer=l1(l1_lambda))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Add multiple residual blocks
    for _ in range(3):  # You can adjust the number of blocks
        # x = residual_block(x, 64) # Prior
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
    epochs = range(1, len(loss) + 1)  # Or len(acc), len(val_loss). They all have the same length

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

    plt.tight_layout()  # Adjusts layout to prevent overlap
    plt.show()



# Define the callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',     # Monitor the validation loss
    patience=20,            # Stop training after 20 epochs with no improvement
    restore_best_weights=True  # Restore the best weights after stopping
)

model_checkpoint = ModelCheckpoint(
    'best_model.keras',        # Save the model to this file
    monitor='val_loss',     # Monitor validation loss to save the best model
    save_best_only=True     # Save only the best model (with lowest validation loss)
)

# *****************************************************************
# Fully Connected Layer Model
# *****************************************************************
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall, AUC

def build_and_train_modelFCL(X_train, y_train, X_valid, y_valid,
                          X_test, y_test, input_shape, epochs=20, batch_size=32, lr_rate=0.001):
    """
    Build and train a feedforward neural network with specified metrics.

    Parameters:
    - X_train, y_train: Training data and labels.
    - X_valid, y_valid: Validation data and labels.
    - X_test, y_test: Test data and labels.
    - input_shape: Shape of the input data.
    - epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    - lr_rate: Learning rate for the optimizer.

    Returns:
    - model: The trained Keras model.
    - history: Training history object.
    - metrics_dict: Dictionary of final evaluation metrics (accuracy, recall, precision, F1-score, AUC, training and validation accuracy).
    """
    # Build the model
    model = keras.Sequential([
        layers.Dense(256, activation="relu", input_shape=input_shape),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    
    # Compile the model with additional metrics
    optimizer = keras.optimizers.Adam(learning_rate=lr_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate the model on test data
    eval_metrics = model.evaluate(X_test, y_test, verbose=0)
    metrics_dict = {
        'test_accuracy': round(eval_metrics[1], 4),
        'test_precision': round(eval_metrics[2], 4),
        'test_recall': round(eval_metrics[3], 4),
        'test_f1_score': round(2 * (eval_metrics[2] * eval_metrics[3]) / (eval_metrics[2] + eval_metrics[3]), 4) if eval_metrics[2] + eval_metrics[3] > 0 else 0.0,
        'test_auc': round(eval_metrics[4], 4),
        'train_accuracy': round(history.history['accuracy'][-1], 4),  # Last epoch's training accuracy
        'val_accuracy': round(history.history['val_accuracy'][-1], 4)  # Last epoch's validation accuracy
    }
    
    return model, history, metrics_dict

# *****************************************************************
# CNN model
# *****************************************************************
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import Precision, Recall, AUC

# def build_and_train_modelCNN(X_train, y_train, X_valid, y_valid,
#                           X_test, y_test, input_shape, epochs=20, batch_size=32, lr_rate=0.001):
def build_and_train_modelCNN(X_train, y_train, X_valid, y_valid,
                          X_test, y_test, epochs=20, batch_size=32, lr_rate=0.001):
    """
    Build and train a 1D Convolutional Neural Network with specified metrics.

    Parameters:
    - X_train, y_train: Training data and labels.
    - X_valid, y_valid: Validation data and labels.
    - X_test, y_test: Test data and labels.
    - input_shape: Shape of the input data.
    - epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    - lr_rate: Learning rate for the optimizer.

    Returns:
    - model: The trained Keras model.
    - history: Training history object.
    - metrics_dict: Dictionary of final evaluation metrics (accuracy, recall, precision, F1-score, AUC, training and validation accuracy).
    """
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape=(X_train.shape[1], 1)

    # Build the model
    model = keras.Sequential([
        layers.Conv1D(64, kernel_size=3, activation="relu", input_shape=input_shape),
        layers.Conv1D(32, kernel_size=3, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(1, activation="sigmoid")
    ])
    
    # Compile the model with additional metrics
    optimizer = keras.optimizers.Adam(learning_rate=lr_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate the model on test data
    eval_metrics = model.evaluate(X_test, y_test, verbose=0)
    metrics_dict = {
        'test_accuracy': round(eval_metrics[1], 4),
        'test_precision': round(eval_metrics[2], 4),
        'test_recall': round(eval_metrics[3], 4),
        'test_f1_score': round(2 * (eval_metrics[2] * eval_metrics[3]) / (eval_metrics[2] + eval_metrics[3]), 4) if eval_metrics[2] + eval_metrics[3] > 0 else 0.0,
        'test_auc': round(eval_metrics[4], 4),
        'train_accuracy': round(history.history['accuracy'][-1], 4),  # Last epoch's training accuracy
        'val_accuracy': round(history.history['val_accuracy'][-1], 4)  # Last epoch's validation accuracy
    }
    
    return model, history, metrics_dict

# *****************************************************************
# Model: Deep CNN Model
# *****************************************************************
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dense

def build_and_train_modelDCNN(X_train, y_train, X_valid, y_valid,
                          X_test, y_test, epochs=20, batch_size=32, lr_rate=0.001):
    """
    Build and train a 1D Convolutional Neural Network with the specified architecture and metrics.
    Input data is reshaped internally for Conv1D compatibility.

    Parameters:
    - X_train, y_train: Training data and labels.
    - X_valid, y_valid: Validation data and labels.
    - X_test, y_test: Test data and labels.
    - epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    - lr_rate: Learning rate for the optimizer.

    Returns:
    - model: The trained Keras model.
    - history: Training history object.
    - metrics_dict: Dictionary of final evaluation metrics (accuracy, recall, precision, F1-score, AUC, training and validation accuracy).

    Example:
    model, history, metrics = build_and_train_model(
    X_train, y_train, X_valid, y_valid, 
    X_test, y_test,
    epochs=20, batch_size=32, lr_rate=0.001
)
    """
    # Reshape input data to include a channel dimension
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, channels)
    
    # Build the model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=3, strides=2, padding='same'))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model with additional metrics
    optimizer = keras.optimizers.Adam(learning_rate=lr_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate the model on test data
    eval_metrics = model.evaluate(X_test, y_test, verbose=0)
    metrics_dict = {
        'test_accuracy': round(eval_metrics[1], 4),
        'test_precision': round(eval_metrics[2], 4),
        'test_recall': round(eval_metrics[3], 4),
        'test_f1_score': round(2 * (eval_metrics[2] * eval_metrics[3]) / (eval_metrics[2] + eval_metrics[3]), 4) if eval_metrics[2] + eval_metrics[3] > 0 else 0.0,
        'test_auc': round(eval_metrics[4], 4),
        'train_accuracy': round(history.history['accuracy'][-1], 4),  # Last epoch's training accuracy
        'val_accuracy': round(history.history['val_accuracy'][-1], 4)  # Last epoch's validation accuracy
    }
    
    return model, history, metrics_dict

# *****************************************************************
# Model: Resnet-1D
# *****************************************************************
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Add, GlobalAveragePooling1D, Dense, Input
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam

def build_and_train_resnet_1d(X_train, y_train, X_valid, y_valid,
                              X_test, y_test, epochs=20, batch_size=64, lr_rate=0.001):
    """
    Build and train a ResNet-50 inspired 1D CNN architecture.

    Parameters:
    - X_train, y_train: Training data and labels.
    - X_valid, y_valid: Validation data and labels.
    - X_test, y_test: Test data and labels.
    - epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    - lr_rate: Learning rate for the optimizer.

    Returns:
    - model: The trained Keras model.
    - history: Training history object.
    - metrics_dict: Dictionary of final evaluation metrics (accuracy, recall, precision, F1-score, AUC, training and validation accuracy).
    """
    # Reshape input data to include a channel dimension
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, channels)
    
    # Residual Block
    def residual_block(x, filters, kernel_size=6, strides=1, downsample=False):
        shortcut = x
        
        # Convolutional Layer 1
        x = Conv1D(filters, kernel_size, padding='same', strides=strides, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Convolutional Layer 2
        x = Conv1D(filters, kernel_size, padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        
        # Downsample the shortcut if required
        if downsample or strides > 1:
            shortcut = Conv1D(filters, kernel_size=1, padding='same', strides=strides)(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        # Add the shortcut
        x = Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    # Build the ResNet-50 Inspired Model
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=6, strides=2, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    # Residual Blocks (using ResNet-50's structure)
    for filters, blocks, downsample in [(64, 3, False), (128, 4, True), (256, 6, True), (512, 3, True)]:
        for block in range(blocks):
            x = residual_block(x, filters, strides=2 if downsample and block == 0 else 1)

    # Global Average Pooling and Dense Layers
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    
    # Compile the model with additional metrics
    optimizer = Adam(learning_rate=lr_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate the model on test data
    eval_metrics = model.evaluate(X_test, y_test, verbose=0)
    metrics_dict = {
        'test_accuracy': round(eval_metrics[1], 4),
        'test_precision': round(eval_metrics[2], 4),
        'test_recall': round(eval_metrics[3], 4),
        'test_f1_score': round(2 * (eval_metrics[2] * eval_metrics[3]) / (eval_metrics[2] + eval_metrics[3]), 4) if eval_metrics[2] + eval_metrics[3] > 0 else 0.0,
        'test_auc': round(eval_metrics[4], 4),
        'train_accuracy': round(history.history['accuracy'][-1], 4),  # Last epoch's training accuracy
        'val_accuracy': round(history.history['val_accuracy'][-1], 4)  # Last epoch's validation accuracy
    }
        
    return model, history, metrics_dict


# Model: ResNet-50 - half
def build_and_train_resnet_1d_half(X_train, y_train, X_valid, y_valid,
                                   X_test, y_test, epochs=20, batch_size=64, lr_rate=0.001):
    """
    Build and train a smaller ResNet-inspired 1D CNN architecture with ~50 layers.

    Parameters:
    - X_train, y_train: Training data and labels.
    - X_valid, y_valid: Validation data and labels.
    - X_test, y_test: Test data and labels.
    - epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    - lr_rate: Learning rate for the optimizer.

    Returns:
    - model: The trained Keras model.
    - history: Training history object.
    - metrics_dict: Dictionary of final evaluation metrics (accuracy, recall, precision, F1-score, AUC, training and validation accuracy).
    """
    # Reshape input data to include a channel dimension
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_valid = X_valid.reshape((X_valid.shape[0], X_valid.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, channels)
    
    # Residual Block
    def residual_block(x, filters, kernel_size=6, strides=1, downsample=False):
        shortcut = x
        
        # Convolutional Layer 1
        x = Conv1D(filters, kernel_size, padding='same', strides=strides, activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Convolutional Layer 2
        x = Conv1D(filters, kernel_size, padding='same', activation=None)(x)
        x = BatchNormalization()(x)
        
        # Downsample the shortcut if required
        if downsample or strides > 1:
            shortcut = Conv1D(filters, kernel_size=1, padding='same', strides=strides)(shortcut)
            shortcut = BatchNormalization()(shortcut)
        
        # Add the shortcut
        x = Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    # Build the smaller ResNet model
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=6, strides=2, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)
    
    # Adjusted Residual Blocks for ~50 layers
    for filters, blocks, downsample in [(64, 2, False), (128, 2, True), (256, 3, True), (512, 2, True)]:
        for block in range(blocks):
            x = residual_block(x, filters, strides=2 if downsample and block == 0 else 1)

    # Global Average Pooling and Dense Layers
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    
    # Compile the model with additional metrics
    optimizer = Adam(learning_rate=lr_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(), Recall(), AUC()]
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    # Evaluate the model on test data
    eval_metrics = model.evaluate(X_test, y_test, verbose=0)

    metrics_dict = {
        'test_accuracy': round(eval_metrics[1], 4),
        'test_precision': round(eval_metrics[2], 4),
        'test_recall': round(eval_metrics[3], 4),
        'test_f1_score': round(2 * (eval_metrics[2] * eval_metrics[3]) / (eval_metrics[2] + eval_metrics[3]), 4) if eval_metrics[2] + eval_metrics[3] > 0 else 0.0,
        'test_auc': round(eval_metrics[4], 4),
        'train_accuracy': round(history.history['accuracy'][-1], 4),  # Last epoch's training accuracy
        'val_accuracy': round(history.history['val_accuracy'][-1], 4)  # Last epoch's validation accuracy
    }
    
    return model, history, metrics_dict
