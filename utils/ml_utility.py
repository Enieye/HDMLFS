import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils import resample

def ml_binClReport(model, X_train, y_train, X_test, y_test):

    # Fit the model
    t1 = time.time()
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    t2 = time.time()
    t_complete = t2 - t1
    print(f"{model.__class__.__name__} binary-classification takes {t_complete:.2f}s to complete")


    # Metrics for train set
    train_acc = accuracy_score(y_train, y_train_pred)

    # Metrics for test set
    test_acc = accuracy_score(y_test, y_test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='binary')

    metrics_df = pd.DataFrame([{
        'Model': model.__class__.__name__,
        'Train Accuracy': round(train_acc, 4),
        'Test Accuracy': round(test_acc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4)
    }])
    return metrics_df, model, y_test_pred


def ml_binConfxMtrx(y_test, y_test_pred):
    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    # f,ax=plt.subplots(figsize=(10,8))
    sns.heatmap(cm,annot=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def ml_multiClReport(model, X_train, y_train, X_test, y_test, label_names):
    t1 = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    t2 = time.time()
    t_complete = t2 - t1

    print(f"{model.__class__.__name__} multi-classification takes {t_complete:.2f}s to complete")
    print(classification_report(y_test, y_pred))
    report_dict = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose().round(4)

    return report_df, model, y_pred

def ml_multiConfxMtrx(y_test, y_pred, label_names):
    cm=confusion_matrix(y_test, y_pred)
    f,ax=plt.subplots(figsize=(8,6))
    sns.heatmap(cm, annot=True, linewidth=0.5, linecolor="red", fmt=".0f", ax=ax, cmap=sns.color_palette("coolwarm", as_cmap=True),
            xticklabels=label_names, yticklabels=label_names    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def generateDataSet(df_train, df_test, removedList, label, validSet=False):

    X_train = df_train.drop(removedList, axis=1).copy()
    y_train = df_train[label].copy()
    X_test = df_test.drop(removedList, axis=1).copy()
    y_test = df_test[label].copy()
    X_valid = None
    y_valid = None


    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    if validSet == True:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


# Decision Trees
dt = DecisionTreeClassifier(max_depth=5, ccp_alpha=0.01)

# Logistic Regression
lr = LogisticRegression(penalty='l1', C=3e-3, solver='liblinear')

# Support Vector Classifier
svc = SVC(kernel='rbf', C=1.0)

# Naive Bayes
nb = GaussianNB()  

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# # XGBoost Classifier
xgb = XGBClassifier(n_estimators=100, max_depth=5, random_state=42, learning_rate=0.02, reg_lambda=1e5)

# Resampling
def resample_mean(train_data, label):
    mean_class_size = train_data[label[0]].value_counts().mean()

    balanced_data = []
    for category in train_data[label[0]].unique():
        category_data = train_data[train_data[label[0]] == category]
        if len(category_data) < mean_class_size:

            category_data = resample(category_data,
                                     replace=True,
                                     n_samples=int(mean_class_size),
                                     random_state=42)
        elif len(category_data) > mean_class_size:

            category_data = resample(category_data,
                                     replace=False,
                                     n_samples=int(mean_class_size),
                                     random_state=42)
        balanced_data.append(category_data)

    train_data_balanced = pd.concat(balanced_data)
    return train_data_balanced