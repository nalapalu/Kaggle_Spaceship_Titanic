import sys

import pandas as pd
from sklearn.calibration import LinearSVC
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

import joblib
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    log_loss,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)
from sklearn.tree import DecisionTreeClassifier



sys.path.append("..")

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

train = pd.read_csv("../../data/processed/train.csv")
train = train.drop(train.columns[0], axis=1)

target = "Survived"

# --------------------------------------------------------------
# Train test split
# --------------------------------------------------------------

X = train.drop([target], axis=1)
y = train[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# --------------------------------------------------------------
# Train model
# --------------------------------------------------------------

# Define preprocessing for numeric columns (scale them)
numeric_features = X.select_dtypes(include=["float64", "int64"]).columns
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

# Define preprocessing for categorical features (encode them)
categorical_features = X.select_dtypes(include=["category"]).columns
categorical_transformer = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# # --------------------------------------------------------------
# # Testing different models
# # --------------------------------------------------------------

classifiers = [
    RandomForestClassifier(),
    LogisticRegression(),
    SVC(),
    KNeighborsClassifier(),
    GaussianNB(),
    Perceptron(),
    LinearSVC(),
    SGDClassifier(),
    DecisionTreeClassifier(),
    CatBoostClassifier()
]

accuracy_scores = []
cv_scores = [] 
for classifier in classifiers:
    # Build the pipeline for the current classifier
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    # Fit the pipeline to train the model on the training set
    model = pipeline.fit(X_train, y_train)

    # Evaluate the model
    # Get predictions
    predictions = model.predict(X_test)

    # Display metrics
    accuracy = accuracy_score(y_test, predictions)
    accuracy_scores.append((classifier.__class__.__name__, accuracy))
    
    # Compute cross-validation scores
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    cv_scores.append((classifier.__class__.__name__, scores.mean()))

train_accuracy = pd.DataFrame(accuracy_scores, columns=['Classifier', 'Accuracy'])
train_CVscore = pd.DataFrame(cv_scores, columns=['Classifier', 'CV_score'])

train_accuracy.sort_values(by = 'Accuracy', ascending = False, ignore_index = True)
train_CVscore.sort_values(by = 'CV_score', ascending = False, ignore_index = True)

# --------------------------------------------------------------
# Testing best models
# --------------------------------------------------------------

# Build the pipeline
pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classfier", SVC())]
)

# fit the pipeline to train the model on the training set
model = pipeline.fit(X_train, y_train)

# --------------------------------------------------------------
# Evaluate the model
# --------------------------------------------------------------

# Get predictions
predictions = model.predict(X_test)

# Display metrics
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# # Visualize results
# plot_predicted_vs_true(y_test, predictions)
# regression_scatter(y_test, predictions)
# plot_residuals(y_test, predictions, bins=15)


# --------------------------------------------------------------
# Hyper parameter tuning
# --------------------------------------------------------------


# --------------------------------------------------------------
# Export model
# --------------------------------------------------------------

ref_cols = list(X.columns)

"""
In Python, you can use joblib or pickle to serialize (and deserialize) an object structure into (and from) a byte stream. 
In other words, it's the process of converting a Python object into a byte stream that can be stored in a file.

https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html

"""

joblib.dump(value=[model, ref_cols, target], filename="../../models/model.pkl")
