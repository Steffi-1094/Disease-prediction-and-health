# ==========================================
# IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

# ==========================================
# LOAD DATASET
# ==========================================
data = pd.read_csv("dataset.csv")

# ==========================================
# SYMPTOM COLUMNS
# ==========================================
symptom_columns = [f"Symptom_{i}" for i in range(1,18)]

# ==========================================
# CLEAN DATA
# ==========================================

# Remove extra spaces
for col in symptom_columns:
    data[col] = data[col].astype(str).str.strip()

# Fix wrong spacing in symptoms
for col in symptom_columns:
    data[col] = data[col].str.replace(" ", "_")

# Replace NaN with empty
data[symptom_columns] = data[symptom_columns].replace("nan","")
data[symptom_columns] = data[symptom_columns].fillna("")

# Remove duplicates
data = data.drop_duplicates()

# ==========================================
# ENCODE DISEASE LABELS
# ==========================================
le_disease = LabelEncoder()
data["Disease"] = le_disease.fit_transform(data["Disease"])

# ==========================================
# CREATE BINARY SYMPTOM FEATURES
# ==========================================
binary_features = pd.get_dummies(
    data[symptom_columns].stack()
).groupby(level=0).sum()

X = binary_features
y = data["Disease"]

print("Total Samples:", X.shape[0])
print("Total Symptoms:", X.shape[1])

# ==========================================
# TRAIN TEST SPLIT
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==========================================
# HANDLE CLASS IMBALANCE
# ==========================================
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", X_train.shape)

# ==========================================
# FEATURE SELECTION
# ==========================================
selector = SelectKBest(mutual_info_classif, k="all")

X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# ==========================================
# MODEL (RANDOM FOREST)
# ==========================================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_selected, y_train)

# ==========================================
# MODEL EVALUATION
# ==========================================
y_pred = model.predict(X_test_selected)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred,
                            target_names=le_disease.classes_))

# ==========================================
# SAVE MODEL FILES
# ==========================================
joblib.dump(model, "xgb_model.pkl")
joblib.dump(selector, "selector.pkl")
joblib.dump(le_disease, "label_encoder.pkl")
joblib.dump(X.columns.tolist(), "binary_features_columns.pkl")

print("\nModel saved successfully!")