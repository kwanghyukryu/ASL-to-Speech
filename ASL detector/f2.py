import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import joblib
import os

# Load your collected data
df = pd.read_csv("real_asl_data/real_asl_dataset.csv")

# Only use samples with an actual label
#df = df[df['letter']].isin(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))]

X = df[[col for col in df.columns if col.startswith("landmark_")]].values
y = df['letter'].values  # <-- Use actual ASL letter labels

X = np.nan_to_num(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=20)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model pipeline
os.makedirs("trained_models", exist_ok=True)
joblib.dump(clf, "trained_models/real_asl_model.joblib")
joblib.dump(scaler, "trained_models/feature_scaler.joblib")
joblib.dump(pca, "trained_models/pca_transformer.joblib")