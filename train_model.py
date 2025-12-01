# train_model.py
import os
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load data
# Make sure you've run generate_minimax_dataset.py first to create this file:
# data/tic_tac_toe_minimax_states.csv
df = pd.read_csv("data/tic_tac_toe_minimax_states.csv")

# Columns in this file are already:
# top_left, top_middle, top_right,
# middle_left, middle_middle, middle_right,
# bottom_left, bottom_middle, bottom_right,
# value  (-1, 0, 1)

X = df.drop("value", axis=1)
y = df["value"].astype(int)   # -1, 0, 1  (X loses, draw, X wins)

# 2. Preprocessor + model
categorical_cols = X.columns.tolist()  # all 9 positions are categorical

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

rf_model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ))
])

# 3. Train on 100% of the data (all reachable minimax states)
rf_model.fit(X, y)

# 4. (Optional) Check training accuracy on all states
y_pred = rf_model.predict(X)
train_acc = accuracy_score(y, y_pred)
print(f"Training accuracy on all minimax states: {train_acc:.4f}")

# 5. Save model
os.makedirs("model", exist_ok=True)
joblib.dump(rf_model, "model/tic_tac_toe_rf.pkl")
print("Model saved to model/tic_tac_toe_rf.pkl")
