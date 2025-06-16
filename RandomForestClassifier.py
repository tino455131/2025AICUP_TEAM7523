import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score

TRAIN_INFO_PATH = "train_info.csv"
TRAIN_DATA_DIR = "./train_data"
TEST_INFO_PATH = "test_info.csv"
TEST_DATA_DIR = "./test_data"
OUTPUT_SUBMISSION = "RandomForestClassifier.csv"

# === 2. Feature Extraction ===
def extract_features(txt_path):
    data = np.loadtxt(txt_path)
    stats = [data.mean(axis=0), data.std(axis=0), data.min(axis=0), data.max(axis=0)]
    return np.concatenate(stats)

# === 3. Load Training Data ===
train_info = pd.read_csv(TRAIN_INFO_PATH)
train_info['gender'] = train_info['gender'].map({1: 1, 2: 0})
train_info['hold racket handed'] = train_info['hold racket handed'].map({1: 1, 2: 0})
train_info['play years'] = train_info['play years'] - 1
train_info['level'] = train_info['level'] - 2

X, y_gender, y_handed, y_years, y_level = [], [], [], [], []
for _, row in train_info.iterrows():
    path = os.path.join(TRAIN_DATA_DIR, f"{row['unique_id']}.txt")
    if os.path.exists(path):
        feat = extract_features(path)
        X.append(feat)
        y_gender.append(row['gender'])
        y_handed.append(row['hold racket handed'])
        y_years.append(row['play years'])
        y_level.append(row['level'])

X = np.array(X)

# === 4. Train Models ===
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model_gender = train_model(X, y_gender)
model_handed = train_model(X, y_handed)
model_years = train_model(X, y_years)
model_level = train_model(X, y_level)

# === 5. Load Test Data ===
test_info = pd.read_csv(TEST_INFO_PATH)
test_ids, test_feats = [], []
for _, row in test_info.iterrows():
    path = os.path.join(TEST_DATA_DIR, f"{row['unique_id']}.txt")
    if os.path.exists(path):
        feat = extract_features(path)
        test_ids.append(row['unique_id'])
        test_feats.append(feat) # 每位選手的

X_test = np.array(test_feats)

# === 6. Predict Probabilities ===
pred_gender = model_gender.predict_proba(X_test)[:, 1]  # male prob
pred_handed = model_handed.predict_proba(X_test)[:, 1]  # right-handed prob
pred_years = model_years.predict_proba(X_test)          # columns: low/mid/high
pred_level = model_level.predict_proba(X_test)          # columns: level 2/3/4/5

# === 7. Format Submission File ===
submission = pd.DataFrame({
    'unique_id': test_ids,
    'gender': pred_gender,
    'hold racket handed': pred_handed
})
submission[['play years_0', 'play years_1', 'play years_2']] = pred_years
submission[['level_2', 'level_3', 'level_4', 'level_5']] = pred_level
submission.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"Saved prediction to {OUTPUT_SUBMISSION}")
