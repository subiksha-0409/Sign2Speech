# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# === CONFIG ===
CSV_FILE = "sign_data.csv"  # Change to "sign_data_balanced.csv" only if you really want that
MODEL_FILE = "sign_model.pkl"

# === STEP 1: Show path & verify file ===
csv_path = os.path.abspath(CSV_FILE)
print(f"📂 Using CSV file: {csv_path}")

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"❌ CSV file not found: {CSV_FILE}")

# === STEP 2: Load data ===
df = pd.read_csv(CSV_FILE)

# Drop rows with missing labels
df = df.dropna(subset=["label"])

if df.empty:
    raise ValueError("❌ No data found in CSV after dropping NaN labels!")

# === STEP 3: Show data summary ===
print("\n📊 Dataset Summary:")
print(df["label"].value_counts())

# === STEP 4: Features and labels ===
X = df.drop("label", axis=1)
y = df["label"]

# === STEP 5: Encode labels ===
le = LabelEncoder()
y_enc = le.fit_transform(y)

# === STEP 6: Train model ===
print("\n🛠 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y_enc)

# === STEP 7: Remove old model ===
if os.path.exists(MODEL_FILE):
    os.remove(MODEL_FILE)
    print(f"🗑 Deleted old model file: {MODEL_FILE}")

# === STEP 8: Save new model & encoder ===
with open(MODEL_FILE, "wb") as f:
    pickle.dump({"model": model, "le": le}, f)

print("\n✅ Training complete! Model saved as", MODEL_FILE)
print("📌 Classes in new model:", list(le.classes_))
