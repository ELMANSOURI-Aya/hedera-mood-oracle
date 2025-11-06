# ==============================
# 3_scientific_ml.py — VERSION ULTRA-OPTIMISÉE (LSTM + RF ONLY)
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# === CHEMINS ===
BASE_DIR = Path("/content/drive/MyDrive/Hedera Mood Oracle")
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
IMAGES_DIR = BASE_DIR / "images"
for d in [DATA_DIR, MODELS_DIR, IMAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =============================================
# 1. EDA
# =============================================
print("ÉTAPE 1 : EDA")
df = pd.read_csv(DATA_DIR / "fear_greed_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print("Shape :", df.shape)
print("Période :", df['timestamp'].min().date(), "→", df['timestamp'].max().date())
print("\nDescription :")
print(df['value'].describe())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(df['value'], bins=50, kde=True, ax=axes[0,0])
axes[0,0].set_title("Distribution Fear & Greed")
df['value'].plot(ax=axes[0,1])
axes[0,1].set_title("Évolution")
df['value'].rolling(30).mean().plot(color='red', ax=axes[1,0])
axes[1,0].set_title("Moyenne mobile 30j")
df['classification'].value_counts().plot(kind='bar', ax=axes[1,1])
axes[1,1].set_title("Classes")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(IMAGES_DIR / "eda_summary.png", dpi=300)
plt.show()

# =============================================
# 2. DATA PREP
# =============================================
print("\nÉTAPE 2 : Préparation")
def classify_5(score):
    if score <= 25: return "Extreme Fear"
    elif score <= 45: return "Fear"
    elif score <= 55: return "Neutral"
    elif score <= 75: return "Greed"
    else: return "Extreme Greed"

df['class_5'] = df['value'].apply(classify_5)
df['day_of_year'] = df['timestamp'].dt.dayofyear
df['month'] = df['timestamp'].dt.month
df['weekday'] = df['timestamp'].dt.weekday
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

for w in [7, 14, 30]:
    df[f"rolling_mean_{w}"] = df['value'].rolling(w).mean()
    df[f"rolling_std_{w}"] = df['value'].rolling(w).std()

for lag in [1, 3, 7, 14, 30]:
    df[f"lag_{lag}"] = df['value'].shift(lag)

df = df.dropna().reset_index(drop=True)
print("Après préparation :", df.shape)

# =============================================
# 3. RÉGRESSION : LSTM SEULEMENT (LE MEILLEUR)
# =============================================
print("\nÉTAPE 3 : Régression (LSTM uniquement)")
sequence_length = 60
scaler = StandardScaler()
values_scaled = scaler.fit_transform(df['value'].values.reshape(-1, 1))

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(values_scaled, sequence_length)
train_size = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:train_size], X_seq[train_size:]
y_train, y_test = y_seq[:train_size], y_seq[train_size:]

model_lstm = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, input_shape=(sequence_length, 1))),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

pred_lstm = scaler.inverse_transform(model_lstm.predict(X_test)).flatten()
mae_lstm = mean_absolute_error(scaler.inverse_transform(y_test), pred_lstm)
print("LSTM MAE:", round(mae_lstm, 2))

# =============================================
# 4. CLASSIFICATION : RF SEULEMENT (LE MEILLEUR)
# =============================================
print("\nÉTAPE 4 : Classification (RF uniquement)")
features_cls = [c for c in df.columns if c not in ['timestamp', 'value', 'classification', 'class_5']]
X_cls = df[features_cls]
y_cls = df['class_5']
le = LabelEncoder()
y_encoded = le.fit_transform(y_cls)

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_cls, y_encoded, test_size=0.2, shuffle=False)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train_rf), y=y_train_rf)
class_weight_dict = dict(zip(np.unique(y_train_rf), class_weights))

rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42)
rf.fit(X_train_rf, y_train_rf)
pred_rf = rf.predict(X_test_rf)
acc_rf = (pred_rf == y_test_rf).mean()
print("RF Accuracy:", round(acc_rf, 3))

importances = rf.feature_importances_
top_features = pd.Series(importances, index=features_cls).sort_values(ascending=False).head(5)
print("Top 5 features:", top_features.to_dict())

# === AJOUT À LA FIN DE TON SCRIPT (APRÈS LES MODÈLES) ===

# Fonction classify_5 (déjà dans ton code)
def classify_5(score):
    if score <= 25: return "Extreme Fear"
    elif score <= 45: return "Fear"
    elif score <= 55: return "Neutral"
    elif score <= 75: return "Greed"
    else: return "Extreme Greed"

# PRÉDICTION DEMAIN + HYBRID ENSEMBLE
last_seq = values_scaled[-sequence_length:].reshape(1, sequence_length, 1)
pred_score = float(scaler.inverse_transform(model_lstm.predict(last_seq))[0][0])
pred_class_idx = rf.predict(X_cls.iloc[-1:].values)[0]
pred_class = le.inverse_transform([pred_class_idx])[0]

score_to_class = classify_5(pred_score)

if score_to_class == pred_class:
    final_score = pred_score
    final_class = pred_class
    confidence = "HAUTE"
else:
    class_midpoints = {"Extreme Fear": 12.5, "Fear": 35, "Neutral": 50, "Greed": 65, "Extreme Greed": 87.5}
    rf_implied_score = class_midpoints[pred_class]
    final_score = 0.7 * pred_score + 0.3 * rf_implied_score
    final_class = classify_5(final_score)
    confidence = "MOYENNE"

# Sauvegarde
result = {
    "predicted_score": round(final_score, 1),
    "predicted_class": final_class,
    "confidence": confidence,
    "lstm_score": round(pred_score, 1),
    "rf_class": pred_class,
    "mae_lstm": round(mae_lstm, 2),
    "accuracy_rf": round(acc_rf, 3)
}
with open(DATA_DIR / "latest_prediction.json", 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nPRÉDICTION HYBRID : {final_score:.1f} → {final_class} [{confidence}]")

# =============================================
# 5. SAUVEGARDE & PRÉDICTION DEMAIN
# =============================================
joblib.dump({
    'lstm_regression': model_lstm,
    'rf_classification': rf,
    'scaler': scaler,
    'label_encoder': le,
    'features': features_cls,
    'mae_lstm': mae_lstm,
    'acc_rf': acc_rf
}, MODELS_DIR / "final_models.pkl")

# PRÉDICTION DEMAIN
last_seq = values_scaled[-sequence_length:].reshape(1, sequence_length, 1)
pred_score = float(scaler.inverse_transform(model_lstm.predict(last_seq))[0][0])
pred_class_idx = rf.predict(X_cls.iloc[-1:].values)[0]
pred_class = le.inverse_transform([pred_class_idx])[0]

result = {
    "predicted_score": round(pred_score, 1),
    "predicted_class": pred_class,
    "mae_lstm": round(mae_lstm, 2),
    "accuracy_rf": round(acc_rf, 3)
}
with open(DATA_DIR / "latest_prediction.json", 'w') as f:
    json.dump(result, f, indent=2)

# VISUALISATION
plt.figure(figsize=(14,6))
plt.plot(df['timestamp'].iloc[-len(y_test):], scaler.inverse_transform(y_test), label='Vrai', linewidth=2)
plt.plot(df['timestamp'].iloc[-len(y_test):], pred_lstm, label=f'LSTM (MAE={mae_lstm:.2f})', linewidth=2)
plt.axvline(df['timestamp'].iloc[train_size], color='gray', linestyle=':', label='Train/Test')
plt.legend()
plt.title("Hedera Mood Oracle — Prédiction LSTM (Régression)")
plt.xlabel("Date")
plt.ylabel("Fear & Greed Index")
plt.tight_layout()
plt.savefig(IMAGES_DIR / "lstm_regression_final.png", dpi=300)
plt.show()

print(f"\nPRÉDICTION DEMAIN : {pred_score:.1f} → {pred_class}")
print(f"LSTM MAE: {mae_lstm:.2f} | RF ACC: {acc_rf:.3f}")
print("final_models.pkl + latest_prediction.json sauvegardés")
