import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
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

# Graphiques
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
plt.savefig(IMAGES_DIR / "eda_summary.png")
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
    df["rolling_mean_" + str(w)] = df['value'].rolling(w).mean()
    df["rolling_std_" + str(w)] = df['value'].rolling(w).std()

for lag in [1, 3, 7, 14, 30]:
    df["lag_" + str(lag)] = df['value'].shift(lag)

df = df.dropna().reset_index(drop=True)
print("Après préparation :", df.shape)

# =============================================
# 3. RÉGRESSION : PROPHET vs LSTM
# =============================================
print("\nÉTAPE 3 : Régression")
sequence_length = 60  # AMÉLIORÉ
scaler = StandardScaler()
values_scaled = scaler.fit_transform(df['value'].values.reshape(-1, 1))

# Prophet
prophet_df = df[['timestamp', 'value']].copy()
prophet_df.columns = ['ds', 'y']
train_size = int(len(prophet_df) * 0.8)
train_p, test_p = prophet_df.iloc[:train_size], prophet_df.iloc[train_size:]

model_prophet = Prophet(daily_seasonality=True, yearly_seasonality=True)
model_prophet.fit(train_p)
future = model_prophet.make_future_dataframe(periods=len(test_p), freq='D')
forecast = model_prophet.predict(future)
pred_prophet = forecast.iloc[train_size:]['yhat'].values
mae_prophet = mean_absolute_error(test_p['y'], pred_prophet)
print("Prophet MAE:", round(mae_prophet, 2))

# LSTM
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(values_scaled, sequence_length)
train_size_lstm = int(len(X_seq) * 0.8)
X_train_l, X_test_l = X_seq[:train_size_lstm], X_seq[train_size_lstm:]
y_train_l, y_test_l = y_seq[:train_size_lstm], y_seq[train_size_lstm:]

model_lstm_reg = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, input_shape=(sequence_length, 1))),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])
model_lstm_reg.compile(optimizer='adam', loss='mse')
model_lstm_reg.fit(X_train_l, y_train_l, epochs=30, batch_size=32, verbose=0)

pred_lstm_reg = scaler.inverse_transform(model_lstm_reg.predict(X_test_l)).flatten()
mae_lstm_reg = mean_absolute_error(scaler.inverse_transform(y_test_l), pred_lstm_reg)
print("LSTM MAE:", round(mae_lstm_reg, 2))

# =============================================
# 4. CLASSIFICATION : RF vs LSTM
# =============================================
print("\nÉTAPE 4 : Classification")
features_cls = [c for c in df.columns if c not in ['timestamp', 'value', 'classification', 'class_5']]
X_cls = df[features_cls]
y_cls = df['class_5']

le = LabelEncoder()
y_cls_encoded = le.fit_transform(y_cls)

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_cls, y_cls_encoded, test_size=0.2, shuffle=False)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train_rf), y=y_train_rf)
class_weight_dict = dict(zip(np.unique(y_train_rf), class_weights))

rf = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42)
rf.fit(X_train_rf, y_train_rf)
pred_rf = rf.predict(X_test_rf)
acc_rf = (pred_rf == y_test_rf).mean()
print("RF Accuracy:", round(acc_rf, 3))

# Feature importance
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=features_cls).sort_values(ascending=False)
print("Top 5 features:", feat_imp.head(5).to_dict())

# LSTM Classification
X_seq_cls, _ = create_sequences(df[features_cls].values, sequence_length)
y_seq_cls = le.transform(df['class_5'].iloc[sequence_length:])
y_seq_cls_cat = to_categorical(y_seq_cls)
train_size_cls = int(len(X_seq_cls) * 0.8)
X_train_cl, X_test_cl = X_seq_cls[:train_size_cls], X_seq_cls[train_size_cls:]
y_train_cl, y_test_cl = y_seq_cls_cat[:train_size_cls], y_seq_cls_cat[train_size_cls:]

model_lstm_cls = Sequential([
    Bidirectional(LSTM(64, return_sequences=True, input_shape=(sequence_length, X_seq_cls.shape[2]))),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(5, activation='softmax')
])
model_lstm_cls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_lstm_cls.fit(X_train_cl, y_train_cl, epochs=30, batch_size=32, verbose=0)

pred_lstm_cls = np.argmax(model_lstm_cls.predict(X_test_cl), axis=1)
acc_lstm_cls = (pred_lstm_cls == np.argmax(y_test_cl, axis=1)).mean()
print("LSTM Accuracy:", round(acc_lstm_cls, 3))

# =============================================
# 5. SAUVEGARDE & PRÉDICTION
# =============================================
best_reg = "LSTM" if mae_lstm_reg < mae_prophet else "Prophet"
best_cls = "RF" if acc_rf > acc_lstm_cls else "LSTM"

joblib.dump(model_prophet if best_reg == "Prophet" else model_lstm_reg, MODELS_DIR / "best_regression_model.pkl")
joblib.dump(rf if best_cls == "RF" else model_lstm_cls, MODELS_DIR / "best_classification_model.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")

# PRÉDICTION DEMAIN
last_date = df['timestamp'].iloc[-1] + pd.Timedelta(days=1)
last_seq = values_scaled[-sequence_length:].reshape(1, sequence_length, 1)

if best_reg == "LSTM":
    pred_tomorrow = float(scaler.inverse_transform(model_lstm_reg.predict(last_seq))[0][0])
else:
    future_df = pd.DataFrame()
    future_df['ds'] = [last_date]
    pred_tomorrow = float(model_prophet.predict(future_df)['yhat'].iloc[0])

# Ensembling
rf_pred = rf.predict_proba(X_cls.iloc[-1:].values)[0]
lstm_input = df[features_cls].iloc[-sequence_length:].values.reshape(1, sequence_length, -1)
lstm_pred = model_lstm_cls.predict(lstm_input)[0]

if rf_pred.max() > 0.7:
    pred_class = le.inverse_transform([np.argmax(rf_pred)])[0]
elif lstm_pred.max() > 0.8:
    pred_class = le.inverse_transform([np.argmax(lstm_pred)])[0]
else:
    pred_class = "Neutral"

result = {
    "predicted_score": round(pred_tomorrow, 1),
    "predicted_class": pred_class,
    "best_regression": best_reg,
    "best_classification": best_cls
}
with open(DATA_DIR / "latest_prediction.json", 'w') as f:
    json.dump(result, f, indent=2)

print("\nMEILLEUR RÉGRESSION :", best_reg)
print("MEILLEUR CLASSIFICATION :", best_cls)
print("PRÉDICTION DEMAIN :", round(pred_tomorrow, 1), "→", pred_class)
print("latest_prediction.json sauvegardé !")
