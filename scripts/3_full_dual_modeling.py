
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

# ========================================
# 1. CHARGEMENT & INSPECTION DES DONNÉES
# ========================================
print("=== 1. CHARGEMENT DES DONNÉES ===")
df = pd.read_csv('../data/fear_greed_data.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"Shape : {df.shape}")
print(f"Période : {df['timestamp'].min()} → {df['timestamp'].max()}")
print("\nInfo :")
print(df.info())
print("\nDescribe :")
print(df.describe())

# Visualisation
plt.figure(figsize=(15, 6))
plt.plot(df['timestamp'], df['value'], label='Fear & Greed Index')
plt.title('Évolution du Fear & Greed Index')
plt.xlabel('Date')
plt.ylabel('Score')
plt.legend()
plt.grid(True)
plt.savefig('/content/drive/MyDrive/Hedera Mood Oracle/images/fear_greed_evolution.png', dpi=150, bbox_inches='tight')
plt.show()

# ========================================
# 2. NETTOYAGE & 5 CLASSES
# ========================================
print("\n=== 2. NETTOYAGE & CLASSES ===")
df = df.dropna().reset_index(drop=True)

def classify_5(score):
    if score <= 25: return "Extreme Fear"
    elif score <= 46: return "Fear"
    elif score <= 54: return "Neutral"
    elif score <= 75: return "Greed"
    else: return "Extreme Greed"

df['class_5'] = df['value'].apply(classify_5)
print("Distribution des classes :")
print(df['class_5'].value_counts())

# ========================================
# 3. FEATURE ENGINEERING
# ========================================
print("\n=== 3. FEATURE ENGINEERING ===")
df['day_of_year'] = df['timestamp'].dt.dayofyear
df['month'] = df['timestamp'].dt.month
df['weekday'] = df['timestamp'].dt.weekday
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

# Lags
for lag in [1, 3, 7, 14, 30]:
    df[f'lag_{lag}'] = df['value'].shift(lag)

# Rolling windows
df['rolling_mean_7'] = df['value'].rolling(7).mean()
df['rolling_std_7'] = df['value'].rolling(7).std()
df['rolling_mean_30'] = df['value'].rolling(30).mean()
df['rolling_std_30'] = df['value'].rolling(30).std()

# Momentum
df['momentum_7'] = df['value'] - df['lag_7']

df = df.dropna().reset_index(drop=True)

# ========================================
# 4. PRÉPARATION DES DONNÉES
# ========================================
features = ['day_of_year', 'month', 'weekday', 'is_weekend',
            'lag_1', 'lag_3', 'lag_7', 'lag_14', 'lag_30',
            'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30', 'momentum_7']

X = df[features]
y_reg = df['value']
y_cls = df['class_5']

# Scaling pour modèles non-LSTM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encodage des classes
le = LabelEncoder()
y_cls_encoded = le.fit_transform(y_cls)
y_cls_categorical = to_categorical(y_cls_encoded)

# Séparation
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X_scaled, y_reg, test_size=0.2, shuffle=False)
_, _, y_train_cls, y_test_cls = train_test_split(X_scaled, y_cls_encoded, test_size=0.2, shuffle=False)
_, _, y_train_cls_cat, y_test_cls_cat = train_test_split(X_scaled, y_cls_categorical, test_size=0.2, shuffle=False)

# Séquence pour LSTM (30 jours)
def create_sequences(X, y, time_steps=30):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

X_seq, y_seq_reg = create_sequences(X_scaled, y_reg.values)
X_seq_cls, y_seq_cls = create_sequences(X_scaled, y_cls_categorical)

X_train_seq, X_test_seq, y_train_seq_reg, y_test_seq_reg = train_test_split(X_seq, y_seq_reg, test_size=0.2, shuffle=False)
X_train_seq_cls, X_test_seq_cls, y_train_seq_cls, y_test_seq_cls = train_test_split(X_seq_cls, y_seq_cls, test_size=0.2, shuffle=False)

# ========================================
# 5. MODÈLES DE RÉGRESSION
# ========================================
print("\n=== 5. RÉGRESSION ===")

# Prophet
prophet_df = df[['timestamp', 'value']].copy()
prophet_df.columns = ['ds', 'y']
train_size = int(len(prophet_df) * 0.8)
train_df = prophet_df.iloc[:train_size]
test_df = prophet_df.iloc[train_size:]

model_prophet = Prophet(daily_seasonality=True, yearly_seasonality=True)
model_prophet.fit(train_df)
future = model_prophet.make_future_dataframe(periods=len(test_df), freq='D')
forecast = model_prophet.predict(future)
pred_prophet = forecast.iloc[train_size:]['yhat'].values
mae_prophet = mean_absolute_error(test_df['y'], pred_prophet)
print(f"Prophet MAE: {mae_prophet:.2f}")

# LSTM Régression
model_lstm_reg = Sequential([
    LSTM(50, return_sequences=True, input_shape=(30, X_train_seq.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model_lstm_reg.compile(optimizer='adam', loss='mse')
model_lstm_reg.fit(X_train_seq, y_train_seq_reg, epochs=20, batch_size=32, verbose=0)
pred_lstm_reg = model_lstm_reg.predict(X_test_seq).flatten()
mae_lstm_reg = mean_absolute_error(y_test_seq_reg, pred_lstm_reg)
print(f"LSTM Régression MAE: {mae_lstm_reg:.2f}")

# ========================================
# 6. MODÈLES DE CLASSIFICATION
# ========================================
print("\n=== 6. CLASSIFICATION ===")

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train_cls)
pred_rf = rf.predict(X_test)
print("Random Forest :")
print(classification_report(y_test_cls, pred_rf, target_names=le.classes_))

# LSTM Classification
model_lstm_cls = Sequential([
    LSTM(50, return_sequences=True, input_shape=(30, X_train_seq_cls.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(5, activation='softmax')
])
model_lstm_cls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_lstm_cls.fit(X_train_seq_cls, y_train_seq_cls, epochs=20, batch_size=32, verbose=0)
pred_lstm_cls = np.argmax(model_lstm_cls.predict(X_test_seq_cls), axis=1)
print("LSTM Classification Accuracy:", (pred_lstm_cls == np.argmax(y_test_seq_cls, axis=1)).mean())

# ========================================
# 7. SAUVEGARDE & PRÉDICTION
# ========================================
joblib.dump(model_prophet, f"{BASE_DIR}/models/prophet_model.pkl")
model_lstm_reg.save(f"{BASE_DIR}/models/lstm_regression.h5")
model_lstm_cls.save(f"{BASE_DIR}/models/lstm_classification.h5")
joblib.dump(rf, f"{BASE_DIR}/models/rf_classifier.pkl")
joblib.dump(scaler, f"{BASE_DIR}/models/scaler.pkl")
joblib.dump(le, f"{BASE_DIR}/models/label_encoder.pkl")

print("TOUS LES MODÈLES SAUVEGARDÉS DANS :")
print(f"{BASE_DIR}/models/")


# Prédiction demain
last_seq = X_seq[-1:].copy()
pred_prophet_tomorrow = model_prophet.predict(pd.DataFrame({'ds': [df['timestamp'].iloc[-1] + pd.Timedelta(days=1)]}))['yhat'].values[0]
pred_lstm_reg_tomorrow = model_lstm_reg.predict(last_seq)[0][0]
pred_rf_tomorrow = le.inverse_transform(rf.predict(scaler.transform(X.iloc[-1:].values)))[0]

print(f"\nPRÉDICTION DEMAIN :")
print(f"   Prophet : {pred_prophet_tomorrow:.1f}")
print(f"   LSTM Reg : {pred_lstm_reg_tomorrow:.1f}")
print(f"   RF Class : {pred_rf_tomorrow}")

result = {
    "prophet_score": round(float(pred_prophet_tomorrow), 1),
    "lstm_reg_score": round(float(pred_lstm_reg_tomorrow), 1),
    "rf_class": pred_rf_tomorrow,
    "best_regression": "Prophet" if mae_prophet < mae_lstm_reg else "LSTM",
    "mae_prophet": round(mae_prophet, 2),
    "mae_lstm_reg": round(mae_lstm_reg, 2)
}
with open('../data/latest_prediction.json', 'w') as f:
    json.dump(result, f)

print("Tout sauvegardé !")
