# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers

st.set_page_config(page_title="NVCA Prediction App", layout="wide")
st.title("🔍 Predicting New Violent Criminal Activity (NVCA)")

# --- File upload ---
st.sidebar.header("Step 1: Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom file uploaded and loaded.")
else:
    df = pd.read_csv("APPR_Fulton_Scored_Data.csv")
    st.info("ℹ️ No file uploaded. Using default dataset.")

df = df[df['release'] == 1]  # Released individuals only

# --- Optional: Download sample CSV ---
with open("APPR_Fulton_Scored_Data.csv", "rb") as file:
    st.sidebar.download_button(
        label="Download Sample CSV",
        data=file,
        file_name="APPR_Fulton_Scored_Data.csv",
        mime="text/csv"
    )

st.write("### Data preview:")
st.dataframe(df.head(100))

# --- Define features and target ---
features = [
    'factor_1', 'factor_2', 'factor_3', 'factor_4', 'factor_5', 'factor_6',
    'factor_7', 'factor_8', 'factor_9', 'race1', 'sex1',
    'time_at_risk_nvca_new', 'nvca_score'
]
target = 'nvca'

df_model = df[features + [target]].dropna()
X = df_model[features].astype(np.float32)
y = df_model[target].astype(np.float32)

# --- Split data ---
X_temp, X_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- Sidebar model config ---
st.sidebar.header("Step 2: Model Configuration")
use_class_weights = st.sidebar.checkbox("Use class weighting", value=True)
hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 3, 2)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)
learning_rate = st.sidebar.select_slider("Learning Rate", options=[0.1, 0.01, 0.001], value=0.001)

# --- Define model ---
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=regularizers.l2(0.001)))
for _ in range(hidden_layers - 1):
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(dropout_rate))
model.add(Dense(1, activation='sigmoid'))

opt = Adam(learning_rate=learning_rate)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

class_weight = None
if use_class_weights:
    from sklearn.utils import class_weight
    class_weights_array = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(class_weights_array))

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

if st.sidebar.button("Train Model"):
    with st.spinner("Training model..."):
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=50,
            batch_size=128,
            verbose=0,
            callbacks=[early_stop],
            class_weight=class_weight
        )

    # --- Evaluate ---
    y_pred_prob = model.predict(X_test_scaled).flatten()
    threshold = 0.5
    y_pred = (y_pred_prob >= threshold).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("📊 Model Performance")
    st.write(f"**Precision:** {precision:.3f}")
    st.write(f"**Recall:** {recall:.3f}")
    st.write(f"**F1 Score:** {f1:.3f}")
    st.write(f"**AUC:** {auc:.3f}")

    st.subheader("🧮 Confusion Matrix")
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], va='center', ha='center')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(fig)


