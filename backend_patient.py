
import pandas as pd
from transformers import AutoTokenizer, TFAutoModel
import tensorflow as tf
import numpy as np

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

def combine_symptoms(row):
    symptoms = [str(row[col]) for col in row.index if "Symptom" in col and pd.notna(row[col])]
    return " ".join(symptoms)

train_texts = train_df.apply(combine_symptoms, axis=1).tolist()
test_texts = test_df.apply(combine_symptoms, axis=1).tolist()

# Labels: Convert Criticality → 0/1/2
label_map = {"Mild": 0, "Moderate": 1, "Critical": 2}
train_labels = train_df["Criticality"].map(label_map)
test_labels = test_df["Criticality"].map(label_map)

# Drop NaN labels
train_mask = ~train_labels.isna()
test_mask = ~test_labels.isna()

train_texts = [t for t, m in zip(train_texts, train_mask) if m]
train_labels = train_labels[train_mask].astype(int).to_numpy()

test_texts = [t for t, m in zip(test_texts, test_mask) if m]
test_labels = test_labels[test_mask].astype(int).to_numpy()

model_name = "Zabihin/Symptom_to_Diagnosis"
tokenizer = AutoTokenizer.from_pretrained(model_name)

max_len = 128
X_train = tokenizer(train_texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="tf")
X_test = tokenizer(test_texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="tf")

encoder = TFAutoModel.from_pretrained(model_name)

input_ids = tf.keras.Input(shape=(X_train["input_ids"].shape[1],), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.Input(shape=(X_train["attention_mask"].shape[1],), dtype=tf.int32, name="attention_mask")

# Encoder output
outputs = encoder(input_ids, attention_mask=attention_mask)
cls_token = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation

# New head
x = tf.keras.layers.Dense(256, activation="relu")(cls_token)
x = tf.keras.layers.Dropout(0.2)(x)
logits = tf.keras.layers.Dense(3, activation="softmax")(x)   # <-- 3 classes

model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x={
        "input_ids": X_train["input_ids"],
        "attention_mask": X_train["attention_mask"]
    },
    y=tf.convert_to_tensor(train_labels, dtype=tf.int32),
    validation_data=(
        {
            "input_ids": X_test["input_ids"],
            "attention_mask": X_test["attention_mask"]
        },
        tf.convert_to_tensor(test_labels, dtype=tf.int32)
    ),
    epochs=3,
    batch_size=8
)

# Save in .keras format
model.save("criticality_model_tf_3class.keras")
print("✅ Model saved at criticality_model_tf_3class.keras")
