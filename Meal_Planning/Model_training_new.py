# train_models.py
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# For neural network
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load training and testing data
X_train, X_test, y_train, y_test = joblib.load('train_test_data.joblib')

# Load target encoders for inverse mapping later
target_encoders = joblib.load('target_encoders.joblib')
target_cols = ['Breakfast (With Allergies)', 'Lunch (With Allergies)', 'Dinner (With Allergies)']

##############################
# 1. Random Forest Classifier
##############################
print("Training Random Forest MultiOutputClassifier...")
rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
rf.fit(X_train, y_train)

# Predict on test set
rf_preds = rf.predict(X_test)

print("\nRandom Forest Classifier Results:")
for i, col in enumerate(target_cols):
    acc = accuracy_score(y_test[col], rf_preds[:, i])
    print(f"Accuracy for {col}: {acc:.4f}")
    cm = confusion_matrix(y_test[col], rf_preds[:, i])
    print(f"Confusion Matrix for {col}:\n{cm}\n")
    print(classification_report(y_test[col], rf_preds[:, i], target_names=target_encoders[col].classes_))

joblib.dump(rf, 'rf_model.joblib')

##############################
# 2. XGBoost Classifier
##############################
print("\nTraining XGBoost MultiOutputClassifier...")
xgb = MultiOutputClassifier(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
xgb.fit(X_train, y_train)

# Predict on test set
xgb_preds = xgb.predict(X_test)

print("\nXGBoost Classifier Results:")
for i, col in enumerate(target_cols):
    acc = accuracy_score(y_test[col], xgb_preds[:, i])
    print(f"Accuracy for {col}: {acc:.4f}")
    cm = confusion_matrix(y_test[col], xgb_preds[:, i])
    print(f"Confusion Matrix for {col}:\n{cm}\n")
    print(classification_report(y_test[col], xgb_preds[:, i], target_names=target_encoders[col].classes_))

joblib.dump(xgb, 'xgb_model.joblib')

##############################
# 3. Neural Network (Keras)
##############################
print("\nTraining Neural Network model...")

# To build a neural network that outputs three predictions, we create three output layers.
input_dim = X_train.shape[1]
inputs = Input(shape=(input_dim,))

# Shared hidden layers
x = Dense(128, activation='relu')(inputs)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)

# For each target, determine number of classes from the encoder
output_layers = []
losses = {}
metrics = {}
for col in target_cols:
    n_classes = len(target_encoders[col].classes_)
    out = Dense(n_classes, activation='softmax', name=col)(x)
    output_layers.append(out)
    losses[col] = 'sparse_categorical_crossentropy'
    metrics[col] = ['accuracy']

model = Model(inputs=inputs, outputs=output_layers)
model.compile(optimizer='adam', loss=losses, metrics=metrics)

# Prepare y for neural network training: each target as a separate array.
y_train_nn = [y_train[col].values for col in target_cols]
y_test_nn = [y_test[col].values for col in target_cols]

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train_nn,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=16,
                    callbacks=[early_stop],
                    verbose=1)

# Evaluate on test set
print("\nNeural Network Evaluation on Test Set:")
nn_eval = model.evaluate(X_test, y_test_nn, verbose=0)
for i, col in enumerate(target_cols):
    loss = nn_eval[1 + i * 2]  # Each output has loss and accuracy
    acc = nn_eval[2 + i * 2]
    print(f"{col} -- Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# Save the neural network model
model.save('nn_model.h5')

print("Model training complete. Models saved as: rf_model.joblib, xgb_model.joblib, and nn_model.h5")
