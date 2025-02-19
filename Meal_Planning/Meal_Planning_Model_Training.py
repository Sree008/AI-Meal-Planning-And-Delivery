import pandas as pd  # For handling datasets
from sklearn.model_selection import train_test_split  # Splitting data into train/test
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For label encoding and scaling
from sklearn.ensemble import RandomForestClassifier  # RandomForest model
from sklearn.svm import SVC  # Support Vector Classifier
import joblib  # To save and load models
import numpy as np  # For arrays and mathematical functions
from xgboost import XGBClassifier  # XGBoost Classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt  # Plotting graphs
import seaborn as sns  # Visualization library
import pickle

# For Deep Learning Neural Network model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Function to plot confusion matrix
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {title}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Load the dataset
data = pd.read_csv('combined_patient_meal_data_indian.csv')

# Preprocessing the data
# Create a new feature "Base Therapeutic Diet" by stripping off the allergy information
data['Base Therapeutic Diet'] = data['Therapeutic Diet'].apply(lambda x: x.split(' (Excludes')[0])

# Label encoding the categorical features
categorical_features = ['Gender', 'Activity Level', 'Allergies', 'Health Condition', 'Reason for Admission',
                        'Portion Size', 'Therapeutic Diet','Breakfast (With Allergies)','Lunch (With Allergies)',
                        'Dinner (With Allergies)']
label_encoders = {}

for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le

# Normalize continuous features
cols_to_normalize = ['Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'BMI', 'Activity Level', 'Health Condition',
                     'Allergies', 'Reason for Admission', 'Recommended Daily Calories', 'Protein Requirement (g)',
                     'Carbohydrate Requirement (g)', 'Fats Requirement (g)', 'Portion Size', 'Therapeutic Diet',
                     'Breakfast (With Allergies)', 'Lunch (With Allergies)','Dinner (With Allergies)']

mins = data[cols_to_normalize].min()
maxs = data[cols_to_normalize].max()
data[cols_to_normalize] = (data[cols_to_normalize] - mins) / (maxs - mins)

# Define input features
X = data[cols_to_normalize]
Encoded_data = data[cols_to_normalize]


# Encode the target variables (Breakfast, Lunch, Dinner)
target_labels = {
    'Breakfast': LabelEncoder().fit_transform(data['Breakfast (With Allergies)']),
    'Lunch': LabelEncoder().fit_transform(data['Lunch (With Allergies)']),
    'Dinner': LabelEncoder().fit_transform(data['Dinner (With Allergies)'])
}
#print(target_labels)
#Encoded_data.merge(target_labels, left_index=True, right_index=True)
Encoded_data.to_csv('Encoded_data.csv', index=False)
# Models to evaluate
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
    #'SVM': SVC(probability=True, random_state=42)
}

# Function to train and evaluate models
def train_and_evaluate_models(X, target_labels, models):
    for target_label, y in target_labels.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Evaluating models for {target_label}")
        for name, model in models.items():
            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # Accuracy and classification report
            print(f"--- {name} ---")
            print(f"Accuracy for {target_label}: {accuracy_score(y_test, y_pred)}")
            print(classification_report(y_test, y_pred))
            # Confusion matrix
            #cm = confusion_matrix(y_test, y_pred)
            #plot_confusion_matrix(cm, title=f'{name} ({target_label})')
            # Save the model
            model_filename = f'{name}_{target_label}_model.pkl'
            with open(model_filename, 'wb') as file:
                pickle.dump(model, file)
            #joblib.dump(model, model_filename)  # Save the trained model
            #print(f"Model saved as {model_filename}")

# Train and evaluate traditional models (RandomForest, SVM, XGBoost)
train_and_evaluate_models(X, target_labels, models)

# Neural Network Implementation for Breakfast Prediction
def train_neural_network(X, y, label_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the data for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the neural network model
    model = Sequential([
        Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),  # Input layer
        Dense(128, activation='relu'),  # Hidden layer 1
        Dense(64, activation='relu'),   # Hidden layer 2
        Dense(len(np.unique(y_train)), activation='softmax')  # Output layer (for multiclass classification)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

    # Evaluate the model
    accuracy = model.evaluate(X_test_scaled, y_test)
    print(f"Neural Network Accuracy for {label_name}: {accuracy[1]}")

    # Save the trained neural network model
    nn_model_filename = f'nn_model_{label_name}.h5'
    model.save(nn_model_filename)
    print(f"Neural network model saved as {nn_model_filename}")

# Train neural network for 'Breakfast' (example)
train_neural_network(X, target_labels['Breakfast'], 'Breakfast')
train_neural_network(X, target_labels['Lunch'], 'Lunch')
train_neural_network(X, target_labels['Dinner'], 'Dinner')
