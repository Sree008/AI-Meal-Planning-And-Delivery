# encoding_preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load the combined dataset generated earlier
data = pd.read_csv('combined_patient_meal_data_indian.csv')

# List of target columns
target_cols = ['Breakfast (With Allergies)', 'Lunch (With Allergies)', 'Dinner (With Allergies)']

# Use all columns except Patient ID and the target columns as features
feature_cols = [col for col in data.columns if col not in (['Patient ID'] + target_cols)]

# Split into features (X) and targets (y)
X = data[feature_cols]
y = data[target_cols]

# Identify categorical and numerical features.
# (Based on the generated dataset, these columns are categorical:)
categorical_features = ['Gender', 'Activity Level', 'Allergies', 'Health Condition',
                        'Reason for Admission', 'Portion Size', 'Therapeutic Diet']

# The remaining features (if not in categorical_features) are assumed numeric.
numerical_features = [col for col in feature_cols if col not in categorical_features]

# Create a ColumnTransformer with MinMaxScaler for numerical features
# and OneHotEncoder for categorical features.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Fit and transform X
X_encoded = preprocessor.fit_transform(X)

# Save the preprocessor for use later during prediction.
joblib.dump(preprocessor, 'preprocessor.joblib')

# For the targets, we need to encode them into numeric labels.
# We create one LabelEncoder per target column.
target_encoders = {}
y_encoded = y.copy()
for col in target_cols:
    le = LabelEncoder()
    y_encoded[col] = le.fit_transform(y[col])
    target_encoders[col] = le

# Save the target encoders for later use
joblib.dump(target_encoders, 'target_encoders.joblib')

# Optionally, you can split into training and testing sets now.
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Save the split data for model training (or you can continue in the training code).
joblib.dump((X_train, X_test, y_train, y_test), 'train_test_data.joblib')

print("Data encoding and preprocessing complete with MinMax normalization. Preprocessor and encoders saved.")
