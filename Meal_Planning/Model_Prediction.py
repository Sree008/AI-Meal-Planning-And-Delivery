import pickle
import pandas as pd
import numpy as np

# Load the models for breakfast, lunch, and dinner
with open('RandomForest_Breakfast_model.pkl', 'rb') as file:
    breakfast_model = pickle.load(file)

with open('RandomForest_Lunch_model.pkl', 'rb') as file:
    lunch_model = pickle.load(file)

with open('RandomForest_Dinner_model.pkl', 'rb') as file:
    dinner_model = pickle.load(file)

# Load new patient data (unseen data)
new_patient_data = pd.read_csv('new_patient_data.csv')
X_new = new_patient_data[['Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'BMI', 'Activity Level', 'Health Condition',
                     'Allergies', 'Reason for Admission', 'Recommended Daily Calories', 'Protein Requirement (g)',
                     'Carbohydrate Requirement (g)', 'Fats Requirement (g)', 'Portion Size', 'Therapeutic Diet']]

# Convert categorical data to numerical
X_new_encoded = pd.get_dummies(X_new)

# Ensure that the encoded columns match the model input
# Adjust columns if necessary (e.g., add missing columns with 0 values or reorder them)
columns_needed_by_model = ['Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'BMI', 'Activity Level', 'Health Condition',
                     'Allergies', 'Reason for Admission', 'Recommended Daily Calories', 'Protein Requirement (g)',
                     'Carbohydrate Requirement (g)', 'Fats Requirement (g)', 'Portion Size', 'Therapeutic Diet']
X_new_encoded = X_new_encoded.reindex(columns=columns_needed_by_model, fill_value=0)

# Predict each meal separately
new_patient_data['Predicted Breakfast'] = breakfast_model.predict(X_new_encoded)
new_patient_data['Predicted Lunch'] = lunch_model.predict(X_new_encoded)
new_patient_data['Predicted Dinner'] = dinner_model.predict(X_new_encoded)

# Merge with room number information
patient_room_mapping = pd.read_csv('patient_room_mapping.csv')
predicted_meals_with_rooms = pd.merge(new_patient_data[['Patient ID', 'Predicted Breakfast', 'Predicted Lunch', 'Predicted Dinner']],
                                      patient_room_mapping, on='Patient ID')

# Save to a CSV file
predicted_meals_with_rooms.to_csv('predicted_meals_with_rooms.csv', index=False)

print("Meal predictions (breakfast, lunch, and dinner) saved to predicted_meals_with_rooms.csv")
