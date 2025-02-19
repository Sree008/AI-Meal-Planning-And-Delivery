import random
import pandas as pd
import numpy as np

# Function to calculate BMI
def calculate_bmi(weight, height):
    height_m = height / 100  # Convert height to meters
    return round(weight / (height_m ** 2), 2)

# Function to calculate BMR (Basal Metabolic Rate)
def calculate_bmr(weight, height, age, gender):
    if gender == 'Male':
        return round(88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age), 2)
    else:
        return round(447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age), 2)

# Function to calculate recommended daily calories based on BMR and activity level
def calculate_calories(bmr, activity_level):
    activity_multipliers = {
        'Sedentary': 1.2,
        'Lightly Active': 1.375,
        'Moderately Active': 1.55,
        'Very Active': 1.725
    }
    return round(bmr * activity_multipliers[activity_level], 2)

# Functions to calculate required protein, carbohydrates and fats
def calculate_protein(recommended_calories):
    return round((recommended_calories * np.random.uniform(0.2, 0.3) / 4))

def calculate_carb(recommended_calories):
    return round((recommended_calories * np.random.uniform(0.45, 0.55) / 4))

def calculate_fats(recommended_calories):
    return round((recommended_calories * np.random.uniform(0.2, 0.45) / 9))

# Function to get therapeutic diet recommendation
def get_therapeutic_diet(health_condition, allergies):
    therapeutic_diet_recommendations = {
        'Diabetes': 'Low-Carb & High-Fiber Diet',
        'Hypertension': 'Low-Sodium & DASH Diet',
        'Cardiovascular Disease': 'Heart-Healthy Diet (Low Saturated Fat)',
        'None': 'Balanced Diet'
    }
    diet = therapeutic_diet_recommendations.get(health_condition, 'Balanced Diet')

    # Add allergies to diet if present
    if allergies != 'None':
        diet += f" (Excludes {allergies})"
    return diet

# Indian therapeutic meal recommendations
therapeutic_meal_recommendations = {
    'Low-Carb & High-Fiber Diet': {
        'Breakfast': 'Upma with Vegetables',
        'Lunch': 'Vegetable Khichdi',
        'Dinner': 'Grilled Chicken with Stir-Fried Vegetables'
    },
    'Low-Sodium & DASH Diet': {
        'Breakfast': 'Idli with Sambar',
        'Lunch': 'Roti with Palak Paneer',
        'Dinner': 'Dal Soup with Roti'
    },
    'Heart-Healthy Diet (Low Saturated Fat)': {
        'Breakfast': 'Oats Porridge with Milk',
        'Lunch': 'Brown Rice with Dal and Sabzi',
        'Dinner': 'Masoor Dal with Rice and Salad'
    },
    'Balanced Diet': {
        'Breakfast': 'Scrambled Eggs with Toast',
        'Lunch': 'Roti with Mixed Vegetable Curry',
        'Dinner': 'Roti with Rajma Curry'
    }
}

# Function to adjust meals based on allergies
def adjust_meal_for_allergies(meal, allergies):
    if allergies != 'None':
        if 'Dairy' in allergies and 'Milk' in meal:
            meal = meal.replace('Milk', 'Almond Milk')
        if 'Nuts' in allergies and 'Vegetable Khichdi' in meal:
            meal = meal.replace('Vegetable Khichdi', 'Brown Rice Pulao')
        if 'Gluten' in allergies and 'Roti' in meal:
            meal = meal.replace('Roti', 'Gluten-Free Roti')
        if 'Seafood' in allergies and 'Fish' in meal:
            meal = meal.replace('Fish', 'Grilled Chicken')
    return meal

# Create a list to store the generated patient data
patients_data = []
meals_data = []

# Sample generator logic for patients
for patient_id in range(1, 201):
    age = random.randint(18, 80)
    gender = random.choice(['Male', 'Female'])
    height = random.randint(150, 190)  # Height in cm
    weight = random.randint(50, 100)  # Weight in kg
    bmi = calculate_bmi(weight, height)
    activity_level = random.choice(['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active'])
    bmr = calculate_bmr(weight, height, age, gender)
    recommended_calories = calculate_calories(bmr, activity_level)
    allergies = random.choice(['None', 'Dairy', 'Nuts', 'Gluten', 'Seafood'])
    health_condition = random.choice(['Diabetes', 'Hypertension', 'Cardiovascular Disease', 'None'])
    reason_for_admission = random.choice(['Surgery', 'Recovery', 'Chronic Illness Management', 'Routine Checkup'])

    # Append patient data
    patients_data.append([f"P{patient_id:03}", age, gender, height, weight, bmi, activity_level,
                          recommended_calories, allergies, health_condition, reason_for_admission])

    # Get therapeutic diet based on health condition and allergies
    therapeutic_diet = get_therapeutic_diet(health_condition, allergies)

    # Without allergy modifications (Base therapeutic diet)
    base_diet = therapeutic_diet.split(" (Excludes")[0]

    # Use the therapeutic diet to recommend meals
    if base_diet in therapeutic_meal_recommendations:
        breakfast = therapeutic_meal_recommendations[base_diet]['Breakfast']
        lunch = therapeutic_meal_recommendations[base_diet]['Lunch']
        dinner = therapeutic_meal_recommendations[base_diet]['Dinner']

        # Adjust meals based on allergies
        breakfast_with_allergies = adjust_meal_for_allergies(breakfast, allergies)
        lunch_with_allergies = adjust_meal_for_allergies(lunch, allergies)
        dinner_with_allergies = adjust_meal_for_allergies(dinner, allergies)
    else:
        # Fallback to a balanced diet if no matching diet is found
        breakfast_with_allergies = 'Scrambled Eggs'
        lunch_with_allergies = 'Roast Beef Sandwich'
        dinner_with_allergies = 'Pasta with Marinara Sauce'

    # Calculate nutrient requirements
    protein_req = calculate_protein(recommended_calories)
    carb_req = calculate_carb(recommended_calories)
    fats_req = calculate_fats(recommended_calories)
    portion_size = random.choice(['Small', 'Standard', 'Large'])

    # Append meal data
    meals_data.append([f"P{patient_id:03}", breakfast, breakfast_with_allergies, lunch, lunch_with_allergies,
                       dinner, dinner_with_allergies, protein_req, carb_req, fats_req, portion_size, therapeutic_diet])

# Create DataFrames for patients and meals
patients_df = pd.DataFrame(patients_data, columns=[
    'Patient ID', 'Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'BMI', 'Activity Level',
    'Recommended Daily Calories', 'Allergies', 'Health Condition', 'Reason for Admission'])

meals_df = pd.DataFrame(meals_data, columns=[
    'Patient ID', 'Breakfast (No Allergies)', 'Breakfast (With Allergies)', 'Lunch (No Allergies)',
    'Lunch (With Allergies)', 'Dinner (No Allergies)', 'Dinner (With Allergies)', 'Protein Requirement (g)',
    'Carbohydrate Requirement (g)', 'Fats Requirement (g)', 'Portion Size', 'Therapeutic Diet'])

# Merge the two dataframes into a single dataframe
combined_df = pd.merge(patients_df, meals_df, on='Patient ID')

# Save the combined dataframe to a CSV file
#combined_df.to_csv('combined_patient_meal_data_indian.csv', index=False)

new_patient_data_df = pd.merge(patients_df[['Patient ID', 'Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'BMI', 'Activity Level',
    'Recommended Daily Calories', 'Allergies', 'Health Condition', 'Reason for Admission']], combined_df[['Patient ID', 'Protein Requirement (g)',
    'Carbohydrate Requirement (g)', 'Fats Requirement (g)', 'Portion Size', 'Therapeutic Diet','Breakfast (With Allergies)',
    'Lunch (With Allergies)', 'Dinner (With Allergies)']], on='Patient ID', how='left')

new_patient_data_df.to_csv('combined_patient_meal_data_indian.csv', index=False)
