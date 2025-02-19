import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Constants
NUM_PATIENTS = 200
ROOMS_PER_WARD = 10
WARDS_PER_FLOOR = 5
FLOORS_PER_BLOCK = 3
BLOCKS = ['A', 'B', 'C']

# Meal schedule time windows
meal_windows = {
    'Breakfast': (7, 30, 9, 0),  # 7:30 AM to 9:00 AM
    'Lunch': (12, 30, 13, 30),  # 12:30 PM to 1:30 PM
    'Dinner': (19, 30, 21, 0)  # 7:30 PM to 9:00 PM
}


# Function to generate a random time within the given window
def generate_random_time(hour_start, minute_start, hour_end, minute_end):
    start_time = datetime(2024, 1, 1, hour_start, minute_start)  # Just a reference date
    end_time = datetime(2024, 1, 1, hour_end, minute_end)
    random_time = start_time + timedelta(seconds=random.randint(0, int((end_time - start_time).total_seconds())))
    return random_time.time()


# Function to generate delivery data for each patient
def generate_delivery_data(patient_id):
    delivery_data = []
    # Assign patient to a room, ward, floor, and block
    room_number = random.randint(1, ROOMS_PER_WARD)
    ward = random.randint(1, WARDS_PER_FLOOR)
    floor = random.randint(1, FLOORS_PER_BLOCK)
    block = random.choice(BLOCKS)
    location = f"Room {room_number}, Ward {ward}, Floor {floor}, Block {block}"

    for meal in meal_windows:
        # Generate scheduled delivery time
        scheduled_time = generate_random_time(*meal_windows[meal])

        # Generate actual delivery time (randomly delayed by 0 to 45 minutes)
        delay_in_minutes = random.randint(0, 45)
        actual_time = (datetime.combine(datetime.today(), scheduled_time) + timedelta(minutes=delay_in_minutes)).time()

        # Determine delivery status
        if delay_in_minutes <= 10:
            delivery_status = "On Time"
        elif delay_in_minutes <= 30:
            delivery_status = "Late"
        else:
            delivery_status = "Very Late"

        # Store delivery record for the patient
        delivery_data.append([
            f"P{patient_id:03}", meal, scheduled_time, actual_time, location, delivery_status, delay_in_minutes
        ])

    return delivery_data


# Create list to store all meal delivery data
meal_delivery_data = []

# Generate delivery data for each patient
for patient_id in range(1, NUM_PATIENTS + 1):
    meal_delivery_data.extend(generate_delivery_data(patient_id))

# Create a DataFrame for meal delivery data
meal_delivery_df = pd.DataFrame(meal_delivery_data, columns=[
    'Patient ID', 'Meal Type', 'Scheduled Delivery Time', 'Actual Delivery Time', 'Location',
    'Delivery Status', 'Delay (Minutes)'
])

# Save to CSV file
meal_delivery_df.to_csv('meal_delivery_data.csv', index=False)
