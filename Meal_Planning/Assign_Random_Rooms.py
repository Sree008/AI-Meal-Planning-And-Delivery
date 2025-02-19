import random
from random import shuffle
import pandas as pd

# Step 1: Generate the hospital structure
def generate_hospital():
    blocks = ['A', 'B', 'C']
    hospital_rooms = []

    # Create the hospital structure with 3 blocks, 5 floors, and 20 rooms per floor
    for block in blocks:
        for floor in range(1, 6):
            for room in range(1, 21):
                room_number = f"{block}{floor:01}{room:02}"
                hospital_rooms.append(room_number)
    shuffle(hospital_rooms)

    return hospital_rooms

# Step 2: Assign 200 patients to random rooms
def assign_patients_to_rooms(patient_ids, hospital_rooms):
    assigned_rooms = random.sample(hospital_rooms, len(patient_ids))
    patient_room_mapping = pd.DataFrame({
        'Patient ID': patient_ids,
        'Room Number': assigned_rooms
    })
    return patient_room_mapping

# Example usage with generated patient IDs
hospital_rooms = generate_hospital()
patient_ids = [f'P{patient_id:03}' for patient_id in range(1, 201)]

# Assign patients to rooms
patient_room_mapping = assign_patients_to_rooms(patient_ids, hospital_rooms[:200])
patient_room_mapping.to_csv('patient_room_mapping.csv', index=False)

print(patient_room_mapping.head())  # Check the assigned rooms
