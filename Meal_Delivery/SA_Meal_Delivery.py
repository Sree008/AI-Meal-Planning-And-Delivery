import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns

# -----------------------
# Helper Functions
# -----------------------

def calculate_distance(room_a, room_b):
    """
    Returns the travel time in seconds between two rooms.
    - Block crossing: 60 sec per block difference.
    - Floor traversal: 60 sec per floor difference.
    - Room crossing: 5 sec per room number difference.
    If either room is "Kitchen", travel time is assumed to be 0.
    """
    if room_a == "Kitchen" or room_b == "Kitchen":
        return 0
    block_a, floor_a, room_a_num = room_a[0], int(room_a[1]), int(room_a[2:])
    block_b, floor_b, room_b_num = room_b[0], int(room_b[1]), int(room_b[2:])
    block_time = abs(ord(block_a) - ord(block_b)) * 60
    floor_time = abs(floor_a - floor_b) * 60
    room_time = abs(room_a_num - room_b_num) * 5
    return block_time + floor_time + room_time


def calculate_total_delivery_time(chromosome, room_mapping, meal_mapping, meal_type,
                                  num_staffs=5, max_trip_capacity=10):
    """
    Calculates the overall delivery time (in seconds) when exactly `num_staffs`
    are available. The chromosome is split into `num_staffs` contiguous segments.
    If a staff’s assigned deliveries exceed `max_trip_capacity`, then that staff makes
    multiple trips. Each trip starts and ends at the Kitchen.

    The time for each trip is calculated as:
      - Travel time from Kitchen to the first room,
      - 30 seconds per room visit plus travel time between rooms,
      - Travel time returning from the last room to the Kitchen.

    Since staffs work in parallel, the overall delivery time is the maximum of the staffs’ times.
    """
    n = len(chromosome)
    chunk_size = math.ceil(n / num_staffs)
    staff_times = []
    for i in range(num_staffs):
        staff_patients = chromosome[i * chunk_size: (i + 1) * chunk_size]
        total_staff_time = 0
        for j in range(0, len(staff_patients), max_trip_capacity):
            trip = staff_patients[j:j + max_trip_capacity]
            trip_time = 0
            current_room = "Kitchen"
            for patient in trip:
                if meal_mapping[patient][meal_type]:
                    next_room = room_mapping[patient]
                    travel_time = calculate_distance(current_room, next_room)
                    trip_time += travel_time + 30  # add 30 sec for room visit
                    current_room = next_room
            trip_time += calculate_distance(current_room, "Kitchen")
            total_staff_time += trip_time
        staff_times.append(total_staff_time)
    total_time = max(staff_times)
    return total_time


def display_delivery_plan(chromosome, room_mapping, meal_mapping, meal_type,
                          num_staffs=5, max_trip_capacity=10):
    n = len(chromosome)
    chunk_size = math.ceil(n / num_staffs)
    for staff_index in range(num_staffs):
        staff_patients = chromosome[staff_index * chunk_size: (staff_index + 1) * chunk_size]
        route_str = ""
        num_trips = math.ceil(len(staff_patients) / max_trip_capacity)
        for trip in range(num_trips):
            trip_patients = staff_patients[trip * max_trip_capacity: (trip + 1) * max_trip_capacity]
            trip_route = "Kitchen -> "
            for patient in trip_patients:
                if meal_mapping[patient][meal_type]:
                    trip_route += f"{room_mapping[patient]} -> "
            trip_route += "Kitchen"
            route_str += f"Trip {trip + 1}: {trip_route}   "
        print(f"Staff {staff_index + 1} delivery route: {route_str}")


# -----------------------
# Data Loading
# -----------------------

patient_room_mapping = pd.read_csv('predicted_meals_with_rooms.csv')
patient_ids = list(patient_room_mapping['Patient ID'])
room_mapping = dict(zip(patient_room_mapping['Patient ID'], patient_room_mapping['Room Number']))
meal_mapping = {
    patient: {
        'Breakfast':
            patient_room_mapping.loc[patient_room_mapping['Patient ID'] == patient, 'Breakfast (With Allergies)'].values[0],
        'Lunch': patient_room_mapping.loc[patient_room_mapping['Patient ID'] == patient, 'Lunch (With Allergies)'].values[0],
        'Dinner': patient_room_mapping.loc[patient_room_mapping['Patient ID'] == patient, 'Dinner (With Allergies)'].values[0]
    }
    for patient in patient_ids
}

# -----------------------
# Optimization Parameters
# -----------------------

num_staffs = 5  # Exactly 5 staffs are used.
max_trip_capacity = 10  # Maximum number of deliveries per trip.
optimal_threshold = 600  # Example threshold (in seconds) for an acceptable total delivery time.
max_generations = 1000
stall_generations = 100
improvement_epsilon = 1e-3

# -----------------------
# 2. Simulated Annealing (SA)
# -----------------------

def simulated_annealing(patient_ids, room_mapping, meal_mapping, meal_type,
                        num_staffs=5, max_trip_capacity=10,
                        initial_temp=1000, final_temp=1, alpha=0.995, iterations=10000):
    current_solution = patient_ids.copy()
    random.shuffle(current_solution)
    current_cost = calculate_total_delivery_time(current_solution, room_mapping, meal_mapping, meal_type,
                                                 num_staffs, max_trip_capacity)
    best_solution = current_solution.copy()
    best_cost = current_cost
    best_cost_history = [current_cost]
    temperature = initial_temp

    for i in range(iterations):
        neighbor = current_solution.copy()
        idx1, idx2 = random.sample(range(len(neighbor)), 2)
        neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        neighbor_cost = calculate_total_delivery_time(neighbor, room_mapping, meal_mapping, meal_type,
                                                      num_staffs, max_trip_capacity)
        cost_diff = neighbor_cost - current_cost
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temperature):
            current_solution = neighbor.copy()
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
        best_cost_history.append(best_cost)
        temperature *= alpha
        if temperature < final_temp or best_cost < optimal_threshold:
            print(f"SA converged after {i + 1} iterations.")
            break
    return best_solution, best_cost, best_cost_history

results = {}  # Store final best cost and performance history

for meal_type in ['Breakfast', 'Lunch', 'Dinner']:
    print(f"\n--- {meal_type} ---")

# SA
    sa_solution, sa_cost, sa_history = simulated_annealing(
        patient_ids, room_mapping, meal_mapping, meal_type,
        num_staffs=num_staffs, max_trip_capacity=max_trip_capacity)
    print(f"SA - Total delivery time for {meal_type}: {sa_cost:.2f} seconds")

    print(f"Delivery plan for {meal_type} (SA):")
    display_delivery_plan(sa_solution, room_mapping, meal_mapping, meal_type,
                          num_staffs=num_staffs, max_trip_capacity=max_trip_capacity)
