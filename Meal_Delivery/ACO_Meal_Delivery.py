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
# 4. Ant Colony Optimization (ACO)
# -----------------------

def ant_colony_optimization(patient_ids, room_mapping, meal_mapping, meal_type,
                            num_staffs=5, max_trip_capacity=10,
                            num_ants=50, iterations=100, alpha=1.0, beta=2.0,
                            evaporation_rate=0.5, Q=100):
    n = len(patient_ids)
    pheromone = np.ones((n, n))
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                heuristic[i, j] = 1.0 / (
                            calculate_distance(room_mapping[patient_ids[i]], room_mapping[patient_ids[j]]) + 0.1)
            else:
                heuristic[i, j] = 0

    best_route = None
    best_cost = float('inf')
    best_cost_history = []

    for iteration in range(iterations):
        all_routes = []
        all_costs = []
        for ant in range(num_ants):
            unvisited = patient_ids.copy()
            route = []
            current_index = random.choice(range(n))
            current_patient = patient_ids[current_index]
            route.append(current_patient)
            unvisited.remove(current_patient)
            while unvisited:
                current_idx = patient_ids.index(current_patient)
                probabilities = []
                for next_patient in unvisited:
                    next_idx = patient_ids.index(next_patient)
                    prob = (pheromone[current_idx, next_idx] ** alpha) * (heuristic[current_idx, next_idx] ** beta)
                    probabilities.append(prob)
                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()
                next_patient = random.choices(unvisited, weights=probabilities, k=1)[0]
                route.append(next_patient)
                unvisited.remove(next_patient)
                current_patient = next_patient

            cost = calculate_total_delivery_time(route, room_mapping, meal_mapping, meal_type,
                                                 num_staffs, max_trip_capacity)
            all_routes.append(route)
            all_costs.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_route = route.copy()
        best_cost_history.append(best_cost)

        pheromone *= (1 - evaporation_rate)
        for route, cost in zip(all_routes, all_costs):
            epsilon = 1e-6
            deposit = Q / (cost + epsilon)
            for i in range(n - 1):
                idx1 = patient_ids.index(route[i])
                idx2 = patient_ids.index(route[i + 1])
                pheromone[idx1, idx2] += deposit
                pheromone[idx2, idx1] += deposit

        if best_cost < optimal_threshold:
            print(f"ACO converged after {iteration + 1} iterations.")
            break
    return best_route, best_cost, best_cost_history

results = {}  # Store final best cost and performance history

for meal_type in ['Breakfast', 'Lunch', 'Dinner']:
    print(f"\n--- {meal_type} ---")

    # ACO
    aco_solution, aco_cost, aco_history = ant_colony_optimization(
        patient_ids, room_mapping, meal_mapping, meal_type,
        num_staffs=num_staffs, max_trip_capacity=max_trip_capacity,
        num_ants=50, iterations=100)
    print(f"ACO - Total delivery time for {meal_type}: {aco_cost:.2f} seconds")

    print(f"Delivery plan for {meal_type} (ACO):")
    display_delivery_plan(aco_solution, room_mapping, meal_mapping, meal_type,
                          num_staffs=num_staffs, max_trip_capacity=max_trip_capacity)
