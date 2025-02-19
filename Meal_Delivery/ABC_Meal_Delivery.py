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
# 3. Artificial Bee Colony (ABC)
# -----------------------

def artificial_bee_colony(patient_ids, room_mapping, meal_mapping, meal_type,
                          num_staffs=5, max_trip_capacity=10,
                          colony_size=50, limit=100, max_cycles=1000):
    """
    Implements the Artificial Bee Colony algorithm.

    Parameters:
      - colony_size: number of food sources (candidate solutions)
      - limit: maximum number of cycles without improvement before reinitializing a solution (scout phase)
      - max_cycles: maximum number of iterations (cycles)

    Returns the best solution found, its cost, and the cost history.
    """
    population = [random.sample(patient_ids, len(patient_ids)) for _ in range(colony_size)]
    costs = [calculate_total_delivery_time(sol, room_mapping, meal_mapping, meal_type, num_staffs, max_trip_capacity)
             for sol in population]
    trials = [0] * colony_size

    best_index = min(range(colony_size), key=lambda i: costs[i])
    best_solution = population[best_index].copy()
    best_cost = costs[best_index]
    best_cost_history = [best_cost]

    cycle = 0
    while cycle < max_cycles:
        # Employed Bees Phase
        for i in range(colony_size):
            candidate = population[i].copy()
            idx1, idx2 = random.sample(range(len(candidate)), 2)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            candidate_cost = calculate_total_delivery_time(candidate, room_mapping, meal_mapping, meal_type,
                                                           num_staffs, max_trip_capacity)
            if candidate_cost < costs[i]:
                population[i] = candidate
                costs[i] = candidate_cost
                trials[i] = 0
            else:
                trials[i] += 1

        # Onlooker Bees Phase: select candidates based on fitness (inverse cost)
        fitness = [1 / (c + 1e-6) for c in costs]
        sum_fitness = sum(fitness)
        probs = [f / sum_fitness for f in fitness]
        for _ in range(colony_size):
            i = random.choices(range(colony_size), weights=probs, k=1)[0]
            candidate = population[i].copy()
            idx1, idx2 = random.sample(range(len(candidate)), 2)
            candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
            candidate_cost = calculate_total_delivery_time(candidate, room_mapping, meal_mapping, meal_type,
                                                           num_staffs, max_trip_capacity)
            if candidate_cost < costs[i]:
                population[i] = candidate
                costs[i] = candidate_cost
                trials[i] = 0
            else:
                trials[i] += 1

        # Scout Bees Phase: Reinitialize solutions that haven't improved for 'limit' cycles.
        for i in range(colony_size):
            if trials[i] >= limit:
                population[i] = random.sample(patient_ids, len(patient_ids))
                costs[i] = calculate_total_delivery_time(population[i], room_mapping, meal_mapping, meal_type,
                                                         num_staffs, max_trip_capacity)
                trials[i] = 0

        cycle_best_index = min(range(colony_size), key=lambda i: costs[i])
        if costs[cycle_best_index] < best_cost:
            best_cost = costs[cycle_best_index]
            best_solution = population[cycle_best_index].copy()

        best_cost_history.append(best_cost)
        cycle += 1

        if best_cost < optimal_threshold:
            print(f"ABC converged after {cycle} cycles.")
            break

    return best_solution, best_cost, best_cost_history

results = {}  # Store final best cost and performance history

for meal_type in ['Breakfast', 'Lunch', 'Dinner']:
    print(f"\n--- {meal_type} ---")
    # ABC
    abc_solution, abc_cost, abc_history = artificial_bee_colony(
        patient_ids, room_mapping, meal_mapping, meal_type,
        num_staffs=num_staffs, max_trip_capacity=max_trip_capacity,
        colony_size=50, limit=100, max_cycles=1000)
    print(f"ABC - Total delivery time for {meal_type}: {abc_cost:.2f} seconds")

    print(f"Delivery plan for {meal_type} (ABC):")
    display_delivery_plan(abc_solution, room_mapping, meal_mapping, meal_type,
                          num_staffs=num_staffs, max_trip_capacity=max_trip_capacity)
