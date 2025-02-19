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
# 5. Particle Swarm Optimization (PSO) for Permutations
# -----------------------

def pso_optimization(patient_ids, room_mapping, meal_mapping, meal_type,
                     num_staffs=5, max_trip_capacity=10,
                     swarm_size=50, max_iter=1000, c1=1.5, c2=1.5):
    """
    A simplified discrete PSO variant for permutation problems.
    Each particle is a permutation (chromosome).
    For each particle, we try to move it toward its personal best and the global best
    by swapping elements that differ from the best solutions.

    c1 and c2 are cognitive and social coefficients that determine the probability
    of swapping toward the personal best and global best respectively.
    """
    n = len(patient_ids)
    # Initialize swarm with random permutations.
    swarm = [random.sample(patient_ids, n) for _ in range(swarm_size)]
    pbest = swarm.copy()
    pbest_costs = [calculate_total_delivery_time(sol, room_mapping, meal_mapping, meal_type,
                                                 num_staffs, max_trip_capacity) for sol in swarm]
    gbest = pbest[pbest_costs.index(min(pbest_costs))].copy()
    gbest_cost = min(pbest_costs)
    best_cost_history = [gbest_cost]

    for iter in range(max_iter):
        for i in range(swarm_size):
            particle = swarm[i].copy()

            # Move toward personal best:
            for j in range(n):
                if particle[j] != pbest[i][j]:
                    # With probability proportional to c1, swap element j with the one that matches pbest
                    if random.random() < c1 / n:
                        # Find the index of the desired element in particle
                        target = pbest[i][j]
                        idx_target = particle.index(target)
                        # Swap positions j and idx_target
                        particle[j], particle[idx_target] = particle[idx_target], particle[j]

            # Move toward global best:
            for j in range(n):
                if particle[j] != gbest[j]:
                    if random.random() < c2 / n:
                        target = gbest[j]
                        idx_target = particle.index(target)
                        particle[j], particle[idx_target] = particle[idx_target], particle[j]

            # Evaluate new particle
            cost = calculate_total_delivery_time(particle, room_mapping, meal_mapping, meal_type,
                                                 num_staffs, max_trip_capacity)
            # Update personal best if improved
            if cost < pbest_costs[i]:
                pbest[i] = particle.copy()
                pbest_costs[i] = cost
            swarm[i] = particle.copy()

        # Update global best
        current_best_index = pbest_costs.index(min(pbest_costs))
        if pbest_costs[current_best_index] < gbest_cost:
            gbest_cost = pbest_costs[current_best_index]
            gbest = pbest[current_best_index].copy()
        best_cost_history.append(gbest_cost)

        if gbest_cost < optimal_threshold:
            print(f"PSO converged after {iter + 1} iterations.")
            break

    return gbest, gbest_cost, best_cost_history

results = {}  # Store final best cost and performance history

for meal_type in ['Breakfast', 'Lunch', 'Dinner']:
    print(f"\n--- {meal_type} ---")

    # PSO
    pso_solution, pso_cost, pso_history = pso_optimization(
        patient_ids, room_mapping, meal_mapping, meal_type,
        num_staffs=num_staffs, max_trip_capacity=max_trip_capacity,
        swarm_size=50, max_iter=1000, c1=1.5, c2=1.5)
    print(f"PSO - Total delivery time for {meal_type}: {pso_cost:.2f} seconds")

    print(f"Delivery plan for {meal_type} (PSO):")
    display_delivery_plan(pso_solution, room_mapping, meal_mapping, meal_type,
                          num_staffs=num_staffs, max_trip_capacity=max_trip_capacity)
