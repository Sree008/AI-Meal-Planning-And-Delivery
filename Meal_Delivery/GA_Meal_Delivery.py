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
# 1. Genetic Algorithm (GA)
# -----------------------

def create_initial_population(pop_size, patient_ids):
    population = []
    for _ in range(pop_size):
        chromosome = random.sample(patient_ids, len(patient_ids))
        population.append(chromosome)
    return population


def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    pointer = 0
    for gene in parent2:
        if gene not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = gene
    return child


def mutation(chromosome, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(chromosome)), 2)
        chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]


def evolve_population(population, room_mapping, meal_mapping, meal_type,
                      mutation_rate=0.1, num_staffs=5, max_trip_capacity=10):
    new_population = []
    population_fitness = [
        (chromosome, calculate_total_delivery_time(chromosome, room_mapping, meal_mapping, meal_type,
                                                   num_staffs, max_trip_capacity))
        for chromosome in population
    ]
    population_fitness.sort(key=lambda x: x[1])
    top_half = [chromosome for chromosome, _ in population_fitness[:len(population) // 2]]
    while len(new_population) < len(population):
        parent1, parent2 = random.sample(top_half, 2)
        child = crossover(parent1, parent2)
        mutation(child, mutation_rate)
        new_population.append(child)
    best_chromosome, best_cost = population_fitness[0]
    return new_population, best_chromosome, best_cost, population_fitness


def genetic_algorithm(patient_ids, room_mapping, meal_mapping, meal_type, pop_size=500,
                      num_staffs=5, max_trip_capacity=10):
    population = create_initial_population(pop_size, patient_ids)
    best_cost_history = []
    best_solution = None
    best_cost = float('inf')
    no_improvement = 0
    for generation in range(max_generations):
        population, current_best_solution, current_best_cost, _ = evolve_population(
            population, room_mapping, meal_mapping, meal_type, mutation_rate=0.1,
            num_staffs=num_staffs, max_trip_capacity=max_trip_capacity)
        best_cost_history.append(current_best_cost)
        if current_best_cost < best_cost - improvement_epsilon:
            best_solution = current_best_solution
            best_cost = current_best_cost
            no_improvement = 0
        else:
            no_improvement += 1

        if best_cost < optimal_threshold or no_improvement >= stall_generations:
            print(f"GA converged after {generation + 1} generations.")
            break
    return best_solution, best_cost, best_cost_history

results = {}  # Store final best cost and performance history

for meal_type in ['Breakfast', 'Lunch', 'Dinner']:
    print(f"\n--- {meal_type} ---")

    # GA
    ga_solution, ga_cost, ga_history = genetic_algorithm(
        patient_ids, room_mapping, meal_mapping, meal_type,
        pop_size=500, num_staffs=num_staffs, max_trip_capacity=max_trip_capacity)
    print(f"GA - Total delivery time for {meal_type}: {ga_cost:.2f} seconds")

    print(f"Delivery plan for {meal_type} (GA):")
    display_delivery_plan(ga_solution, room_mapping, meal_mapping, meal_type,
                          num_staffs=num_staffs, max_trip_capacity=max_trip_capacity)
