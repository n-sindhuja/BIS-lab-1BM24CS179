# Algorithm: Ant Colony Optimization (ACO)
# Problem: Traveling Salesman Problem (TSP)

import numpy as np
import random

# City coordinates
cities = np.array([[0, 0], [2, 3], [5, 2], [6, 6], [8, 3]])
n = len(cities)

# Calculate distance between cities
dist = np.linalg.norm(cities[:, None] - cities, axis=2)

# Initialize pheromone and visibility
pheromone = np.ones((n, n))
visibility = 1 / (dist + np.eye(n))

# ACO parameters
ants, iterations = 5, 100
alpha, beta, rho, Q = 1, 2, 0.5, 100

best_tour, best_dist = None, float('inf')

# ACO Algorithm
for _ in range(iterations):
    tours = []

    for _ in range(ants):
        start = random.randrange(n)
        tour = [start]
        unvisited = set(range(n)) - {start}

        # Construct tour
        while unvisited:
            curr = tour[-1]
            cities_left = list(unvisited)

            weights = [
                (pheromone[curr][j] ** alpha) *
                (visibility[curr][j] ** beta)
                for j in cities_left
            ]

            nxt = random.choices(cities_left, weights=weights)[0]
            tour.append(nxt)
            unvisited.remove(nxt)

        # Return to starting city
        tour.append(start)

        # Calculate tour length
        length = sum(
            dist[tour[i], tour[i + 1]]
            for i in range(n)
        )

        if length < best_dist:
            best_tour, best_dist = tour.copy(), length

        tours.append((tour, length))

    # Pheromone evaporation
    pheromone *= (1 - rho)

    # Pheromone deposition
    for tour, length in tours:
        for i in range(n):
            a, b = tour[i], tour[i + 1]
            pheromone[a][b] += Q / length
            pheromone[b][a] += Q / length

# Display results
print("Best route:", [c + 1 for c in best_tour])
print("Minimum distance:", round(best_dist, 2))

