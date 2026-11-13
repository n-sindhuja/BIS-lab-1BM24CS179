# Algorithm: Particle Swarm Optimization (PSO)
# Problem: Air Pollution Weight Optimization

import numpy as np
import matplotlib.pyplot as plt

# Generate simulated pollution data: PM2.5, PM10, NO2
np.random.seed(1)
X = np.random.rand(50, 3)
y = 100 * (X @ np.array([0.5, 0.3, 0.2]))

# Initialize particles
n = 20
pos = np.random.rand(n, 3)
pos /= pos.sum(axis=1, keepdims=True)

vel = np.zeros((n, 3))

# Initialize personal best
pbest = pos.copy()

# Fitness function
fitness = lambda p: np.mean((100 * X @ p - y) ** 2)

scores = np.array([fitness(p) for p in pos])

# Initialize global best
gbest = pbest[np.argmin(scores)].copy()

history = []

# PSO Algorithm
for _ in range(50):

    # Update velocity
    vel = (
        0.7 * vel
        + 1.5 * np.random.rand(n, 3) * (pbest - pos)
        + 1.5 * np.random.rand(n, 3) * (gbest - pos)
    )

    # Update position
    pos = np.clip(pos + vel, 0, 1)
    pos /= pos.sum(axis=1, keepdims=True)

    # Evaluate new positions
    new_scores = np.array([fitness(p) for p in pos])

    # Update personal best
    better = new_scores < scores
    pbest[better] = pos[better]
    scores[better] = new_scores[better]

    # Update global best
    gbest = pbest[np.argmin(scores)].copy()

    history.append(min(scores))

# Display results
print("Optimized weights (PM2.5, PM10, NO2):", gbest)
print("Minimum MSE:", min(scores))

# Plot convergence
plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("Best MSE")
plt.title("PSO Air Pollution Optimization")
plt.show()

