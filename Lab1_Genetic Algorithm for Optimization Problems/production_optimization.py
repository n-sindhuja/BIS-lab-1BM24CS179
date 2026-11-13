import numpy as np


# Crop data: Rice, Wheat, Maize, Groundnut
profit = np.array([25000, 21000, 19500, 37000])  # ₹ per acre
water = np.array([4000, 2500, 2200, 1800])       # m³ per acre


LAND, WATER_LIMIT = 10, 28000
POP, GEN = 50, 100


# Fitness function
def fitness(x):
    p = np.dot(x, profit)
    w = np.dot(x, water)
    penalty = 100 * max(0, w - WATER_LIMIT)
    return p - penalty


# Initialize population
population = np.random.dirichlet(np.ones(4), POP) * LAND
best = None
best_score = -np.inf
history = []


# Genetic Algorithm
for g in range(GEN):
    scores = np.array([fitness(x) for x in population])


    i = np.argmax(scores)
    if scores[i] > best_score:
        best, best_score = population[i].copy(), scores[i]


    history.append(best_score)


    # Select top individuals
    parents = population[np.argsort(scores)[-10:]]
    new_population = [x.copy() for x in parents]


    while len(new_population) < POP:
        p1, p2 = parents[np.random.randint(10, size=2)]
        child = (p1 + p2) / 2


        if np.random.rand() < 0.2:
            child += np.random.normal(0, 0.5, 4)


        child = np.clip(child, 0, None)
        if child.sum() > LAND:
            child *= LAND / child.sum()


        new_population.append(child)


    population = np.array(new_population)


# Output
crops = ["Rice", "Wheat", "Maize", "Groundnut"]


print("Optimal Crop Allocation:")
for crop, acres in zip(crops, best):
    print(f"{crop}: {acres:.2f} acres")


print(f"\nExpected Profit: ₹{np.dot(best, profit):,.2f}")
print(f"Water Used: {np.dot(best, water):,.0f} m³")


import matplotlib.pyplot as plt
plt.plot(history)
plt.xlabel("Generation")
plt.ylabel("Best Fitness (₹)")
plt.title("Crop Production Optimization")
plt.grid()
plt.show()

