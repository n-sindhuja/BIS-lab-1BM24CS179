# Algorithm: Cuckoo Search (CS)
# Problem: Knapsack Optimization

import numpy as np

# Knapsack data
weights = np.array([2, 3, 4, 5])
profits = np.array([20, 30, 40, 50])
capacity = 10

# Cuckoo Search parameters
n = 10
Pa = 0.25
MaxIter = 100
m = len(weights)

# Fitness function
def fitness(x):
    w = np.sum(x * weights)
    p = np.sum(x * profits)

    return p if w <= capacity else 0

# Generate random nests
nests = np.random.randint(0, 2, (n, m))
scores = np.array([fitness(x) for x in nests])

best = nests[np.argmax(scores)].copy()

# Cuckoo Search Algorithm
for iteration in range(MaxIter):

    for i in range(n):

        # Generate new solution using random step
        step = np.random.normal(0, 1, m)
        new = (nests[i] + step > 0.5).astype(int)

        # Compare with a randomly selected nest
        j = np.random.randint(n)

        if fitness(new) > scores[j]:
            nests[j] = new
            scores[j] = fitness(new)

    # Abandon worst nests
    count = max(1, int(Pa * n))
    worst = np.argsort(scores)[:count]

    for j in worst:
        nests[j] = np.random.randint(0, 2, m)
        scores[j] = fitness(nests[j])

    # Update best solution
    if max(scores) > fitness(best):
        best = nests[np.argmax(scores)].copy()

# Display results
print("Best Solution:", best)
print("Selected Items:", np.where(best == 1)[0] + 1)
print("Total Weight:", np.sum(best * weights))
print("Maximum Profit:", fitness(best))
