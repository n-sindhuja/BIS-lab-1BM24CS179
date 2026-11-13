import numpy as np

# Parameters
N = 20
T = 100
LB, UB = 0, 40
Isc, Voc = 8, 40

# Current function
def current(V):
    return Isc * (1 - (V / Voc)**8)

# Fitness function (minimization)
def fitness(V):
    return -(V * current(V))

# Initialize wolves
wolves = np.random.uniform(LB, UB, N)

# GWO iterations
for t in range(T):

    # Sort wolves according to fitness
    fit = np.array([fitness(x) for x in wolves])
    idx = np.argsort(fit)

    alpha = wolves[idx[0]]
    beta = wolves[idx[1]]
    delta = wolves[idx[2]]

    # Decreasing parameter
    a = 2 - 2 * t / T

    for i in range(N):

        X = wolves[i]

        # Alpha wolf
        A1 = 2 * a * np.random.rand() - a
        C1 = 2 * np.random.rand()
        X1 = alpha - A1 * abs(C1 * alpha - X)

        # Beta wolf
        A2 = 2 * a * np.random.rand() - a
        C2 = 2 * np.random.rand()
        X2 = beta - A2 * abs(C2 * beta - X)

        # Delta wolf
        A3 = 2 * a * np.random.rand() - a
        C3 = 2 * np.random.rand()
        X3 = delta - A3 * abs(C3 * delta - X)

        # Update wolf position
        wolves[i] = np.clip((X1 + X2 + X3) / 3, LB, UB)

# Find optimal solution
fit = np.array([fitness(x) for x in wolves])
best_idx = np.argmin(fit)

opt_V = wolves[best_idx]
opt_C = current(opt_V)
max_P = opt_V * opt_C

# Display results
print("Optimal Voltage:", round(opt_V, 4), "V")
print("Optimal Current:", round(opt_C, 4), "A")
print("Maximum Power:", round(max_P, 4), "W")

