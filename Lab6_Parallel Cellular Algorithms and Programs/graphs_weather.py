
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Generate simulated weather data
g, days = 4, 200
temp = np.zeros((g, g, days))

for i in range(g):
    for j in range(g):
        t = np.arange(days)
        temp[i,j] = (25 + 7*np.sin(2*np.pi*t/120)
                     + 0.4*i + 0.3*j + np.random.randn(days))

humidity = 65 - 0.7*(temp-25) + np.random.randn(g,g,days)*2
pressure = 1013 - 0.3*(temp-25) + np.random.randn(g,g,days)

# Prepare input and target data
X = np.stack([temp, humidity, pressure], axis=-1)[:, :, :-1]
Y = temp[:, :, 1:]

# Split training and testing data
split = int(0.8*(days-1))
mean = X[:,:,:split].mean((0,1,2))
std = X[:,:,:split].std((0,1,2))
my, sy = Y[:,:,:split].mean(), Y[:,:,:split].std()

X = (X-mean)/std
Y = (Y-my)/sy

Xtr, Xte = X[:,:,:split], X[:,:,split:]
Ytr, Yte = Y[:,:,:split], Y[:,:,split:]

# Initialize cells and parameters
weights = np.random.rand(g,g,4)*0.1
lr, nw, iterations = 0.05, 0.2, 100

# Fitness function
def fitness(w, x, y):
    xb = np.c_[x, np.ones(len(x))]
    return np.mean((xb @ w-y)**2)

# Update individual cell
def update(i, j, old):
    x, y = Xtr[i,j], Ytr[i,j]
    xb = np.c_[x, np.ones(len(x))]

    # Local optimization
    error = xb @ old[i,j] - y
    gradient = 2*(xb.T @ error)/len(y)
    local = old[i,j] - lr*gradient

    # Identify neighbours
    neighbors = [((i-1)%g,j), ((i+1)%g,j),
                 (i,(j-1)%g), (i,(j+1)%g)]

    # Find best neighbour
    best = min(neighbors,
               key=lambda p: fitness(old[p], x, y))

    # Update using neighbour information
    return (1-nw)*local + nw*old[best]

# Parallel Cellular Optimization
for _ in range(iterations):
    old = weights.copy()
    new = old.copy()

    for i in range(g):
        for j in range(g):
            new[i,j] = update(i,j,old)

    weights = new.copy()

# Evaluate test predictions
pred = np.zeros_like(Yte)

for i in range(g):
    for j in range(g):
        xb = np.c_[Xte[i,j], np.ones(len(Xte[i,j]))]
        pred[i,j] = xb @ weights[i,j]

# Calculate prediction error
mse = np.mean((pred-Yte)**2)
rmse = np.sqrt(mse)*sy

print("Parallel Cellular Weather Forecasting")
print("--------------------------------------")
print("Grid Size:", g, "x", g)
print("Iterations:", iterations)
print("Test MSE:", round(mse,4))
print("Test RMSE:", round(rmse,2), "°C")

# Predict next-day temperature
future = X[:,:,-1]
forecast = np.zeros((g,g))

for i in range(g):
    for j in range(g):
        xb = np.append(future[i,j],1)
        forecast[i,j] = (xb @ weights[i,j])*sy + my

print("\nNext Day Temperature Forecast:")
print(np.round(forecast,2))

# Plot temperature forecast
plt.imshow(forecast, cmap="coolwarm")
plt.colorbar(label="Temperature (°C)")
plt.title("Next Day Weather Forecast")
plt.xlabel("Grid Column")
plt.ylabel("Grid Row")
plt.show()

# Plot actual vs predicted temperature
actual = Yte[0,0]*sy + my
predicted = pred[0,0]*sy + my

plt.plot(actual, label="Actual")
plt.plot(predicted, label="Predicted")
plt.xlabel("Test Day")
plt.ylabel("Temperature (°C)")
plt.title("Actual vs Predicted Weather")
plt.legend()
plt.show()
