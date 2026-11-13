
import numpy as np

np.random.seed(1)

# Generate simulated weather data
g, days = 4, 200
t = np.arange(days)
temp = np.zeros((g,g,days))

for i in range(g):
    for j in range(g):
        temp[i,j] = 25 + 7*np.sin(2*np.pi*t/120) + np.random.randn(days)

humidity = 65 - 0.7*(temp-25) + np.random.randn(g,g,days)*2
pressure = 1013 - 0.3*(temp-25) + np.random.randn(g,g,days)

# Prepare input and target
X = np.stack([temp, humidity, pressure], axis=-1)[:,:,:-1]
Y = temp[:,:,1:]

split = int(0.8*(days-1))
mean, std = X[:,:,:split].mean((0,1,2)), X[:,:,:split].std((0,1,2))
my, sy = Y[:,:,:split].mean(), Y[:,:,:split].std()

X = (X-mean)/std
Y = (Y-my)/sy
Xtr, Xte = X[:,:,:split], X[:,:,split:]
Ytr, Yte = Y[:,:,:split], Y[:,:,split:]

# Initialize cells
weights = np.random.rand(g,g,4)*0.1
lr, iterations, nw = 0.05, 100, 0.2

def fitness(w, x, y):
    xb = np.c_[x, np.ones(len(x))]
    return np.mean((xb @ w-y)**2)

# Parallel Cellular Optimization
for _ in range(iterations):
    old = weights.copy()
    new = old.copy()

    for i in range(g):
        for j in range(g):
            x, y = Xtr[i,j], Ytr[i,j]
            xb = np.c_[x, np.ones(len(x))]

            error = xb @ old[i,j] - y
            local = old[i,j] - lr*2*(xb.T @ error)/len(y)

            neighbors = [((i-1)%g,j), ((i+1)%g,j),
                         (i,(j-1)%g), (i,(j+1)%g)]
            best = min(neighbors, key=lambda p: fitness(old[p],x,y))

            new[i,j] = (1-nw)*local + nw*old[best]

    weights = new.copy()

# Test the model
pred = np.zeros_like(Yte)

for i in range(g):
    for j in range(g):
        pred[i,j] = np.c_[Xte[i,j], np.ones(len(Xte[i,j]))] @ weights[i,j]

mse = np.mean((pred-Yte)**2)
print("Test MSE:", round(mse,4))
print("Test RMSE:", round(np.sqrt(mse)*sy,2), "°C")

# Next-day temperature forecast
forecast = np.zeros((g,g))

for i in range(g):
    for j in range(g):
        forecast[i,j] = (np.append(X[i,j,-1],1) @ weights[i,j])*sy + my

print("\nNext Day Temperature Forecast:")
print(np.round(forecast,2))

