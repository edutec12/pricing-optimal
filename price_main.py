import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the horizon
HORIZON = 24
elasticity = 0.2

# Define the objective function for the users
def user_objective(prices, demand):
    return -np.sum(prices * demand + prices * elasticity)

# Define the objective function for the electric service provider
def provider_objective(prices, demand):
    return np.sum(prices * demand)

# Define the constraints
def constraint(demand):
    return np.sum(demand) - 1000

# Define the bounds for the variables
bounds = [(0, None) for _ in range(HORIZON)]

# Define the initial prices and demand
initial_prices = np.ones(HORIZON) * 10
initial_users_demand = np.ones(HORIZON) * 500
initial_provider_demand = np.ones(HORIZON) * 500

# Define the optimization function
def optimize(prices, objective, other_prices, other_demand):
    result = minimize(
        lambda x: -objective(x, initial_users_demand) if objective == user_objective else objective(x, initial_provider_demand),
        prices,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'eq', 'fun': constraint}
    )
    demand = optimize_demand(result.x, other_prices, other_demand) if objective == user_objective else initial_provider_demand
    return result.x, demand

# Define the function to optimize demand given prices and the other agent's demand
def optimize_demand(prices, other_prices, other_demand):
    demand = minimize(
        lambda x: -user_objective(prices, x) if np.array_equal(prices, initial_prices) else -provider_objective(prices, x),
        initial_users_demand if np.array_equal(prices, initial_prices) else initial_provider_demand,
        method='SLSQP',
        bounds=[(0, None) for _ in range(HORIZON)],
        constraints={'type': 'eq', 'fun': constraint}
    ).x
    return demand

# Run the Nash-equilibrium algorithm
max_iter = 100
tolerance = 1e-6
for i in range(max_iter):
    # Update the users' prices and demand based on the provider's prices
    users_prices, users_demand = optimize(initial_prices, user_objective, initial_prices, initial_provider_demand)
    # Update the provider's prices and demand based on the users' prices
    provider_prices, provider_demand = optimize(initial_prices, provider_objective, users_prices, users_demand)
    # Check for convergence
    if np.linalg.norm(users_prices - initial_prices) < tolerance and np.linalg.norm(provider_prices - initial_prices) < tolerance:
        break
    initial_prices = (users_prices + provider_prices) / 2
    initial_users_demand = users_demand
    initial_provider_demand = provider_demand

social_welfare = -user_objective(users_prices, users_demand) + provider_objective(provider_prices, provider_demand)

# Print the results
print("Social welfare:", social_welfare)
print("Users optimal prices:", users_prices)
print("Users optimal demand:", users_demand)
print("Users objective value:", user_objective(users_prices, users_demand))
print("Electric service provider optimal prices:", provider_prices)
print("Electric service provider optimal demand:", provider_demand)
print("Electric service provider objective value:", provider_objective(provider_prices, provider_demand))


# Plot the optimal prices for each agent
plt.plot(users_prices, label='Users')
plt.plot(provider_prices, label='Electric service provider')
plt.title('Optimal prices')
plt.xlabel('Hour')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plot the optimal demand for each agent
plt.plot(users_demand, label='Users')
plt.plot(provider_demand, label='Electric service provider')
plt.title('Optimal demand')
plt.xlabel('Hour')
plt.ylabel('Demand')
plt.legend()
plt.show()
