import regression
import pandas as pd  
import numpy as np
import matplotlib.pyplot  as plt

g = regression.SGDRegressor()
data = pd.read_excel('data.xlsx')
Y = data['CONSOMMATION']
X = data['REVENU']

g.fit(X,Y,np.array([12,1]),learning_rate=0.0001,max_iter=100)



params = g.params
print(params)



# Plot scatter plot
one = np.ones(80,dtype=int).reshape(80,1)
y =  np.vstack((one.T, X)).T @ params

plt.scatter(X.values, Y.values.reshape(80,1), label='Data')
# Plot lie

plt.plot(X.values, y, color='red', label='Line of Best Fit')
# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot with Line of Best Fit')
plt.legend()

# Show plot
plt.grid(True)
plt.show()

