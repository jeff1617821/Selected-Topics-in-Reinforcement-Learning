import pandas as pd
import numpy as np
data = pd.read_csv('output.csv', sep = '\t', header = None)
import matplotlib.pyplot as plt
x = np.linspace(0, len(data.T) * 1000, len(data.T))
plt.plot(x, data.T)
plt.xlabel("episode")
plt.ylabel("score", rotation = 0, va = 'top')
plt.show()
