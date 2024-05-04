import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# Cities distances to each other
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
distances = np.array([
    [0, 2445, 789, 1630, 2145, 97, 1783, 2425, 1548, 2932],
    [2445, 0, 1745, 1377, 357, 2409, 1200, 111, 1235, 340],
    [789, 1745, 0, 940, 1445, 758, 1121, 1725, 800, 1857],
    [1630, 1377, 940, 0, 1012, 1544, 197, 1302, 225, 1597],
    [2145, 357, 1445, 1012, 0, 2075, 980, 304, 887, 621],
    [97, 2409, 758, 1544, 2075, 0, 1682, 2352, 1463, 2877],
    [1783, 1200, 1121, 197, 980, 1682, 0, 1150, 252, 1489],
    [2425, 111, 1725, 1302, 304, 2352, 1150, 0, 1172, 417],
    [1548, 1235, 800, 225, 887, 1463, 252, 1172, 0, 1435],
    [2932, 340, 1857, 1597, 621, 2877, 1489, 417, 1435, 0]
])

# Applying MDS
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=6)
cities_2d = mds.fit_transform(distances)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(16, 8))

# Heatmap of the original distance matrix
cax = ax[0].matshow(distances, cmap='coolwarm')
fig.colorbar(cax, ax=ax[0])
ax[0].set_title('Original Distance Matrix')
ax[0].set_xticks(np.arange(len(cities)))
ax[0].set_yticks(np.arange(len(cities)))
ax[0].set_xticklabels(cities, rotation=90)
ax[0].set_yticklabels(cities)

# MDS plot
ax[1].scatter(cities_2d[:, 0], cities_2d[:, 1], color='orange', s=100, edgecolor='black')
for i, city in enumerate(cities):
    ax[1].annotate(city, (cities_2d[i, 0], cities_2d[i, 1]), textcoords="offset points", xytext=(0,10), ha='center')
ax[1].set_title('US Cities Visualized by MDS')
ax[1].set_xlabel('MDS Dimension 1')
ax[1].set_ylabel('MDS Dimension 2')
ax[1].grid(True)

plt.tight_layout()
plt.show()
