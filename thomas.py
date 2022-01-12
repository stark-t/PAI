import numpy as np
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
pts1 = np.array([[2, 2], [2, 3],[4, 3], [4, 2]])
pts2 = np.array([[2, 3], [2, 5],[5, 3], [4, 2]])
p1 = Polygon(pts1, closed=False, color="red", alpha=0.7)
p2 = Polygon(pts2, closed=False, color="blue", alpha=0.7)
ax = plt.gca()
ax.add_patch(p1)
ax.add_patch(p2)
ax.set_xlim(1, 7)
ax.set_ylim(1, 8)
plt.show()

