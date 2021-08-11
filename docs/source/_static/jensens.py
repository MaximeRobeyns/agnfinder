#!/usr/bin/env python3.9

import numpy as np
import matplotlib.pyplot as plt

xs = np.linspace(0.1, 2, 100)
convex = (xs-0.75)**2 + 0.1

plt.figure(figsize=(5, 2), dpi=200)
plt.plot(xs, convex, linewidth=2)
r = [10, -10]
exs = [0.1, 2]
eys = [0.1, 1.4]
plt.plot(exs, eys, c='g', linewidth=2)
plt.tight_layout()
plt.savefig("jensens-inequality.svg")


