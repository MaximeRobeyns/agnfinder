#!/usr/bin/env python3.9

import numpy as np
import matplotlib.pyplot as plt

xs = np.linspace(0.1, 2, 100)
# log of some exp dist or something idk looks alright...
concave = np.log(np.exp(-(xs-0.75)**2))

plt.figure(figsize=(5, 2), dpi=200)
plt.plot(xs, concave, linewidth=2)
r = [10, -10]
plt.plot([xs[i] for i in r], [concave[i] for i in r], c='g', linewidth=2)
plt.tight_layout()
plt.savefig("jensens-inequality.svg")
