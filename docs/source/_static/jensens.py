#!/usr/bin/env python3.9

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)


def concave(xs: np.ndarray) -> np.ndarray:
    # log of some exp dist or something idk looks alright...
    return np.log(np.exp(-(xs-1)**2)) * 5


xs = np.linspace(0.1, 1, 100)
d_xs = np.random.beta(2, 2, 10)
mean_xs = np.mean(d_xs)
f_xs = concave(xs)

plt.figure(figsize=(5, 2), dpi=200)
plt.plot(xs, f_xs, linewidth=2, c='#252224')

r = '#f25d41'
# f(E[x])
plt.scatter(mean_xs, [-2.29], c=r, marker='s')
plt.vlines(mean_xs, [-2.3], concave(mean_xs), color=r, linewidth=1.5)
ll = plt.hlines(concave(mean_xs), [0], mean_xs, color=r, linewidth=1.5)
ll.set_label(r'$\varphi(\mathbb{E}[x])$')
plt.scatter([0.01], concave(mean_xs), c=r, marker='s')

# E[f(x)]
lb = '#94c1d8'
b = '#076ea1'
plt.scatter(d_xs, [-2.3 for _ in d_xs], c=lb, marker='o', s=10)
plt.vlines(d_xs, [-2.3 for _ in d_xs], concave(d_xs), color=lb, linewidth=0.75)
plt.hlines(concave(d_xs), [0 for _ in d_xs], d_xs, color=lb, linewidth=0.75)
l = plt.hlines(np.mean(concave(d_xs)), 0, 0.58, color=b, linewidth=1.5)
l.set_label(r'$\mathbb{E}[\varphi(x)]$')
plt.scatter(np.zeros_like(d_xs), concave(d_xs), c=lb, marker='o', s=10)
plt.scatter([0.01], np.mean(concave(d_xs)), c=b, marker='s')

plt.legend(loc='lower left', fontsize=12)

plt.xlim(0, 1)
plt.ylim(bottom=-2.3)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig("jensens-inequality.svg")
