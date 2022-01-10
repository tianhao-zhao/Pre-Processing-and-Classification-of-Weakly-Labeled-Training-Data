# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 07:43:44 2021

@author: tianh
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt




fig, ax = plt.subplots()
im = ax.imshow(results)

# We want to show all ticks...
ax.set_xticks(np.arange(len(C)))
ax.set_yticks(np.arange(len(gamma)))
# ... and label them with the respective list entries
ax.set_xticklabels(C)
ax.set_yticklabels(gamma)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.

plt.colorbar(im)
ax.set_title("Cross-Validation Results")
ax.set_xlabel("SVM rbf C")
ax.set_ylabel("SVM rbf gamma")
fig.tight_layout()
plt.show()