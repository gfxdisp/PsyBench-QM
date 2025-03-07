from metrics import cvvdp_met
from pyst.ContrastMatching import *
from pyst.ContrastMasking import *
from pyst.GaborPatch import *
import matplotlib.pyplot as plt
import numpy as np


""" First thing we want to generate the necessary images for the grid
which tests to consider:
* Contrast Detection - Spatial Freq 
* Contrast Masking - Phase Coherent
* Contrast Matching - Both tests 
"""

# Contrast Detection
T = GaborPatchSpatialFreqAchTest(width=512, height=512, device='cpu')

y_index = 11
x_index = 0


index = y_index*20+x_index
T.save_as_png(index, 'figs/detection/teaser_test.png', 'figs/detection/teaser_ref.png')

# Contrast masking
T = ContrastMakingCoherentAchTest(width=512, height=512, device='cpu')
y_index = 13
x_index = -3


index = y_index*20+x_index
T.save_as_png(index, 'figs/masking/teaser_test.png', 'figs/masking/teaser_ref.png')







