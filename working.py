from os.path import os

from cstr import ContinuouslyStirredTankReactor, pca, accuracy_fault_no_fault
import matplotlib.pyplot as plt
import numpy as np


path = os.path.join(os.path.dirname(__file__), 'results/')
cstr = ContinuouslyStirredTankReactor(path)
cstr.process_all(k=2)
cstr.process_all(k=3)


