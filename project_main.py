import cmath
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib import rc # imported for showing the animation on colaboratory
from IPython.display import HTML # imported for showing the animation on colaboratory
import matplotlib as mpl
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["animation.embed_limit"] = 2**128
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arrow