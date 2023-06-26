import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

from IPython import embed

sns.set()

N = 6

f = open("figs/heatmap_method.pkl", "rb")
R = pickle.load(f)
f.close()

R = pd.DataFrame(R, columns=np.arange(1,N+1), index=np.arange(1,N+1))

fig = plt.figure()
# cmap='RdBu_r'
sns_plot = sns.heatmap(R, cmap='YlGnBu', xticklabels=1, yticklabels=1)

sns_plot.tick_params(labelsize=15, direction='in')

cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=15, direction='in', top='off', bottom='off', left='off', right='off')

fig.savefig("figs/heatmap.png", bbox_inches='tight')