import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Niloofar.SOM import SOM
import seaborn as sns

dataset = pd.read_csv('summery_processed.csv')
dataset = dataset.sample(frac=1).reset_index(drop=True)

dataset = np.array(dataset)

normalize = lambda data: (data - data.mean()) / (data.std())

x_train = dataset[:, 6:-4].astype(np.float)
x_train = normalize(x_train)
y_train = dataset[:, -1].astype(np.float)

model = SOM(iterations=0,
            learning_rate=0.0,
            learning_rate_decay_type="linear",
            radius=1,
            radius_decay_type="linear",
            neighbourhood_dim="2D",
            neighbourhood_shape="circle",
            neighbourhood_strength_func="exponential",
            constant_k=0.2,
            winner_selection_type="neuron activation",
            neurons=64)

updated_weight = model.fit(x_train)
aaa, dictionary = model.clustering(x_train)

model.features_heatmap(x_train)

# hitmap = np.zeros((64))
# for key, values in dictionary.items():
#     hitmap[key] = len(values)
#
# sns.heatmap(
#     hitmap.reshape((8, 8)), annot=True, center=0
# )
# plt.show()
# print(model.purity(y_train, aaa))
#
