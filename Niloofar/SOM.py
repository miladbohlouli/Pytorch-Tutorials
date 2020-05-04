import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

class SOM:

    def __init__(self, iterations, learning_rate, learning_rate_decay_type,
                 radius, radius_decay_type, neighbourhood_dim, neighbourhood_shape,
                 neighbourhood_strength_func, constant_k, winner_selection_type, neurons):

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.learning_rate_decay_type = learning_rate_decay_type
        self.radius = radius
        self.radius_decay_type = radius_decay_type
        self.neighbourhood_dim = neighbourhood_dim
        self.neighbourhood_shape = neighbourhood_shape
        self.neighbourhood_strength_func = neighbourhood_strength_func
        self.constant_k = constant_k
        self.winner_selection_type = winner_selection_type
        self.neurons = neurons
        self.weight_matrix = [None]

    def updating_radius(self, epoch):

        if self.radius <= 1:
            return self.radius

        else:

            if self.radius_decay_type == "linear":
                self.radius = self.radius * (1 - epoch / self.iterations)

            elif self.radius_decay_type == "exponential":
                self.radius = self.radius * (np.exp(-epoch / self.iterations))

        return self.radius

    def updating_learning_rate_decay(self, epoch):
        if self.learning_rate_decay_type == "linear":
            self.learning_rate = self.learning_rate * (1 - epoch / self.iterations)

        elif self.learning_rate_decay_type == "exponential":
            self.learning_rate = self.learning_rate * (np.exp(-epoch / self.iterations))

        elif self.learning_rate_decay_type == "others":
            self.learning_rate = 2 / (3 + epoch)

        return self.learning_rate

    def selecting_winner(self, sample):

        winners = None
        if self.winner_selection_type == "neuron activation":
            winners = np.argmax(np.matmul(self.weight_matrix, sample))

        elif self.winner_selection_type == "distance to input vector":
            winners = np.argmin(np.linalg.norm(sample - self.weight_matrix, axis=1))
        return winners

    def finding_neighbourhood(self, winner_index):
        neighbourhood_idx = []

        squared_neighbourhood_row = []
        squared_neighbourhood_col = []
        circle_neighbourhood_row = []
        circle_neighbourhood_col = []

        winner_row_idx = int((winner_index / np.sqrt(self.neurons)))
        winner_col_idx = int((winner_index % np.sqrt(self.neurons)))

        if self.neighbourhood_dim == "1D" and self.neighbourhood_shape == "linear":
            for i in range(max(0, int(winner_index - self.radius)), min(int(winner_index + self.radius + 1), (int(self.neurons)))):
                neighbourhood_idx.append(i)

        elif self.neighbourhood_dim == "2D":

            if self.neighbourhood_shape == "square" or self.neighbourhood_shape == "circle":

                for i in range(max(0, int(winner_row_idx - self.radius)),
                               min(int(winner_row_idx + self.radius + 1), int(np.sqrt(self.neurons)))):
                    for j in range(max(0, int(winner_col_idx - self.radius)),
                                   min(int(winner_col_idx + self.radius + 1), int(np.sqrt(self.neurons)))):

                        if (i - winner_row_idx) ** 2 + (j - winner_col_idx) ** 2 <= self.radius ** 2:
                            circle_neighbourhood_row.append(i)
                            circle_neighbourhood_col.append(j)

                        if self.neighbourhood_shape == "square":
                            squared_neighbourhood_row.append(i)
                            squared_neighbourhood_col.append(j)

        for i in squared_neighbourhood_row:
            neighbourhood_idx.append(i * np.sqrt(self.neurons) + winner_col_idx)

        for i in squared_neighbourhood_col:
            neighbourhood_idx.append(winner_row_idx * np.sqrt(self.neurons) + i)

        for i in circle_neighbourhood_row:
            neighbourhood_idx.append(i * np.sqrt(self.neurons) + winner_col_idx)

        for i in circle_neighbourhood_col:
            neighbourhood_idx.append(winner_row_idx * np.sqrt(self.neurons) + i)

        neighbourhood_idx = np.array(neighbourhood_idx)
        return np.unique(neighbourhood_idx).astype("int32")

    def neighbourhood_strength(self, winner_indx, neighourhood_indices, epoch):
        ns = 0
        dist = np.linalg.norm(self.weight_matrix[winner_indx] - self.weight_matrix[neighourhood_indices])

        if self.neighbourhood_strength_func == "linear":
            pass

        elif self.neighbourhood_strength_func == "gaussian":
            ns = np.exp(-np.power(dist, 2) / (2 * np.power(self.updating_radius(epoch)),2))

        elif self.neighbourhood_strength_func == "exponential":
            ns = np.exp(-self.constant_k * np.power(dist, 2))

        return ns

    def fit(self, samples):
        self.weight_matrix = np.random.normal(0, 1, (self.neurons, samples.shape[1]))
        # print("initialized weights:\n",self.weight_matrix)
        for i in range(self.iterations):
            for j in range(len(samples)):
                # print("samples shape:", samples[j,:].shape)
                # print("weight shape", self.weight_matrix.shape)
                winner_idx = self.selecting_winner(samples[j, :])
                neighbourhood_idx = self.finding_neighbourhood(winner_idx)
                for a in neighbourhood_idx:
                    neighborhood_strength = self.neighbourhood_strength(winner_idx, a, i)
                    self.weight_matrix[a, :] += self.learning_rate * neighborhood_strength * (samples[j, :] - self.weight_matrix[a, :])
            self.updating_learning_rate_decay(i)
            self.updating_radius(i)
        return self.weight_matrix

    def clustering(self, samples):
        winner_dic = {}

        for i in range(len(samples)):
            winner_idx = self.selecting_winner(samples[i])
            if winner_idx in winner_dic:
                winner_dic[winner_idx].append(i)

            else:
                samples_in_cluster = []
                samples_in_cluster.append(i)
                winner_dic[winner_idx] = samples_in_cluster

        predictions = np.zeros(samples.shape[0])
        for i, values in enumerate(winner_dic.values()):
            predictions[values] = i
        return predictions, winner_dic

    def feature_extraction(self, samples):
        extracted_features = np.zeros((len(samples), self.neurons))
        for i in range(len(samples)):
            # print("distance:", np.linalg.norm(samples[i] - self.weight_matrix, axis=1))
            extracted_features[i,:] = 1 / (np.linalg.norm(samples[i] - self.weight_matrix, axis=1))
        return extracted_features

    def features_heatmap(self, samples, feature_names=None, figsize=[14, 10], num_columns=6):
        """
        :param samples: Your input data
        :param feature_names: The list containing the name of the features, it should have length equal to your
            samples dimension
        :param figsize: A tuple containing figure size of the plot
        :param num_columns: The number of the columns that the subplot will be constructed with
        """

        # checking the integrity of the inputs
        if feature_names is not None:
            assert len(feature_names) == samples.shape[1]
        assert len(figsize) == 2

        _, winner_dic = self.clustering(samples)
        num_d = samples.shape[1]

        num_rows = (num_d // num_columns) // 2 + 1  # number of the rows of the sub plot
        plt.figure(figsize=figsize)
        neurons_matrix = np.zeros((self.neurons, num_d))
        index = 1
        for i in range(num_d):
            if index is 30:
                plt.figure(figsize=figsize)
                index = 1
            for key, val in winner_dic.items():
                neurons_matrix[key, i] = np.mean(samples[val, i], axis=0)

            # This section is for making the subplot
            plt.subplot(num_rows, num_columns, index)
            sns.heatmap(neurons_matrix[:, i].reshape((int(self.neurons ** 0.5), -1)), center=0, cbar=False)
            plt.xticks([]), plt.yticks([])
            plt.title(feature_names[i], size=6) if feature_names is not None else plt.title(f"feature {i}", size=6)
            index += 1

        plt.show()

    def purity(self, y_true, y_prediction):
        assert len(y_true) == len(y_prediction)
        y_true = np.asarray(y_true).squeeze()
        y_prediction = np.asarray(y_prediction).squeeze()
        num_data = len(y_true)

        unique_true = np.unique(y_true)
        unique_prediction = np.unique (y_prediction)

        confusion_matrix = np.zeros((unique_prediction.shape[0], unique_true.shape[0]))
        for i, predicted_label in enumerate(unique_prediction):
            for j, true_label in enumerate(unique_true):
                confusion_matrix[i, j] = np.sum(np.logical_and(y_true == true_label, y_prediction == predicted_label))

        return np.sum(np.max(confusion_matrix, axis=1)) / num_data









        




















