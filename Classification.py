from Perceptron import Perceptron

class PerceptronClassification:
    def predict(self, test_data, bias, weight):
        for row in test_data:
            activation = self.compute_activation(row, weight, bias)

    def compute_accuracy(self, test_data, bias, weight)