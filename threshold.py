import itertools

class Perceptron:
    def __init__(self, num_inputs, zero=0, one=1):
        self._weights = [0] * (num_inputs + 1)
        self._zero = zero
        self._one = one

    def train(self, data):
        ''' Train on a set of (input, label). '''
        label_error = True
        while label_error:
            label_error = False
            for inputs, actual_label in data:
                label = self.label(inputs)
                if label == actual_label: continue
                # We made an error in labelling: keep looping
                label_error = True
                if label == self._zero:
                    # We got a 0 but should have got a 1: increase weights
                    self._increase_weights(inputs)
                else:
                    # We got a 1 but should have got a 0: decrease weights
                    self._decrease_weights(inputs)

    def label(self, inputs):
        ''' Label inputs as either zero or one. '''
        # We must prepend a -1 for a bias number
        inputs = itertools.chain(self._gen_first_input(), inputs)
        label_value = 0
        for value, weight in zip(inputs, self._weights):
            label_value += weight * value
        return self._activation(label_value)

    def _activation(self, value):
        ''' Activation function for this perceptron. '''
        return self._one if value >= 0 else self._zero

    def _increase_weights(self, inputs):
        inputs = itertools.chain(self._gen_first_input(), inputs)
        for index, value in enumerate(inputs):
            self._weights[index] += value

    def _decrease_weights(self, inputs):
        inputs = itertools.chain(self._gen_first_input(), inputs)
        for index, value in enumerate(inputs):
            self._weights[index] -= value

    def _gen_first_input(self):
        yield -1


def read_data(inputs_file, labels_file):
    with open(inputs_file) as in_f:
        with open(labels_file) as l_f:
            inputs = (list(map(int, line.split(','))) for line in in_f)
            labels = (int(line) for line in l_f)
            return list(zip(inputs, labels))


def read_labels(filename):
    with open(filename) as f:
        return [int(line) for line in f]


def score(perceptron, data):
    correct = 0
    for inputs, label in data:
        if perceptron.label(inputs) == label:
            correct += 1
    return correct / len(data)


def main():
    # Train our perceptron
    train_data = read_data('trainData.csv', 'trainLabels.csv')

    perceptron = Perceptron(len(train_data[0][0]), zero=5, one=6)
    perceptron.train(train_data)
    accuracy = score(perceptron, train_data)
    # Perceptron training algorithm only terminates when perfect
    print('Training accuracy (should be 1): {}'.format(accuracy))

    # Test the perceptron
    test_data = read_data('testData.csv', 'testLabels.csv')
    accuracy = score(perceptron, test_data)
    print('Testing accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
