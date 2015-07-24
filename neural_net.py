import math
import random
from collections import namedtuple

class NeuralNode():
    def __init__(self, num_inputs):
        self.weights = [random.random() - 0.5] * (num_inputs + 1)
        # Store these values at each node
        self.activation = None
        self.input = None
    def process(self, inputs):
        self.input = 0
        for w, i in zip(self.weights, inputs):
            self.input += w * i
        try:
            self.activation = sigmoid(self.input)
        except:
            print('What the fuck. The weights are: {}'.format(self.weights))
            print('wtf and the inputs are {}'.format(inputs))
            print('wtf net input: {}'.format(self.input))


def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except:
        if x > 0: return 1
        else: return 0


def sigmoid_prime(x):
    y = sigmoid(x)
    return y * (1 - y)


def sigmoid_to_label(x):
    ''' Convert activation function output to label. '''
    return 5 if x <= 0.5 else 6


def label_to_output(l):
    ''' Convert a label to desired activation function output. '''
    return 0 if l == 5 else 1


class NeuralNetwork():
    def __init__(self, levels, activation_fun, activation_prime):
        ''' The first entry in "levels" is the number of inputs. '''
        self._act = activation_fun
        self._act_prime = activation_prime
        self._levels = []

        for num_inputs, num_nodes in zip(levels, levels[1:]):
            self._levels.append([NeuralNode(num_inputs)
                                 for _ in range(num_nodes)])

    def label(self, inputs):
        ''' Label a set of inputs. '''
        # Append 1 to the inputs for the bias
        inputs = inputs[:]
        inputs.append(1)
        for level in self._levels:
            new_inputs = []
            for node in level:
                node.process(inputs)
                new_inputs.append(node.activation)
            # Append 1 for the bias node
            new_inputs.append(1)
            inputs = new_inputs
        return self._levels[-1][0].activation

    def _learn_example(self, inputs, true_out, rate):
        ''' Example inputs, the true output and the learning rate. '''
        # Our current estimation for the inputs
        net_out = self.label(inputs)

        # Backpropagation of error: Iterate over levels in reverse
        for ln in range(len(self._levels) - 1, -1, -1):
            level = self._levels[ln]
            for i in range(len(level)):
                # Do the ith node on this level
                node = level[i]
                if ln == len(self._levels) - 1:
                    # We are an output node, so we need to compute our delta
                    # a little differently
                    node.delta = (true_out - net_out)
                else:
                    # Get the weight to each child of this node
                    d = (n.weights[i] * n.delta for n in self._levels[ln + 1])
                    node.delta = sum(d)

        # Go forward again to update the weights
        for ln in range(len(self._levels)):
            level = self._levels[ln]
            for node in level:
                # Get the derivative of our activation function
                gp = self._act_prime(node.input)
                if ln > 0:
                    for pi, parent in enumerate(self._levels[ln - 1]):
                        node.weights[pi] += gp * rate * node.delta * parent.activation
                        #node.weights[pi] +=  rate * node.delta * parent.activation
                else:
                    # Top level of hidden nodes
                    for pi, act in enumerate(inputs):
                        #node.weights[pi] += gp * rate * node.delta * act
                        node.weights[pi] +=  rate * node.delta * act
                # Account for bias in every node
                node.weights[-1] += gp * rate * node.delta * 1
                #node.weights[-1] +=  rate * node.delta * 1

    def learn(self, examples, rate, num_passes=1000):
        ''' Learn from examples. '''
        for x in range(num_passes):
            if x % 20 == 0: print('Doing {}'.format(x))
            for data, true_label in examples:
                true_out = label_to_output(true_label)
                self._learn_example(data, true_out, rate)


def read_data(inputs_file, labels_file):
    with open(inputs_file) as in_f:
        with open(labels_file) as l_f:
            inputs = (list(map(int, line.split(','))) for line in in_f)
            labels = (int(line) for line in l_f)
            return list(zip(inputs, labels))


def score(net, data):
    correct = 0
    for inputs, label in data:
        if sigmoid_to_label(net.label(inputs)) == label:
            correct += 1
    return correct / len(data)


def main():
    train_data = read_data('trainData.csv', 'trainLabels.csv')
    #train_data = [([0,0],5), ([1,0],5), ([0,1], 5), ([1,1],6)]
    #for lst, _ in train_data:
    #    lst.insert(0, -1)
    num_inputs = len(train_data[0][0])
    #train_data=train_data[:1]
    for hidden_nodes in range(5, 16):
        net = NeuralNetwork([num_inputs, hidden_nodes, 1], sigmoid, sigmoid_prime)
        net.learn(train_data, .001, 1000)
        print('{} training score: {}'.format(hidden_nodes, score(net, train_data)))

        test_data = read_data('testData.csv', 'testLabels.csv')
        print('{} testing score: {}'.format(hidden_nodes, score(net, test_data)))

if __name__ == '__main__':

    main()
