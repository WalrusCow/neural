import math
import random
from collections import namedtuple

class Perceptron():
    def __init__(self, num_inputs):
        self.weights = [random.random() - 0.5] * (num_inputs + 1)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_prime(x):
    y = sigmoid(x)
    return y * (1 - y)


def sigmoid_to_label(x):
    ''' Convert activation function output to label. '''
    return 5 if x <= 0.5 else 6


def label_to_output(l):
    ''' Convert a label to desired activation function output. '''
    return 0 if l == 5 else 1


NodeData = namedtuple('NodeData', ['inp', 'act'])


class NeuralNetwork():
    def __init__(self, levels, activation_fun, activation_prime):
        ''' The first entry in "levels" is the number of inputs. '''
        self._act = activation_fun
        self._act_prime = activation_prime
        self._levels = []
        for num_inputs, num_nodes in zip(levels, levels[1:]):
            self._levels.append([Perceptron(num_inputs)
                                 for _ in range(num_nodes)])

    def label(self, inputs):
        ''' Label a set of inputs. '''
        all_outputs = self._get_outputs(inputs)
        # Final output is the label result
        return all_outputs[-1][0].act

    def _get_outputs(self, inputs):
        # Outputs for every level
        all_outputs = []
        inputs = [NodeData(0, i) for i in inputs]
        all_outputs.append(inputs)
        for level in self._levels:
            outputs = []
            for node in level:
                inp = sum(w * n.act for w, n in zip(node.weights, inputs))
                outputs.append(NodeData(inp, self._act(inp)))
            # Also append a bias
            outputs.append(NodeData(0, -1))
            all_outputs.append(outputs)
            inputs = outputs
        return all_outputs

    def _learn_example(self, inputs, true_out, rate):
        ''' Example inputs, the true output and the learning rate. '''
        node_outputs = self._get_outputs(inputs)
        net_out = node_outputs[-1][0].act

        prev_deltas = []

        # Backpropagation
        for ln in range(len(self._levels) - 1, -1, -1):
            level = self._levels[ln]
            deltas = []
            for i in range(len(level)):
                # Do the ith node on this level
                node = level[i]
                node_data = node_outputs[ln + 1][i]

                # Compute the delta
                gp = self._act_prime(node_data.inp)
                if not prev_deltas:
                    # We are an output node, so we need to compute our delta
                    # a little differently
                    delta = gp * (true_out - net_out)
                    #print('')
                    #print('')
                    #print('')
                    #print('#'*80)
                    #print('#'*80)
                    #print('gp: {} true: {} ours: {}'.format(gp,true_out, net_out))
                    #print('Got the delta as the first node: {}'.format(delta))
                    #print('#'*80)
                    #print('#'*80)
                    #print('')
                    #print('')
                    #print('')
                else:
                    # Get the weight to each child of this node
                    ww = (n.weights[i] for n in self._levels[ln + 1])
                    delta = gp * (sum(w * d for w, d in zip(ww, prev_deltas)))
                    #if (gp < 1e-10):
                        #print('SMALL GP {} on {}'.format(gp, node_data.inp))
                    #print('\n\nGot the delta hidden node: {} {} {}'.format(delta, gp, prev_deltas))

                # Now that we have our delta...
                for pi, parent in enumerate(node_outputs[ln]):
                    # Update the weights for each parent according to how
                    # much they contributed towards the error (i.e. "delta")
                    #print('delta: {} act:{}'.format(delta, parent.act))
                    #print('Updating node weight {} w/ {}'.format(pi, rate*delta*parent.act))
                    node.weights[pi] += rate * delta * parent.act
                deltas.append(delta)
            prev_deltas = deltas

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
    for lst, _ in train_data:
        lst.insert(0, -1)
    num_inputs = len(train_data[0][0])
    #train_data=train_data[:1]
    NN = NeuralNetwork([num_inputs, 5, 1], sigmoid, sigmoid_prime)
    NN.learn(train_data, 2.1, 1000)
    print('Training score: {}'.format(score(NN, train_data)))

    test_data = read_data('testData.csv', 'testLabels.csv')

if __name__ == '__main__':
    main()
