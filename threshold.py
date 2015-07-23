def read_data(inputs_file, labels_file):
    with open(inputs_file) as in_f:
        with open(labels_file) as l_f:
            inputs = (list(map(int, line.split(','))) for line in in_f)
            labels = (int(line) for line in l_f)
            return list(zip(inputs, labels))

def read_labels(filename):
    with open(filename) as f:
        return [int(line) for line in f]

if __name__ == '__main__':
    # Train our perceptron
    train_data = read_data('trainData.csv', 'trainLabels.csv')

    # Test the perceptron
    test_data = read_data('testData.csv', 'testLabels.csv')

    #tests_passed = 0
    #tests_failed = 0
    #for input, label in test_data:
    #    if perceptron.label(input) == label:
    #        tests_passed += 1
    #    else:
    #        tests_failed += 1
