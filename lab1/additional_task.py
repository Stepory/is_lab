import math

def mean(values):
    return sum(values) / len(values)


def variance(values):
    mean_value = mean(values)
    return sum([(x - mean_value) ** 2 for x in values]) / len(values)


def gaussian_probability(x, mean, var):
    if var == 0:
        return 1 if x == mean else 0
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var)))
    return (1 / math.sqrt(2 * math.pi * var)) * exponent


def summarize_by_class(features, labels):
    summaries = {}
    class_labels = set(labels)
    
    for class_value in class_labels:
        class_features = [features[i] for i in range(len(features)) if labels[i] == class_value]
        class_features = list(zip(*class_features))
        summaries[class_value] = [(mean(feature), variance(feature)) for feature in class_features]
    
    return summaries


def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean_value, var = class_summaries[i]
            probabilities[class_value] *= gaussian_probability(input_vector[i], mean_value, var)
    return probabilities


def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    return max(probabilities, key=probabilities.get)


file = open('IS-Lab1/Data.txt')

spalva, apvalumas, expected = [], [], []

for line in file:
    line = line.rsplit(',')
    spalva.append(float(line[0]))
    apvalumas.append(float(line[1]))
    expected.append(int(line[2]))

file.close()

features = list(zip(spalva, apvalumas))
summaries = summarize_by_class(features, expected)

test_features = [[0.36484, 0.46111, 0.08838, 0.098166],
                 [0.8518, 0.82518, 0.62068, 0.79092]]
expected_result = [1, 1, -1, -1]

predictions = []
for i in range(len(test_features[0])):
    input_vector = [test_features[0][i], test_features[1][i]]
    predictions.append(predict(summaries, input_vector))

correct = sum(1 for i in range(len(expected_result)) if expected_result[i] == predictions[i])
accuracy = (correct / len(expected_result)) * 100

print(f"Accuracy: {accuracy}%")
print(f"Predictions: {predictions}")
