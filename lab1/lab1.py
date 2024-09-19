import random

def calculate_weighted_sum(w1, w2, b, x1, x2):
    return w1 * x1 + w2 * x2 + b


def perceptron_output(weighted_sum):
    return 1 if weighted_sum > 0 else -1


def test_perceptron(w1, w2, b, test_features, expected_values):
    correct_predictions = 0
    total_tests = len(expected_values)
    
    for i in range(total_tests):
        x1 = test_features[0][i]
        x2 = test_features[1][i]
        
        weighted_sum = calculate_weighted_sum(w1, w2, b, x1, x2)
        
        y = perceptron_output(weighted_sum)
        
        if y == expected_values[i]:
            correct_predictions += 1
    
    accuracy = (correct_predictions / total_tests) * 100
    return accuracy


file = open('IS-Lab1/Data.txt')

color, roundness, label = [], [], []

for line in file:
    line = line.rsplit(',')
    color.append(float(line[0]))
    roundness.append(float(line[1]))
    label.append(int(line[2]))

file.close()

estimated_features = [color, roundness]

w1 = random.uniform(-1, 1) # Weight 1 
w2 = random.uniform(-1, 1) # Weight 2
b = random.uniform(-1, 1) # Bias

eta = 0.001 # Learning rate


############
# TRAINING #
############

error = float('inf')
while error != 0:
    total_error = 0
    for i in range(len(label)):
        x1 = estimated_features[0][i]
        x2 = estimated_features[1][i]
        
        weighted_sum = calculate_weighted_sum(w1, w2, b, x1, x2)
        
        y = perceptron_output(weighted_sum)
        
        error = label[i] - y
        
        total_error += abs(error)
        
        w1 += eta * error * x1
        w2 += eta * error * x2
        b += eta * error
        
        error = total_error
        
###########
# TESTING #
###########
        
# Test data:
# 0.36484,0.8518,1
# 0.46111,0.82518,1
# 0.08838,0.62068,-1
# 0.098166,0.79092,-1

test_features = [[0.36484, 0.46111, 0.08838, 0.098166],
                 [0.8518, 0.82518, 0.62068, 0.79092]]
expected_result = [1, 1, -1, -1]
accuracy = test_perceptron(w1, w2, b, test_features, expected_result)
print(f"Accuracy: {accuracy}%")
print(f"Trained weights: w1={w1}, w2={w2}, bias={b}")

# Conclusion: with 10 data samples, the perceptron model was able to achieve 100% accuracy most of the time. Need better testing
