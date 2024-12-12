import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite("lab4/contours.png", cv2.drawContours(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), contours, -1, (0, 255, 0), 2))

    segments = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        segment = binary_image[y:y+h, x:x+w]
        segment = cv2.resize(segment, (50, 70))

        features = []
        for i in range(7):
            for j in range(5):
                block = segment[i*10:(i+1)*10, j*10:(j+1)*10]
                features.append(np.mean(block) / 255.0)

        segments.append(features)

    segments = sorted(segments, key=lambda seg: np.mean(seg))
    return np.array(segments)

def train_network(train_image_path):
    features = preprocess_image(train_image_path)

    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "J"] * 8

    label_binarizer = LabelBinarizer()
    labels_binary = label_binarizer.fit_transform(labels)

    mlp = MLPClassifier(hidden_layer_sizes=(13,), max_iter=1000, random_state=1)
    mlp.fit(features, labels_binary)

    return mlp, label_binarizer

def recognize_characters(test_image_path, mlp, label_binarizer):
    features = preprocess_image(test_image_path)

    predictions = mlp.predict(features)
    predicted_labels = label_binarizer.inverse_transform(predictions)

    return "".join(predicted_labels)

if __name__ == "__main__":
    # TRAINING
    train_image = "lab4/training_data_og.png"
    test_image = "lab4/testing_fikcija.png"

    recognizer, binarizer = train_network(train_image)

    # TESTING
    result = recognize_characters(test_image, recognizer, binarizer)

    print("Recognized Text:", result)
