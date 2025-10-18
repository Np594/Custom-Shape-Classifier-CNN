import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.metrics import Precision, Recall, AUC, F1Score
import cv2
import cvzone
from cvzone.ClassificationModule import Classifier

Shapes_Data = "Shape_Dataset/train"
QS_Data = "QSC_Dataset/No_split"

data_dir = QS_Data  # Dataset directory
img_width = 512
img_height = 512

# Load and preprocess data
dataset = tf.keras.utils.image_dataset_from_directory(data_dir, seed=123, image_size=(img_width, img_height), batch_size=32)

class_names = dataset.class_names
num_classes = len(class_names)
print(f"There are {num_classes} classes, and they are: {class_names}")

# Shuffle dataset
dataset = dataset.shuffle(buffer_size=1000, seed=123)

# Get the total number of samples
dataset_size = len(dataset)

# Calculate the sizes for train, validation, and test sets
train_size = int(0.7 * dataset_size)  # 70% for training
val_size = int(0.2 * dataset_size)  # 20% for validation
test_size = dataset_size - train_size - val_size  # Remaining 10% for test set

# Use Take and skip to create Validation and test sets
train_data = dataset.take(train_size)
val_data = dataset.skip(train_size).take(val_size)
test_data = dataset.skip(train_size + val_size)

print("Training set size:", len(train_data))
print("Validation set size:", len(val_data))
print("Test set size:", len(test_data))

def preprocess_data(image, label):
    return image / 255.0, tf.one_hot(label, depth=num_classes)

train_data = train_data.map(preprocess_data)
val_data = val_data.map(preprocess_data)
test_data = test_data.map(preprocess_data)

# Optimise data loading
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)


def Train_Model():
    # Build the model
    model = Sequential([
        Conv2D(16, (3, 3), 1, activation="relu", input_shape=(img_width, img_height, 3)), BatchNormalization(), MaxPooling2D(),
        Conv2D(32, (3, 3), 1, activation="relu"), BatchNormalization(), MaxPooling2D(), Dropout(0.2),
        Conv2D(64, (3, 3), 1, activation="relu"), BatchNormalization(), MaxPooling2D(),
        Conv2D(128, (3, 3), 1,  activation="relu"), BatchNormalization(), MaxPooling2D(),
        GlobalAveragePooling2D(),  # Reduces number of parameters for the next layer
        Dense(256, activation="relu"), BatchNormalization(), Dropout(0.3),  # Fully connected layer
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam", loss=tf.losses.CategoricalCrossentropy(), metrics=["accuracy", AUC(), F1Score(), Precision(), Recall()])  # f1 average = "macro"
    model.summary()

    # Train the model
    history = model.fit(train_data, epochs=20, validation_data=val_data)

    # Plot training history
    fig = plt.figure()
    plt.plot(history.history["loss"], color="teal", label="Train Loss")
    plt.plot(history.history["val_loss"], color="orange", label="Val Loss")
    fig.suptitle("Loss", fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    fig = plt.figure()
    plt.plot(history.history["accuracy"], color="teal", label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], color="orange", label="Val Accuracy")
    fig.suptitle("Accuracy", fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    # Evaluate the model on test data
    test_loss, test_accuracy, test_AUC, test_f1score, test_precision, test_recall = model.evaluate(test_data)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test AUC:{test_AUC}")
    print(f"Precision: {test_precision}, Recall: {test_recall}, F1 Score: {test_f1score}")

    # Save the model in Keras format
    model.save(os.path.join('Models', 'QS_Classifier.keras'))


    # Save the model in TensorFlow SavedModel format
    model.export("Models/SavedModel")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open("Models/QS_Classifier.tflite", "wb") as f:
        f.write(tflite_model)


def Test_Model():
    # Load model and labels
    QS_model, QS_Labels = "Models/QS_Classifier.keras", "Models/QS_labels.txt"
    shape_model, shape_labels = "Models/Shape_Models/shape_detector_model.h5", "Models/Shape_Models/Shape_Labels.txt"
    qs_classifier = Classifier(shape_model, shape_labels)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (512, 512))

        # Make Predictions
        predictions, index = qs_classifier.getPrediction(frame, scale=1)
        print(f"Predictions: {predictions}, Class Index: {index}")

        # Show frame
        cv2.imshow("Classifier", frame)

        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


#Train_Model()
Test_Model()
