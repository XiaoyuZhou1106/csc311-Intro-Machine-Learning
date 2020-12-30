import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import data
import numpy as np


def training(train_data, train_label):
    model = Sequential()

    #add layers to the model
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation='relu'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    # Convert labels to categorical one-hot encoding
    one_hot_labels = keras.utils.to_categorical(train_label, num_classes=10)
    # Train the model
    model.fit(train_data, one_hot_labels, epochs=10, batch_size=32)

    return model

def main():
    #load the data and initialize the MLP model
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    model = training(train_data, train_labels)

    #calculate the accuracy
    predict_result = np.argmax(model.predict(test_data), axis=1)
    sum = 0
    for i in range(len(predict_result)):
        if predict_result[i] == test_labels[i]:
            sum += 1
    accuracy = sum / len(test_labels)
    print(accuracy)
    score = model.evaluate(test_data, test_labels, batch_size=32)
    print(score)

if __name__ == '__main__':
    final = main()