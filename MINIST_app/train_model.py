import os

import keras
from keras import Sequential
from keras.layers import Dense, Flatten


MODEL_PATH = os.path.join("models", "digits_intel.h5")


def build_model():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(units=100, activation="relu"))
    model.add(Dense(units=100, activation="relu"))
    model.add(Dense(units=10, activation="softmax"))
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    return model


def main():
    os.makedirs("models", exist_ok=True)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = build_model()
    model.fit(x=x_train, y=y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
    model.save(MODEL_PATH)
    print(f"Model saved to: {os.path.abspath(MODEL_PATH)}")


if __name__ == "__main__":
    main()
