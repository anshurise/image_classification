import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def train_and_save(data_path: str, model_path: str):
    # Dataset loading
    data_dir = Path(data_path)
    categories = ['cat', 'dog']

    images, labels = [], []

    for label, category in enumerate(categories):
        for img_file in (data_dir / category).iterdir():
            img = tf.keras.utils.load_img(img_file, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(label)

    # Split data
    images = np.array(images)
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Model definition
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)

    # Freeze base layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="binary_crossentropy", metrics=["accuracy"])

    # Train model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

    # Save model
    model.save(model_path)
    print(f"Model saved at {model_path}")


if __name__ == "__main__":
    train_and_save("data", "models/cat_dog_model.h5")
