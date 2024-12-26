import tensorflow as tf
import numpy as np

def load_model(model_path: str):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def predict(image_path: str, model_path: str):
    model = load_model(model_path)

    # Preprocess image
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Batch dimension

    # Predict
    prediction = model.predict(img_array)
    label = "dog" if prediction[0][0] > 0.5 else "cat"
    return {"prediction": label, "confidence": float(prediction[0][0])}


if __name__ == "__main__":
    result = predict("test_image.jpg", "models/cat_dog_model.h5")
    print(result)
