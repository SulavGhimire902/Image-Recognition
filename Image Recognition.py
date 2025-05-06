import os
import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd

# Paths
images_folder = "D:\\archive\\images"  
excel_file = "D:\\archive\\Harumanis_mango_weight_grade.xlsx"  
model_file = "mango_ripeness_model.keras"  

# Function to preprocess images
def preprocess_image(image_path, label=None, target_size=(128, 128)):
    img = tf.io.read_file(image_path)  
    img = tf.image.decode_jpeg(img, channels=3) 
    img = tf.image.resize(img, target_size)  
    img = img / 255.0  
    if label is None:
        return img
    return img, label

# Create TensorFlow datasets
def create_dataset(image_paths, labels=None, batch_size=32):
    image_paths = tf.constant(image_paths)  
    if labels is not None:
        labels = tf.constant(labels)  
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(lambda x, y: preprocess_image(x, y))  
    else:
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(preprocess_image)  
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Step 1: Train the Model (if it doesn't exist)
if not os.path.exists(model_file):
    print("Training a new model...")

    # Load Metadata from Excel
    metadata = pd.read_excel(excel_file, engine='openpyxl')

    # Function to clean unwanted characters
    def clean_string(s):
        if isinstance(s, str):
            s = s.replace('\u2501', '')  
        return s

    metadata["Fruit No"] = metadata["Fruit No"].apply(clean_string)
    metadata["Color K Yellow P Green"] = metadata["Color K Yellow P Green"].apply(clean_string)

    metadata["image_path"] = metadata["Fruit No"].apply(lambda x: os.path.join(images_folder, str(x)))
    metadata["label"] = metadata["Color K Yellow P Green"].map({"K": 1, "P": 0})

    # Load image paths and labels
    image_paths = metadata["image_path"].values
    labels = metadata["label"].values

    # Train-test split
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = create_dataset(train_paths, train_labels)
    val_dataset = create_dataset(val_paths, val_labels)

    # Build the CNN Model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_dataset, validation_data=val_dataset, epochs=10)

    # Evaluate the model
    loss, accuracy = model.evaluate(val_dataset)
    print(f"Validation Accuracy: {accuracy:.2f}")

    # Save the trained model
    model.save(model_file)
    print(f"Model saved to {model_file}")
else:
    print(f"Loading model from {model_file}...")
    model = tf.keras.models.load_model(model_file)

# Step 2: Predict from User Input
def predict_image(model, image_path, target_size=(128, 128)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = preprocess_image(image_path, target_size=target_size)
    img = tf.expand_dims(img, axis=0) 
    prediction = model.predict(img)
    return "Ripe" if prediction[0][0] > 0.5 else "Raw"

# User input for prediction
user_image_path = input("Enter the path of the mango image: ").strip()
try:
    result = predict_image(model, user_image_path)
    print(f"The mango in the image is: {result}")
except FileNotFoundError as e:
    print(e)
