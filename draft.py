import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import MobileNetV2
import numpy as np


# Function to load a limited dataset
def load_limited_dataset(directory, subset, validation_split, limit_per_class, seed):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        image_size=(img_height, img_width),
        batch_size=1,  # Set to 1 initially to filter images individually
        subset=subset,
        validation_split=validation_split,
        seed=seed,
        shuffle=True,
    )

    # Store class names
    class_names = dataset.class_names

    # Limit the number of images per class
    limited_dataset = []
    class_counts = {class_name: 0 for class_name in class_names}

    for images, labels in dataset:
        for i in range(len(images)):
            class_name = class_names[labels[i].numpy()]
            if class_counts[class_name] < limit_per_class:
                limited_dataset.append((images[i], labels[i]))
                class_counts[class_name] += 1

            # Break the loop if the limit for all classes is reached
            if all(count >= limit_per_class for count in class_counts.values()):
                break
        if all(count >= limit_per_class for count in class_counts.values()):
            break

    # Convert to TensorFlow dataset and batch it
    limited_dataset = tf.data.Dataset.from_tensor_slices(
        (
            np.array([img for img, _ in limited_dataset]),
            np.array([label for _, label in limited_dataset]),
        )
    )

    # Batch the dataset to the desired batch size
    return limited_dataset.batch(batch_size), class_names


# Define directories and parameters
raw_data_dir = "data/ccmt_raw"
augmented_data_dir = "data/ccmt_augmented"
batch_size = 32
img_height, img_width = 224, 224
limit_per_class = 100  # Use a larger limit for more data

# Data augmentation layer
data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ]
)

# Load datasets with the previously defined load_limited_dataset function
train_dataset, class_names = load_limited_dataset(
    augmented_data_dir,
    subset="training",
    validation_split=0.2,
    limit_per_class=limit_per_class,
    seed=123,
)
validation_dataset, _ = load_limited_dataset(
    augmented_data_dir,
    subset="validation",
    validation_split=0.2,
    limit_per_class=limit_per_class,
    seed=123,
)

# Print out the classes
print("Classes:", class_names)


# Using MobileNetV2 for transfer learning
def create_transfer_model(input_shape=(img_height, img_width, 3), num_classes=5):
    base_model = MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # Freeze the base model

    model = models.Sequential(
        [
            Input(shape=input_shape),
            data_augmentation,
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model


# Create the transfer learning model
num_classes = len(class_names)
model = create_transfer_model(num_classes=num_classes)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train the model for more epochs with early stopping
epochs = 15
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping],
)

# Evaluate the model on the validation dataset
val_loss, val_accuracy = model.evaluate(validation_dataset)
print(f"Validation Accuracy: {val_accuracy:.2f}")
