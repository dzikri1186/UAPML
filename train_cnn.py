import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf

# Cek versi TensorFlow dan status GPU
print(f"TensorFlow version: {tf.__version__}")
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Path dataset dengan forward slash
train_dir = "C:/Users/Dzikri/OneDrive/Documents/Semester 7/chicken-classification/data/train"
val_dir = "C:/Users/Dzikri/OneDrive/Documents/Semester 7/chicken-classification/data/val"
test_dir = "C:/Users/Dzikri/OneDrive/Documents/Semester 7/chicken-classification/data/test"

# Verifikasi apakah path dataset ada
print(f"Train directory exists: {os.path.exists(train_dir)}")
print(f"Val directory exists: {os.path.exists(val_dir)}")
print(f"Test directory exists: {os.path.exists(test_dir)}")

# Data preprocessing
print("Creating data generators...")
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

print("Data generators created successfully!")

print("Building CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax"),  # Jumlah kelas
])
print("Model built successfully!")

print("Compiling model...")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("Model compiled successfully!")

print("Starting model training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
)
print("Model training completed!")

# Save the trained model
print("Saving trained model...")
os.makedirs("C:/Users/Dzikri/OneDrive/Documents/Semester 7/chicken-classification/models", exist_ok=True)
model.save("C:/Users/Dzikri/OneDrive/Documents/Semester 7/chicken-classification/models/cnn_model.h5")
print("Model saved successfully!")
