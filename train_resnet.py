import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Path dataset
train_dir = "C:/Users/Dzikri/OneDrive/Documents/Semester 7/chicken-classification/data/train"
val_dir = "C:/Users/Dzikri/OneDrive/Documents/Semester 7/chicken-classification/data/val"
test_dir = "C:/Users/Dzikri/OneDrive/Documents/Semester 7/chicken-classification/data/test"

# Data preprocessing
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

# Load pretrained ResNet50 model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Tambahkan lapisan kustom di atas ResNet
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # Jumlah kelas (10 jenis ayam)
])

# Freeze lapisan pretrained (opsional)
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
print("Starting ResNet training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
)
print("ResNet training completed!")

# Save ResNet model
print("Saving ResNet model...")
os.makedirs("C:/Users/Dzikri/OneDrive/Documents/Semester 7/chicken-classification/models", exist_ok=True)
model.save("C:/Users/Dzikri/OneDrive/Documents/Semester 7/chicken-classification/models/resnet_model.h5")
print("ResNet model saved successfully!")
