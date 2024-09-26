import os
import numpy as np
import cv2  # Import OpenCV for reading images
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Path to the dataset
base_dir = 'D:/DIP/FINGER-VEIN-OUT/'

# Image dimensions
img_size = 64  # Resize images to 64x64

# Subfolder names within each main folder
hand_sides = ['left', 'right']  # Folders for left and right hand
finger_types = ['index', 'middle', 'ring']  # Finger subfolders

# Initialize lists to hold data and labels
data = []
labels = []
image_count = 0  # To keep track of how many images are loaded

# Traverse the dataset directory
for folder in os.listdir(base_dir):
    main_folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(main_folder_path):  # Check if it is a directory
        for hand_side in hand_sides:  # Iterate over left and right folders
            hand_folder_path = os.path.join(main_folder_path, hand_side)
            if os.path.isdir(hand_folder_path):  # Check if left/right folder exists
                for finger in finger_types:  # Iterate over index, middle, ring
                    finger_folder_path = os.path.join(hand_folder_path, finger)
                    if os.path.isdir(finger_folder_path):  # Check if finger folder exists
                        # Load all .bmp images in the folder
                        for img_name in os.listdir(finger_folder_path):
                            if img_name.lower().endswith('.bmp'):  # Process only BMP images
                                img_path = os.path.join(finger_folder_path, img_name)
                                
                                # Load image in grayscale mode using OpenCV
                                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    # Resize the image to the target size
                                    img = cv2.resize(img, (img_size, img_size))
                                    data.append(img)  # Append the image to data

                                    # Create a label based on the folder structure (category, hand, finger)
                                    labels.append(f"{folder}_{hand_side}_{finger}")  # Label format: folder_left_index

                                    image_count += 1  # Increment the image count

# Convert the list to numpy arrays and normalize pixel values
data = np.array(data).reshape(-1, img_size, img_size, 1)  # Add a channel dimension for grayscale images
data = data / 255.0  # Normalize pixel values to [0, 1]

# Print out how many images were loaded
print(f"Total images loaded: {len(data)}")
print(f"Total labels: {len(labels)}")

# Convert labels to numeric using LabelEncoder
label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)  # Convert string labels to numeric values

# One-hot encode the labels for multi-class classification
num_classes = len(np.unique(numeric_labels))
numeric_labels = to_categorical(numeric_labels, num_classes=num_classes)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, numeric_labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),  # For grayscale images
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Optional: Adding Dropout layer
    Dense(num_classes, activation='softmax')  # Multi-class classification output layer
])

# Compile the model
#optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # For multi-class classification
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {test_acc}")

# Optionally, plot training history
plt.figure(figsize=(8, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
