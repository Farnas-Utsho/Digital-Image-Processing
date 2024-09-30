import os
import numpy as np
import cv2  # Import OpenCV for reading images
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# Print shapes of the train and validation datasets
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# Import necessary layers and regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Layer, Concatenate
from tensorflow.keras.backend import mean, max
from tensorflow.keras.layers import multiply

# Define the Spatial Attention Layer as a custom layer
class SpatialAttention(Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv_attention = Conv2D(1, (3, 3), padding='same', activation='sigmoid')

    def call(self, input_tensor):
        """Spatial Attention Module"""
        avg_pool = mean(input_tensor, axis=-1, keepdims=True)
        max_pool = max(input_tensor, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool, max_pool])
        attention = self.conv_attention(concat)
        return multiply([input_tensor, attention])

# Define the CNN model with Spatial Attention and L2 regularization
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    AveragePooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    AveragePooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    AveragePooling2D((2, 2)),
    
    Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    AveragePooling2D((2, 2)),
    
    SpatialAttention(),  # Use the custom Spatial Attention layer
    
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')  # Output layer for multi-class classification
])

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_val, y_val)
print(f"Validation accuracy: {test_acc}")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
