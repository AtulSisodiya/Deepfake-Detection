import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Define paths
DATA_DIR = 'data'
REAL_DIR = os.path.join(DATA_DIR, 'real')
FAKE_DIR = os.path.join(DATA_DIR, 'fake')
OUTPUT_REAL_FRAMES = os.path.join('output', 'real_frames')
OUTPUT_FAKE_FRAMES = os.path.join('output', 'fake_frames')
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'deepfake_detector.h5')

# Create necessary directories
for folder in [OUTPUT_REAL_FRAMES, OUTPUT_FAKE_FRAMES, MODEL_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def extract_frames_from_videos(input_dir, output_dir, step=30):
    """
    Extract frames from all videos in the input directory.
    
    Args:
        input_dir (str): Path to the directory containing videos.
        output_dir (str): Path to the directory to save extracted frames.
        step (int): Extract one frame every 'step' frames.
    """
    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % step == 0:
                frame_filename = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}_frame_{frame_count}.jpg")
                frame_resized = cv2.resize(frame, (224, 224))
                cv2.imwrite(frame_filename, frame_resized)
            frame_count += 1
        cap.release()
        print(f"Extracted frames from {video_file}")

def load_data(real_frames_dir, fake_frames_dir):
    """
    Load images and labels from directories.
    
    Args:
        real_frames_dir (str): Directory containing real frames.
        fake_frames_dir (str): Directory containing fake frames.
    
    Returns:
        X (np.array): Array of image data.
        y (np.array): Array of labels.
    """
    real_images = []
    fake_images = []
    
    # Load real images
    for filename in os.listdir(real_frames_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(real_frames_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            real_images.append(img)
    
    # Load fake images
    for filename in os.listdir(fake_frames_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(fake_frames_dir, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fake_images.append(img)
    
    X = np.array(real_images + fake_images)
    y = np.array([0] * len(real_images) + [1] * len(fake_images))  # 0: Real, 1: Fake
    print(f"Loaded {len(real_images)} real images and {len(fake_images)} fake images.")
    return X, y

def preprocess_data(X):
    """
    Normalize image data.
    
    Args:
        X (np.array): Array of image data.
    
    Returns:
        X_normalized (np.array): Normalized image data.
    """
    X_normalized = X.astype('float32') / 255.0
    return X_normalized

def build_model(input_shape=(224, 224, 3)):
    """
    Build and compile the CNN model.
    
    Args:
        input_shape (tuple): Shape of the input images.
    
    Returns:
        model (keras.Model): Compiled CNN model.
    """
    model = Sequential()

    # First Convolutional Block
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Convolutional Block
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third Convolutional Block
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and Dense Layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    # Compile the model
    optimizer = Adam(lr=1e-4)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model

def plot_training(history):
    """
    Plot training and validation accuracy and loss.
    
    Args:
        history (keras.callbacks.History): History object from model training.
    """
    # Accuracy plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def main():
    # Step 1: Extract frames from videos
    print("Extracting frames from real videos...")
    extract_frames_from_videos(REAL_DIR, OUTPUT_REAL_FRAMES, step=30)
    print("Extracting frames from fake videos...")
    extract_frames_from_videos(FAKE_DIR, OUTPUT_FAKE_FRAMES, step=30)

    # Step 2: Load data
    print("Loading data...")
    X, y = load_data(OUTPUT_REAL_FRAMES, OUTPUT_FAKE_FRAMES)

    # Step 3: Preprocess data
    print("Preprocessing data...")
    X = preprocess_data(X)

    # Step 4: Split data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 5: Data Augmentation
    print("Setting up data augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # Step 6: Build the model
    print("Building the model...")
    model = build_model(input_shape=(224, 224, 3))

    # Step 7: Define callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_loss',
                                 save_best_only=True, verbose=1)

    # Step 8: Train the model
    print("Starting training...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        steps_per_epoch=len(X_train) // 32,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint]
    )

    # Step 9: Plot training history
    print("Plotting training history...")
    plot_training(history)

    # Step 10: Evaluate the model
    print("Evaluating the model on test data...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Step 11: Save the model
    print(f"Saving the model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Model training and saving completed.")

if __name__ == "__main__":
    main()
