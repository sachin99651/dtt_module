import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import seaborn as sns

# --- Load Data ---
PROCESSED_DATA_PATH = "processed_data"
X = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_data.npy'))
y = np.load(os.path.join(PROCESSED_DATA_PATH, 'y_labels.npy'))
gestures = np.array(['hello', 'thank_you', 'yes', 'no', 'goodbye'])

# --- Prepare Data for Training ---
y_categorical = to_categorical(y).astype(int)

# --- NEW: Splitting data into train, validation, and test sets (60/20/20 split) ---
# First, split into 80% for training+validation and 20% for the final test set
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical)

# Now, split the 80% into training and validation sets
# Here, test_size=0.25 means 25% of the 80% becomes validation data (i.e., 20% of the total)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)


# --- Build the CNN Model ---
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(75, 63)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(gestures.shape[0], activation='softmax'))

# --- Compile the Model ---
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- Train the Model ---
print("Training the CNN model with validation data...")
# --- MODIFIED: Added validation_data and assigned output to 'history' ---
history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_valid, y_valid))


# --- Save the Model ---
model.save('sign_language_model_cnn.h5')
print("\nModel saved as sign_language_model_cnn.h5")


# --- NEW: Evaluate the Model on the Test Set ---
print("\nEvaluating model on the test set...")

# Get predictions on the test set
y_pred_probs = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Print classification report
print("\n--- Classification Report ---")
print(classification_report(y_true_labels, y_pred_labels, target_names=gestures, labels=np.arange(len(gestures))))

# Generate and plot confusion matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=np.arange(len(gestures)))
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=gestures, yticklabels=gestures)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')

# --- NEW: Visualize Training History ---
print("\nVisualizing model performance...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Safely get the accuracy metric key (varies by TF version)
acc_key = 'categorical_accuracy' if 'categorical_accuracy' in history.history else 'accuracy'
val_acc_key = 'val_categorical_accuracy' if 'val_categorical_accuracy' in history.history else 'val_accuracy'

# Plot training & validation accuracy values
ax1.plot(history.history[acc_key])
ax1.plot(history.history[val_acc_key])
ax1.set_title('Model Accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])
ax2.set_title('Model Loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Validation'], loc='upper left')

model.summary()
plt.show()
