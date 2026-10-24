import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- Load Data ---
# Path to processed_data folder (local)
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed_data')
# IMPORTANT: This gestures list must match the one used for data collection
gestures = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'nothing'])
X = np.load(os.path.join(PROCESSED_DATA_PATH, 'X_data.npy')) # Corrected filename
y = np.load(os.path.join(PROCESSED_DATA_PATH, 'y_labels.npy')) # Corrected filename

# --- Prepare Data for Training ---
y_categorical = to_categorical(y, num_classes=len(gestures)).astype(int) # Added num_classes
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# --- Build a More Powerful CNN Model for the Alphabet ---
model = Sequential()
model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2)) # Dropout after conv layer
model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2)) # Dropout after conv layer
model.add(Flatten())

# Dense layers are now larger and include more Dropout
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# The final layer must match the number of classes (27)
model.add(Dense(gestures.shape[0], activation='softmax'))

# --- Compile and Train ---
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# Training will take longer with more data and a bigger model
history = model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test))

# --- Save the Final Model ---
model.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sign_language_model_alphabet.h5'))
print("\nAlphabet model saved as sign_language_model_alphabet.h5")

# --- Visualize Training History ---
# (Visualization code is the same as before)
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
cm = confusion_matrix(y_true_labels, y_pred_labels)
print(cm)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=gestures, yticklabels=gestures)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# --- NEW: Visualize Training History ---
print("\nVisualizing model performance...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot training & validation accuracy values
ax1.plot(history.history['categorical_accuracy'])
ax1.plot(history.history['val_categorical_accuracy'])
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

plt.show()
model.summary()