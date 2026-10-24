# Sign Language Detection Project

This project is a real-time Sign Language Translate web application built using Flask, MediaPipe, and TensorFlow. It captures video feed from your webcam, extracts hand landmarks, and uses pre-trained AI models to predict either sign language "words" or "alphabets".

---

## 🛠️ Prerequisites
Ensure you have Python installed on your system. It is highly recommended to use Python versions between **3.8 and 3.10** for the best compatibility with TensorFlow and MediaPipe. 
*(If you are setting this up for the first time, you might also want to set up a virtual environment).*

## 📦 Required Libraries
The project now includes a `requirements.txt` file with exact compatible versions. The required packages and their tested versions are:
- `Flask==3.0.0`
- `flask-cors==4.0.0`
- `Flask-SocketIO==5.3.6`
- `opencv-python==4.9.0.80`
- `numpy==1.26.4`
- `mediapipe==0.10.9`
- `tensorflow==2.13.0`

## 🚀 Step-by-Step Setup and Execution Guide

### Step 1: Open Terminal / Command Prompt
Open your terminal (Command Prompt or PowerShell) and navigate to the project directory where `app.py` is located:
```bash
cd "c:\Users\rouni\Desktop\last final project"
```

### Step 2: Install the Dependencies
Run the following pip command to install all necessary dependencies with their specific versions using the requirements file:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Model Files
Before starting the application, ensure the following two trained model files are present in the same directory as `app.py`:
1. `sign_language_model_cnn.h5` (Used for word predictions)
2. `sign_language_model_alphabet.h5` (Used for single alphabet predictions)

### Step 4: Run the Application
Start the backend server by running the main Python script:
```bash
python app.py
```

### Step 5: Access the Web Interface
If the setup is successful, the console will display:
```
* Starting WebSocket Server at http://0.0.0.0:5001
  Open http://localhost:5001 in your browser
```
Open your preferred web browser (Chrome/Edge/Firefox) and navigate to:
**👉 http://localhost:5001**

---

## 🎮 How to Use
1. **Camera Permissions:** When you navigate to the webpage, your browser will ask for camera permissions. Click **Allow**.
2. **Navigation:** Use the web UI to navigate between the "Detect Words" and "Detect Alphabets" modes.
3. **Detection:** 
   - Show your hand gestures in the camera frame.
   - The system will process the video, analyze the hand landmarks, and display the predicted phrase or letter along with a confidence score.
