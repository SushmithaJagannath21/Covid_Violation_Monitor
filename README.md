# Covid Violation Monitor

The **Covid Violation Monitor** is a Python-based system designed to detect and monitor violations of Covid-19 safety protocols, such as social distancing and face mask usage. Using computer vision and deep learning techniques, the system identifies individuals who are not adhering to these safety guidelines in real-time.

## Features
- **Face Mask Detection**: Detects whether individuals are wearing face masks using a pre-trained **CNN (Convolutional Neural Network)** model.
- **Social Distancing Monitoring**: Measures the distance between individuals to ensure compliance with social distancing guidelines using **YOLO (You Only Look Once)** object detection.
- **Facial Recognition**: Implements **FaceNet** for identifying individuals, aiding in potential follow-up or alerts for repeated violations.
- **Real-Time Processing**: Processes live video feeds or camera input at up to 60 FPS for real-time monitoring of Covid-19 violations.
- **Visual Alerts**: Highlights violations with visual cues on the video feed, enabling easier detection by authorities or security personnel.

## Technologies Used
- **Python**: Core programming language for the project.
- **OpenCV**: Used for real-time video processing and object detection.
- **YOLO (You Only Look Once)**: Deep learning model used for object detection, specifically to identify people.
- **CNN (Convolutional Neural Network)**: Trained to detect face mask compliance.
- **Streamlit**: Provides the web interface to showcase the video feed and processing in real-time.

## Project Structure

```bash
├── static/                         # Static resources (images, videos, etc.)
├── .gitattributes                   # Git configuration for handling model and weights files
├── app.py                           # Streamlit app for running the Covid Violation Monitor
├── coco.names                       # YOLO class names file
├── detector.pickle                  # Pre-trained face mask detector
├── encodings.pickle                 # Encoded facial recognition data
├── environment.yml                  # Conda environment configuration
├── my_model1.h5                     # Pre-trained mask detection CNN model
├── packages.txt                     # List of packages and dependencies
├── yolov3.cfg                       # YOLOv3 configuration file for object detection
└── yolov3.weights                   # YOLOv3 weights file for object detection
```


## Installation
Clone the repository:
git clone https://github.com/SushmithaJagannath21/Covid_Violation_Monitor.git

## Navigate to the project directory:
cd Covid_Violation_Monitor

## Create a virtual environment and install the dependencies:
conda env create -f environment.yml
conda activate covid_violation_monitor

## Run the Streamlit application:
streamlit run app.py

## Usage
Ensure that your camera feed or video input is connected.
Launch the Streamlit app using the command above.
The web interface will display the video feed with real-time monitoring of social distancing and face mask violations.
Violations will be visually highlighted on the video feed.

## Results
Accuracy: The system achieves an accuracy of 92.2% for mask detection and 95% accuracy for facial recognition using pre-trained models.
Real-Time Performance: Processes up to 60 frames per second (FPS), ensuring efficient monitoring for real-time applications.

