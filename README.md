# Real-Time Face Mask Detection Web Application

## Objective
This project uses a deep learning model to detect whether a person is wearing a face mask in real-time and serves the result as a web application using Flask.

---

## How It Works
1.  **Model:** A Convolutional Neural Network (CNN) was built using TensorFlow/Keras and trained on a public dataset to classify images as 'Mask' or 'No Mask'.
2.  **Backend Server:** A Flask server (`app.py`) loads the trained model. It uses OpenCV to capture a video stream from the webcam.
3.  **Real-Time Detection:** For each frame of the video, the server uses a Haar Cascade classifier to find faces. Each face is then processed and fed to the CNN model for prediction.
4.  **Web Streaming:** The processed video frames (with detection boxes and audio alerts) are streamed from the server to the front end.
5.  **Frontend Interface:** A simple HTML/CSS/JavaScript page (`templates/index.html`) provides a professional UI, including a "Start" button and contact information.

---

## Technologies Used
* **Backend:** Python, Flask, TensorFlow, Keras, OpenCV, NumPy, Winsound
* **Frontend:** HTML, CSS, JavaScript

---

## How to Run Locally
1.  Clone or download this repository.
2.  Install the required libraries: `pip install Flask tensorflow opencv-python numpy`
3.  Ensure the file structure is correct:
    ```
    .
    |-- app.py
    |-- face_mask_detector.h5
    |-- haarcascade_frontalface_default.xml
    `-- templates/
        `-- index.html
    ```
4.  Run the Flask application from your terminal: `python app.py`
5.  Open your web browser and navigate to `http://127.0.0.1:5000`.

---
## Creator
* **Name:** kalyan
* **Contact:** 8317593773
* **Email:** [kalyan24252@gmail.com](mailto:kalyan24252@gmail.com)
* **LinkedIn:** [linkedin.com/in/kalyan-ram-garaga](https://www.linkedin.com/in/kalyan-ram-garaga)
* **GitHub:** [github.com/Bunty722](https://github.com/Bunty722)
