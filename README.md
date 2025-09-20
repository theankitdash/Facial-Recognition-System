# Face Authentication with Liveness Detection

This project is a **Python-based face authentication system** that combines **face recognition** with **liveness detection** to prevent spoofing attacks (e.g., using photos or videos).

## Features

* **Face Registration & Recognition**

  * Uses **MTCNN** for face detection.
  * Uses **FaceNet (InceptionResnetV1)** for generating face embeddings.
  * Stores embeddings locally in `faces_db/`.

* **Liveness Detection (Anti-Spoofing)**

  * **Blink Detection** – verifies natural eye blinks.
  * **Head Movement** – ensures the user turns left/right.
  * **Smile Detection** – detects mouth widening and lip movement.
  * Randomized challenges make it harder to cheat.

* **Interactive Menu (OpenCV GUI)**

  * Register new users.
  * Login with liveness + face recognition.
  * Quit the system.

## Tech Stack

* **Python**, **OpenCV** – video capture & display.
* **Mediapipe** – facial landmarks (eyes, mouth, head movement).
* **Facenet-PyTorch** – face embeddings & recognition.
* **NumPy, Torch** – vector computations & ML.

## Workflow

1. **Register** → Capture 50 samples of user face, store average embedding.
2. **Login** → Run randomized liveness challenges (blink, head, smile).
3. **Authenticate** → Compare captured embedding against saved embeddings.
4. **Result** → Access granted if similarity > threshold.

## Usage

1. Run the Python script.
2. Use the menu to register or login.
3. Follow the on-screen prompts for liveness challenges.

## Directory Structure

```
faces_db/      # Stores user embeddings
test.py        # Main script with all functionality
```
