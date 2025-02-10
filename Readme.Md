# Drowsiness Detection System

## 📌 Project Overview

This project is a **real-time drowsiness detection system** using OpenCV, Dlib, and Streamlit. It helps detect drowsiness in drivers or users by analyzing eye closure duration using the PERCLOS method.

## ⚙️ Features

- **Real-time eye detection** using Dlib's face landmark predictor
- **PERCLOS Calculation** for drowsiness detection
- **Audio Alerts** to warn the user when drowsiness is detected
- **Live Streaming UI** built with Streamlit for easy interaction

---

## 🛠 Installation Guide

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/ayushh0406/Drowsiness_-Detection.git
cd Drowsiness_-Detection
```

### 2️⃣ Install Dependencies

Ensure you have Python installed (recommended: Python 3.8+). Then, install dependencies:

```sh
pip install -r requirements.txt
```

### 3️⃣ Download Shape Predictor Model

Download the `shape_predictor_68_face_landmarks.dat` file from:
🔗 [Dlib Model Download](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

Extract the `.bz2` file and place `shape_predictor_68_face_landmarks.dat` inside the project folder.

---

## 🚀 How to Run the Project

### ▶️ Running with OpenCV (Terminal)

```sh
python app.py
```

### ▶️ Running with Streamlit (Web UI)

```sh
streamlit run app.py
```

This will launch a **web-based UI** where you can see the live video feed and drowsiness alerts.

---

## 🌍 Deployment on Streamlit Cloud

To deploy the app online:
1️⃣ Go to **[Streamlit Cloud](https://share.streamlit.io/)**
2️⃣ Click **"New App"**
3️⃣ Select your **GitHub repo**
4️⃣ Choose **`app.py`** as the entry file
5️⃣ Click **"Deploy"**

---

##

---

## 🤝 Contributing

Feel free to **fork** this repo, make improvements, and submit a **pull request**! 😊

---

## 📩 Contact

For queries, reach out to [Your Email or GitHub Profile].

---

⭐ **If you like this project, please give it a star on GitHub!** ⭐

