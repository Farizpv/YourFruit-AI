# FruitAI-Flask-App 🍎🍌🍇

## Intelligent Fruit Identification and Freshness Advisor

---

## 🌟 Overview

The FruitAI-Flask-App is a web application designed to help users identify various types of fruits and assess their freshness based on an uploaded image. It leverages machine learning models to provide instant predictions, along with practical tips on shelf life and optimal storage. This project aims to contribute to reducing food waste and promoting healthier eating habits.

---

## ✨ Features

* **Fruit Identification:** Identifies common fruits from uploaded images.
* **Freshness Assessment:** Predicts the freshness status (e.g., ripe, unripe, overripe, rotten) of the identified fruit.
* **Confidence Scores:** Shows how confident the model is in its predictions.
* **Smart Storage Tips:** Offers practical advice on optimal shelf life and storage methods for each fruit and its freshness state.
* **User-Friendly Interface:** A simple web interface for easy image uploads and result viewing.

---

## 🚀 Getting Started (Local Development)

To get a local copy of this project up and running on your machine, follow these steps.

### Prerequisites

Make sure you have the following installed:

* **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
* **Git**: [Download Git](https://git-scm.com/downloads)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Farizpv/YourFruit-AI.git
    cd YourFruit-AI
    ```

2.  **Set up a virtual environment:**
    It's good practice to create a virtual environment to manage project dependencies.
    ```bash
    python -m venv venv
    ```
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

Once your dependencies are installed, you can run the Flask application:

1.  **Ensure you are in your project's root directory** and your virtual environment is activated.
2.  **Run the Flask development server:**
    ```bash
    flask run
    ```

    The application will typically be accessible in your web browser at `http://127.0.0.1:5000/`.

### Important Note on AI Model Usage

* This application uses TensorFlow/Keras models for fruit identification and freshness prediction.
* During local execution, these models will run on your **CPU**.
* The project includes a setting (`os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`) in `app/predict_combined.py` to ensure TensorFlow avoids looking for a GPU, which is helpful for local development and future CPU-only deployments.

---

## 📂 Project Structure

├── app/
│   ├── init.py         # Flask app initialization

│   ├── routes.py           # Defines web routes and handles image upload/prediction

│   ├── predict_combined.py # Contains model loading and prediction logic

│   ├── templates/          # HTML templates (e.g., index.html)

│   └── static/             # CSS, JS, images for the frontend

├── models/                 # Directory containing the AI models (.h5 files)

│   ├── fruit_classifier_mobilenetv2.h5

│   └── freshness_models/

│       ├── freshness_classifier_apple.h5

│       └── ... (other freshness models and their label JSONs)

├── fruit_details.json      # JSON file containing fruit freshness tips and details

├── run.py                  # Entry point for the Flask application

└── requirements.txt        # Python dependencies

---

## 📈 Future Vision

* **Cloud Deployment:** Deploy the Flask web application to a platform like Render.com and offload AI inference to a dedicated serverless function (e.g., Google Cloud Functions) for better scalability and resource management.
* **Mobile Application:** Explore developing a native Android (or cross-platform) mobile application to bring FruitAI directly to users' smartphones.
* **More Fruit Varieties:** Expand the dataset and train models to identify a wider range of fruits.
* **Enhanced UI/UX:** Improve the user interface for a more polished and interactive experience.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions, bug reports, or ideas for improvements, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

---

## 📞 Contact

For any questions or collaborations, feel free to reach out:

* **Muhammed Fariz PV**- https://github.com/Farizpv
* **Email:** farizpv1001@gmail.com

---
