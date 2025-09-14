# 🏡 House Price Prediction

Welcome to the **House Price Prediction** project! This repository contains a complete machine learning solution for predicting house prices based on various features. The project includes a Python backend for model inference and a simple, elegant frontend for user interaction.

---

## 🚀 Features

- **End-to-End ML Pipeline:** Data preprocessing, model training, and prediction using scikit-learn.
- **REST API Backend:** Fast and easy predictions via a Python Flask API.
- **User-Friendly Frontend:** Clean HTML/CSS interface for uploading data and viewing results.
- **Batch & Single Prediction:** Supports both single house and batch CSV predictions.
- **Pre-trained Model:** Ready-to-use model included for instant results.

---

## 📁 Project Structure

```
house-price-prediction/
│
├── backend/
│   ├── app.py                # Main Flask API server
│   ├── main.py               # Model loading and prediction logic
│   ├── processed/            # Saved models and processed files
│   └── uploads/              # Uploaded CSVs for batch prediction
│
├── data/
│   ├── housing.csv           # Example dataset
│   └── output.csv            # Example output
│
├── frontend/
│   ├── index.html            # Main web interface
│   ├── style.css             # Custom styles
│   └── README.txt            # Frontend usage notes
│
├── README.md                 # Project documentation
└── ...
```

---

## 🛠️ Setup & Installation

1. **Clone the repository:**
	```sh
	git clone https://github.com/ALLI-CHIRANJEEVI/house-price-prediction.git
	cd house-price-prediction
	```

2. **Install backend dependencies:**
	```sh
	cd backend
	pip install -r requirements.txt
	```

3. **Run the backend server:**
	```sh
	python app.py
	```

4. **Open the frontend:**
	- Open `frontend/index.html` in your browser.

---

## 🧑‍💻 Usage

1. **Single Prediction:**
	- Enter house features in the web form and click **Predict**.

2. **Batch Prediction:**
	- Upload a CSV file with house data and download the predictions.

---

## 📊 Example

**Input:**
| Feature 1 | Feature 2 | ... |
|-----------|-----------|-----|
| value     | value     | ... |

**Output:**
| Predicted Price |
|-----------------|
| $123,456        |

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [scikit-learn](https://scikit-learn.org/)
- [Flask](https://flask.palletsprojects.com/)
- [HTML5 UP](https://html5up.net/) (for design inspiration)

---

<p align="center">
  <b>Happy Predicting! 🏠✨</b>
</p>
