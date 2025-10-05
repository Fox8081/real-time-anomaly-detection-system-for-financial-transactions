# real-time-anomaly-detection-system-for-financial-transactions
The goal of this project is to build a real-time anomaly detection system for financial transactions related to government welfare schemes. The system uses a deep learning model to identify suspicious patterns that might indicate fraud and presents the findings on an interactive dashboard for auditors.
# Government Scheme Transaction Anomaly Detection Dashboard

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.48.0-orange.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-green.svg)](https://www.tensorflow.org/)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Data Generation and Simulation](#data-generation-and-simulation)
- [Troubleshooting](#troubleshooting)
- [Results](#result)
- [Contributing](#contributing)
---

## Project Overview

This project is designed to detect anomalies in government scheme transactions by leveraging deep learning techniques, specifically an autoencoder neural network model trained on synthetic and real-world-inspired transaction data. The solution includes:

- A **data simulator** generating realistic transaction data with anomalies.
- A **model training pipeline** using TensorFlow autoencoder for unsupervised detection.
- A **real-time monitoring service** to score live transactions.
- An **interactive Streamlit dashboard** providing visual insights, filters, and downloadable data.

---

## Features

- Synthetic transaction data generation with diverse anomaly types.
- Deep autoencoder for anomaly detection on tabular transactional data.
- Real-time batch scoring and status updates.
- Interactive visualizations including trends, geo-maps, and heatmaps.
- User login and download filtered data functionalities.
- Easy deployment and environment setup.

---

## Prerequisites

- Python 3.11 or higher
- Conda (recommended) or virtual environment

---

## Installation

**1. Clone the repository**

git clone https://github.com/Fox8081/real-time-anomaly-detection-system-for-financial-transactions/edit/main

cd yourproject ( location here all project files are stored)


**2. Create and activate a Python environment in anaconda**

cd " location where all project files are stored " ( if you downloaded all files )

conda create -n anomalyenv python=3.11

conda activate anomalyenv

**3. Install project dependencies**

pip install -r requirements.txt


## Project Structure

**dashboard.py** :- Streamlit dashboard app
  
**train.py** :- Model training script (autoencoder)

**monitor.py** :- Real-time anomaly scoring batch process

**live_feed_simulator.py** :- Synthetic transaction data simulator

**generate_database.py** :- Bulk transaction data generation

**requirements.txt** :- Project dependencies

**transactions.db** :- SQLite transaction database (generated)

**modeling/** :- Model and scaler artifacts
      
      autoencoder_model.h5
      
      scaler.pkl
      
      model_config.json
      
**README.md**  



## Usage

### Generate Bulk Data

python generate_database.py

### Run Real-time Simulator

python live_feed_simulator.py

### Train the Model

python train.py (if you have  autoencoder_model.h5 , scaler.pkl, model_config.jso not need to run this file)

### Start the Anomaly Monitor

python monitor.py

### Run Dashboard

streamlit run dashboard.py

## Model Architecture

- The model is a **deep autoencoder** neural network composed of fully connected (Dense) layers.
- Trained on normal transactions to learn their data distribution.
- Anomalies detected by high reconstruction error indicating deviation from learned patterns.
- Feature engineering includes amount, log(amount), time features (hour, day), and normalized location.

---

## Data Generation and Simulation

- Synthetic data simulates normal and anomalous transactions according to Indian government scheme scenarios.
- Anomaly types include unusual amounts, off-hour transactions, frequent bursts, and collusion rings.
- Geographic locations are constrained within Indian bounding boxes with some outliers for testing.

---

## Troubleshooting

- **Model loading errors:**       Retrain model using `train.py`.
- **Dashboard blank or errors:**  Check simulator and monitor are running; verify database connectivity.
- **Dependency issues:**          Reinstall with `pip install -r requirements.txt`.
- **Database issues:**            Delete or archive `transactions.db` and regenerate data.
- **Login errors:**               Verify credentials in `dashboard.py`.

---
## Result 


## Contributing

Contributions are welcome!  
Please fork the repo, create feature branches, and submit pull requests.  
For major changes, open an issue first to discuss your plans.
