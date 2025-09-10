# Hotel Booking Cancellation Prediction 🏨

This repository contains the project for the **CSE422: Artificial Intelligence** course (Section 04, Summer 2025). The project focuses on developing a predictive model to forecast hotel booking cancellations using machine learning.

**Group 14 Members:**
* Jotee Sarkar Joy (ID: 22301040)
* Istiak Al Imran (ID: 22301001)

---

## 📝 Table of Contents
* [Introduction](#-introduction)
* [Dataset](#-dataset)
* [Methodology](#-methodology)
* [Getting Started](#-getting-started)
* [Results](#-results)
* [Work Distribution](#-work-distribution)
* [Conclusion](#-conclusion)

---

## 🧐 Introduction

The primary goal of this project is to develop a robust predictive model to forecast hotel booking cancellations. By accurately identifying bookings that are likely to be canceled, hotels can better optimize room allocation, manage revenue streams, and implement effective customer retention strategies. This project leverages machine learning to provide actionable insights from historical booking data.

---

## 📊 Dataset

The dataset used for this project contains **119,390 booking records** with **32 distinct features**. This is a **binary classification problem** where the goal is to predict the `is_canceled` feature (1 for canceled, 0 for not canceled).

* **Key Features:** `lead_time`, `adr`, `hotel`, `country`, `previous_cancellations`.
* **Data Imbalance:** The dataset is imbalanced, with **37.04%** of bookings being canceled and **62.96%** not canceled. This was addressed using stratified splitting to maintain class distribution during training and testing.

---

## 🤖 Methodology

The project followed a structured machine learning workflow:

1.  **Data Pre-processing:**
    * Handled missing values by dropping the `company` column and imputing values for `agent`, `country`, and `children`.
    * Prevented data leakage by removing the `reservation_status` and `reservation_status_date` columns.
    * Removed invalid records where the total number of guests was zero.
    * Normalized numerical features using `StandardScaler`.

2.  **Dataset Splitting:**
    * The data was split into an **80% training set** and a **20% test set** using a stratified approach.

3.  **Model Training & Testing:**
    * **Supervised Models:**
        * Logistic Regression
        * Decision Tree Classifier
        * K-Nearest Neighbors (KNN)
        * Neural Network (MLPClassifier)
    * **Unsupervised Model:**
        * K-Means Clustering

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Ensure you have Python 3.x installed on your system. You will need the following libraries:
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/joysarkar077/CSE422-Lab-Project-Hotel-Booking-Cancellation-Prediction-Summer-2024.git](https://github.com/joysarkar077/CSE422-Lab-Project-Hotel-Booking-Cancellation-Prediction-Summer-2024.git)
    ```
2.  **Navigate to the project directory:**
    ```sh
    cd CSE422-Lab-Project-Hotel-Booking-Cancellation-Prediction-Summer-2024
    ```
3.  **Install the required packages (recommended):**
    ```sh
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```
4.  **Run the Jupyter Notebook:**
    ```sh
    jupyter notebook "Your_Notebook_Name.ipynb"
    ```

---

## 📈 Results

The performance of the supervised models was compared based on Accuracy, Precision, and Recall. The **Neural Network (MLP)** was selected as the best model due to its superior performance across all metrics.

| Model                  | Accuracy          | Precision         | Recall            |
| ---------------------- | ----------------- | ----------------- | ----------------- |
| Logistic Regression    | 80.33%            | 77.88%            | 65.54%            |
| Decision Tree          | 85.49%            | 84.07%            | 77.66%            |
| K-Nearest Neighbors    | 83.59%            | 79.63%            | 76.75%            |
| **Neural Network (MLP)** | **87.15%** | **84.27%** | **82.42%** |

The Neural Network also achieved the highest AUC score, confirming its effectiveness in distinguishing between canceled and non-canceled bookings.

---

## ✨ Conclusion

This project successfully demonstrates the application of machine learning in the hospitality industry. The **Neural Network (MLP Classifier)** proved to be a highly effective model for predicting hotel booking cancellations, achieving an accuracy of **87.15%**. The insights from this model can help hotels minimize revenue loss and improve operational efficiency.
