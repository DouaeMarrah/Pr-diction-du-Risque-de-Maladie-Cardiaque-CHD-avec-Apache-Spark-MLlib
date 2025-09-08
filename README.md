# Heart Disease Prediction (CHD) Project

This project implements a machine learning pipeline to predict the 10-year risk of Coronary Heart Disease (CHD) using patient medical data. Built with **Apache Spark** and **Java**, it demonstrates a complete workflow from data loading and preprocessing to model training, evaluation, and visualization.

## Project Overview

Coronary Heart Disease is a leading cause of mortality worldwide. This project aims to identify key risk factors and build a predictive model to assess an individual's risk of developing CHD within a decade. The solution is designed to be scalable for large datasets using distributed computing with Spark.

##  Tech Stack & Architecture

- **Language:** Java 8
- **Big Data Framework:** Apache Spark 3.2.0 (Core, SQL, MLlib)
- **Data Visualization:** JFreeChart 1.5.3
- **Build Tool:** Maven
- **Logging:** SLF4J with Log4j


##  Dataset

The project uses the Framingham Heart Study dataset, containing medical attributes for patients. The target variable is `TenYearCHD`, a binary indicator (1: Yes, 0: No).

**Key Features:**
- `age`, `sex`, `is_smoking`, `cigsPerDay`
- `BPMeds`, `prevalentHyp`, `diabetes`
- `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose`

##  Methodology

1.  **Data Preparation:**
    - Load and infer schema from CSV files.
    - Handle missing values by imputation with median.
    - Encode categorical variables (e.g., sex, smoking status).

2.  **Exploratory Data Analysis (EDA):**
    - Generate descriptive statistics.
    - Create visualizations for key features vs. CHD risk (Age, Sex, Smoking Status, etc.).

3.  **Modeling:**
    - **Algorithm:** Logistic Regression.
    - **Pipeline:** Feature assembly, scaling, and model training.
    - **Validation:** 5-Fold Cross-Validation for hyperparameter tuning.
    - **Evaluation Metric:** Accuracy.

4.  **Prediction & Results:**
    - Generate predictions on a test set.
    - Export results to a CSV file.
    - Analyze and visualize prediction distributions.

##  Key Features

- **Scalable Processing:** Leverages Apache Spark for distributed data processing.
- **Robust Pipeline:** Includes comprehensive data cleaning, transformation, and modeling.
- **Interactive Visualizations:** Uses JFreeChart to create graphs for data and result analysis.
- **Reproducible:** Cross-validation ensures model reliability and generalizability.

## Results

The logistic regression model achieved an **accuracy of 86.6%** on the test dataset.

**Prediction Distribution on Test Set:**
- **Class 0 (No CHD Risk):** 840 cases
- **Class 1 (CHD Risk):** 8 cases

Visual analysis confirmed expected trends: predicted risk (red bars) increased with age, smoking, and other known clinical factors.

![Age Analysis](images/age_analysis.png)

## 🚀 How to Run

1.  **Prerequisites:**
    - Java 8 JDK
    - Apache Spark 3.2.0
    - Maven

2.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd projet-chd
    ```

3.  **Build the project:**
    ```bash
    mvn clean package
    ```

4.  **Run the application:**
    ```bash
    spark-submit --class spark.batch.Main target/projet-chd-1.0-SNAPSHOT.jar
    ```


##  Future Improvements

- **Model Diversity:** Experiment with other algorithms (Random Forest, Gradient Boosting, SVM).
- **Enhanced Metrics:** Incorporate AUC-ROC, Precision, Recall, and F1-Score for a better evaluation of class imbalance.
- **Advanced Feature Engineering:** Create new features and perform more sophisticated selection.
- **Deployment:** Package the project for easier deployment and integrate a simple web interface for predictions.
