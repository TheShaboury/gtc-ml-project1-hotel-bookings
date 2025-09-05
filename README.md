# Hotel Booking Cancellations – Data Cleaning & Preprocessing

## 📌 Project Overview
This project focuses on building a robust **data preprocessing pipeline** for hotel booking cancellation prediction.  

The **business problem**: Last-minute booking cancellations significantly impact hotel profitability. While the end goal is to build a prediction model, this project’s main task is **preparing the raw dataset** into a clean, machine-learning-ready format.

Dataset used: `hotel_bookings.csv`  
Size: ~119k records, 32 features  

---

## 🏗️ Project Phases

### 1. Exploratory Data Analysis (EDA)
- Generated **summary statistics** (`.describe()`, `.info()`).
- Identified missing values across 4 columns:
  - `company` (94.3%), `agent` (13.7%), `country` (0.41%), `children` (0.003%).
- Detected **outliers** in:
  - `adr` (3.18%), `lead_time` (2.52%).
- Visualized missingness (heatmap, bar plots).
- Dataset overview:
  - Records: 119,390
  - Features: 32
  - Target: `is_canceled`

### 2. Data Cleaning
- **Missing values**:
  - `company` → filled with `"None"`
  - `agent` → filled with `"None"`
  - `country` → imputed with mode (`PRT`)
  - `children` → imputed with median (`0`)
- **Duplicates**:
  - Removed 32,013 duplicate rows
- **Outliers**:
  - Capped `adr` values above 212
- **Data types**:
  - Converted `reservation_status_date` to `datetime`
  - Ensured categorical features are strings
  - Ensured numerical features are numeric
- Final cleaned dataset:
  - Shape: `(87,377, 32)`
  - Missing values: 0
  - Duplicates: 0

### 3. Feature Engineering & Preprocessing
- Created new features:
  - `total_guests = adults + children + babies`
  - `total_nights = stays_in_weekend_nights + stays_in_week_nights`
  - `is_family = (children > 0 or babies > 0)`
- Encoded categorical variables:
  - **Low-cardinality** (≤10 unique) → One-Hot Encoding
  - **High-cardinality** (>10 unique) → Frequency grouping + Label Encoding
- Removed **data leakage**:
  - Dropped `reservation_status_date`
- Final dataset shape: `(87,377, 58)`
- Target distribution: `{0: 63,353 (Not Canceled), 1: 24,024 (Canceled)}`
- **Feature scaling**:
  - Applied `StandardScaler` to numerical columns

---

## 🛠️ Tech Stack
- **Languages**: Python 3
- **Libraries**:  
  - Data manipulation: `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`  
  - Preprocessing: `scikit-learn`  

---

## 📂 Repository Structure
```
├── Project_1.ipynb      # Main Colab notebook with full workflow
├── hotel_bookings.csv   # Dataset (large, may need to download separately)
├── GTC ML Project 1.pdf # Project description & requirements
└── README.md            # Project documentation
```

---

## 🚀 How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/hotel-booking-cancellations.git
   cd hotel-booking-cancellations
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:
   ```bash
   jupyter notebook Project_1.ipynb
   ```
4. Run all cells to reproduce results.

---

## 📊 Results
- Successfully cleaned and preprocessed the dataset.
- Engineered meaningful new features.
- Produced a **ML-ready dataset** for future modeling tasks.

---

## 📌 Next Steps
- Train machine learning models (e.g., Logistic Regression, Random Forest, XGBoost).
- Evaluate model performance using metrics such as **accuracy, precision, recall, F1-score, ROC-AUC**.
- Deploy the pipeline for real-time hotel booking cancellation prediction.

---

## 👤 Author
**Ahmed Shaboury**  
