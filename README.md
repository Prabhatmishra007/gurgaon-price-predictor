# 🏙️ Gurgaon Property Price Predictor

A complete end-to-end machine learning project that predicts residential property prices in Gurgaon, Haryana. Built with Python and deployed as an interactive **Streamlit** web app powered by a **Random Forest** model trained on real Gurgaon real estate data.

---

## 📸 Demo

> Run the app locally and visit `http://localhost:8501`

---

## 📁 Project Structure

```
price_predictor_clean/
│
├── 📓 Notebooks
│   ├── data-preprocessing-flats.ipynb          # Preprocess flat listings
│   ├── data-preprocessing-houses.ipynb         # Preprocess house listings
│   ├── data-preprocessing-level-2.ipynb        # Secondary cleaning
│   ├── merge-flats-and-house.ipynb             # Merge both datasets
│   ├── feature-engineering.ipynb               # Feature creation
│   ├── outlier-treatment.ipynb                 # Outlier removal
│   ├── missing-value-imputation.ipynb          # Handle missing values
│   ├── feature-selection.ipynb                 # Select best features
│   ├── model-selection.ipynb                   # Train & compare models
│   ├── eda-univariate-analysis.ipynb           # Univariate EDA
│   ├── eda-multivariate-analysis.ipynb         # Multivariate EDA
│   └── eda-pandas-profiling.ipynb              # Auto EDA report
│
├── 📊 Data Files
│   ├── flats.csv                               # Raw flats data
│   ├── houses.csv                              # Raw houses data
│   ├── appartments.csv                         # Apartments reference data
│   ├── flats_cleaned.csv                       # Cleaned flats
│   ├── house_cleaned.csv                       # Cleaned houses
│   ├── gurgaon_properties.csv                  # Merged dataset
│   ├── gurgaon_properties_cleaned_v1.csv       # After level-2 cleaning
│   ├── gurgaon_properties_cleaned_v2.csv       # After feature engineering
│   ├── gurgaon_properties_outlier_treated.csv  # After outlier treatment
│   ├── gurgaon_properties_missing_value_imputation.csv
│   └── gurgaon_properties_post_feature_selection_v2.csv
│
├── 🤖 Model
│   ├── pipeline.pkl                            # Trained ML pipeline (gitignored)
│   └── df.pkl                                  # Feature dataframe
│
├── 🌐 App
│   └── app.py                                  # Streamlit web app
│
├── .gitignore
└── README.md
```

---

## 🔄 ML Pipeline

```
Raw Data (flats.csv + houses.csv)
        ↓
Data Preprocessing (cleaning, merging)
        ↓
Feature Engineering (new features from appartments.csv)
        ↓
Outlier Treatment
        ↓
Missing Value Imputation
        ↓
EDA (Univariate + Multivariate)
        ↓
Feature Selection
        ↓
Model Training (Random Forest - 500 trees)
        ↓
Streamlit App (app.py)
```

---

## 🧠 Model Details

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Number of Trees | 500 |
| Target | Property Price (in Crores) |
| Target Transformation | log1p (log transform) |
| Encoding | OrdinalEncoder + OneHotEncoder |
| Scaling | StandardScaler |
| Training Data | 3,554 properties |
| Features | 12 |

**Features used for prediction:**
- Property Type, Sector, Bedrooms, Bathrooms, Balconies
- Built-up Area, Age/Possession, Furnishing Type
- Luxury Category, Floor Category, Servant Room, Store Room

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gurgaon-price-predictor.git
cd gurgaon-price-predictor

# Install dependencies
pip install streamlit scikit-learn pandas numpy
```

### Run the App

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> ⚠️ `pipeline.pkl` is excluded from the repo (>100MB). To regenerate it, run all cells in `model-selection.ipynb`.

---

## 📦 Regenerating the Model

If `pipeline.pkl` is missing, run the notebooks in this order:

1. `data-preprocessing-flats.ipynb`
2. `data-preprocessing-houses.ipynb`
3. `merge-flats-and-house.ipynb`
4. `data-preprocessing-level-2.ipynb`
5. `feature-engineering.ipynb`
6. `outlier-treatment.ipynb`
7. `missing-value-imputation.ipynb`
8. `feature-selection.ipynb`
9. `model-selection.ipynb` ← saves `pipeline.pkl` and `df.pkl`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| pandas & NumPy | Data manipulation |
| scikit-learn | ML pipeline & model |
| Streamlit | Web app deployment |
| Jupyter Notebook | Analysis & EDA |
| Matplotlib & Seaborn | Visualization |

---

## 📊 Dataset

The dataset contains real estate listings scraped from Gurgaon property portals, covering **104 sectors** across Gurgaon and surrounding areas including Dwarka Expressway, Sohna Road, Manesar, and Gwal Pahari.

---

## 👤 Author

Prabhat Mishra


---

## 📄 License

This project is open source and available under the [MIT License]
