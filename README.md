# 📦 Olist Delay Prediction – Ensemble ML Project

This project is a machine learning application designed to predict whether a product delivery will be delayed, based on order information from the [Olist e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). It uses an ensemble model to improve predictive accuracy and fairness, and includes an interactive web app built with Streamlit.

---

## 🚀 Features

- Cleaned and merged real-world logistics data
- Feature engineering (shipping time, encoded categories)
- Handling of imbalanced data with class weighting
- Training of Logistic Regression, Decision Tree, and Random Forest models
- Voting Classifier ensemble model for improved performance
- Interactive Streamlit app with dropdowns and predictions

---

## 🧠 Tech Stack

- Python (Pandas, Scikit-learn, Matplotlib, Seaborn)
- Streamlit (for UI and interaction)
- Joblib (model saving)
- Jupyter Notebooks (for development and analysis)

---

## 🛠 How to Run the App

### 1. Clone this repository
```bash
git clone https://github.com/your-username/olist-delay-prediction.git
cd olist-delay-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
📦 olist-delay-prediction
├── app.py                            # Streamlit app
├── voting_ensemble_model.pkl        # Saved ensemble model
├── Olist_Cleaned_Real_Dataset.csv   # Cleaned dataset with labels + encodings
├── 01_olist_full_pipeline.ipynb     # Notebook with cleaning, EDA, modeling
├── requirements.txt
└── README.md
```

---

## 📊 Model Performance

| Model              | Accuracy | Notes |
|-------------------|----------|-------|
| Logistic Regression | 92%      | High accuracy but poor recall for delays |
| Decision Tree       | 89%      | Better at catching delay cases |
| Random Forest       | 91%      | Balanced but overfitting possible |
| **Voting Classifier** | ✅ **Best** | Best tradeoff in fairness and performance |

---

## 🙌 Team & Acknowledgments

Developed by Luana, Angie, Jhonatan, Juan and David for a final Business Analytics project.  
Powered by open datasets from Kaggle and open-source tools.

---

## 📬 Contact

Have questions or feedback? Reach out via lambton email.
