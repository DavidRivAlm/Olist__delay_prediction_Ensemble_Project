import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# LOAD MODEL AND DATASET
# ---------------------------

# Load the trained Voting Classifier model
model = joblib.load('voting_ensemble_model.pkl')

# Load the full cleaned dataset with labels and encoded values
df = pd.read_csv('Olist_Cleaned_Real_Dataset.csv')

# ---------------------------
# CREATE LABEL TO ENCODED MAPPINGS
# ---------------------------

# Create mapping for customer states
state_mapping = dict(
    zip(
        df['customer_state'].astype('category').cat.categories,
        df['customer_state'].astype('category').cat.codes
    )
)

# Create mapping for product categories
category_mapping = dict(
    zip(
        df['product_category_name'].astype('category').cat.categories,
        df['product_category_name'].astype('category').cat.codes
    )
)

# ✅ Create mapping for seller states
seller_mapping = dict(
    zip(
        df['seller_state'].astype('category').cat.categories,
        df['seller_state'].astype('category').cat.codes
    )
)

# ---------------------------
# STREAMLIT APP UI
# ---------------------------

# Title and instructions
st.title("📦 Olist Shipping Delay Prediction")
st.write("Use the form below to estimate whether an order will be delayed or arrive on time.")

# Shipping days
shipping_days = st.slider(
    "Estimated Shipping Days",
    1, 30, 7,
    help="How many days were promised to the customer (usually 1–30)."
)

# Product weight
product_weight_g = st.number_input(
    "Product Weight (grams)",
    min_value=100, max_value=10000, value=1000,
    help="Enter the product's weight (100–10,000g)."
)

# Dropdown: Customer State
state_label = st.selectbox("Customer State", list(state_mapping.keys()))
customer_state_encoded = state_mapping[state_label]

# Dropdown: Product Category
category_label = st.selectbox("Product Category", list(category_mapping.keys()))
product_category_encoded = category_mapping[category_label]

# ✅ Dropdown: Seller State
seller_label = st.selectbox("Seller State", list(seller_mapping.keys()))
seller_state_encoded = seller_mapping[seller_label]

# ---------------------------
# PREDICTION
# ---------------------------

if st.button("Predict Delay"):
    input_data = pd.DataFrame([{
        'shipping_days': shipping_days,
        'product_weight_g': product_weight_g,
        'customer_state_encoded': customer_state_encoded,
        'product_category_encoded': product_category_encoded,
        'seller_state_encoded': seller_state_encoded
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ This order is likely to be DELAYED.")
    else:
        st.success("✅ This order is likely to arrive ON TIME.")

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("Project by Us - Olist Delay Prediction | Powered by Streamlit")
