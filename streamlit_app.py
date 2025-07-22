# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

st.title("GL Outlier Detection Agent")

# Upload GL Data
uploaded_file = st.file_uploader("Upload GL CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview", df.head())

    test = st.selectbox("Select a test to run", [
        "Unusual Transaction Amounts",
        "Duplicate or Near-Duplicate Entries",
        "Anomalous Vendor or Customer Activity",
        "Rare GL Code Combinations",
        "Transactions Outside Policy Threshold"
    ])

    if test == "Unusual Transaction Amounts":
        st.subheader("Detecting Unusual Transaction Amounts using Isolation Forest")
        X = df[['amount']].fillna(0)
        model = IsolationForest(contamination=0.01, random_state=42)
        df['anomaly_score'] = model.fit_predict(X)
        result = df[df['anomaly_score'] == -1]
        st.write("Anomalies Found:", result)

    elif test == "Duplicate or Near-Duplicate Entries":
        st.subheader("Detecting Duplicate or Near-Duplicate Entries")
        subset_cols = ['vendor_id', 'amount', 'transaction_date']
        df['duplicate_count'] = df.duplicated(subset=subset_cols, keep=False)
        result = df[df['duplicate_count']]
        st.write("Duplicate Entries Found:", result)

    elif test == "Anomalous Vendor or Customer Activity":
        st.subheader("Detecting Anomalous Vendor Activity using Isolation Forest")
        grouped = df.groupby('vendor_id')['amount'].mean().reset_index()
        model = IsolationForest(contamination=0.05, random_state=42)
        grouped['anomaly'] = model.fit_predict(grouped[['amount']])
        outliers = grouped[grouped['anomaly'] == -1]['vendor_id']
        result = df[df['vendor_id'].isin(outliers)]
        st.write("Vendor-based Anomalies:", result)

    elif test == "Rare GL Code Combinations":
        st.subheader("Detecting Rare GL Code Combinations")
        df['combo'] = df['gl_code'].astype(str) + '_' + df['vendor_id'].astype(str)
        combo_counts = df['combo'].value_counts()
        rare_combos = combo_counts[combo_counts < 3].index
        result = df[df['combo'].isin(rare_combos)]
        st.write("Rare Combinations Found:", result)

    elif test == "Transactions Outside Policy Threshold":
        st.subheader("Flagging Entries Exceeding Policy Limits (Z-score)")
        amount_mean = df['amount'].mean()
        amount_std = df['amount'].std()
        df['z_score'] = (df['amount'] - amount_mean) / amount_std
        result = df[np.abs(df['z_score']) > 3]
        st.write("Out-of-Threshold Transactions:", result)
