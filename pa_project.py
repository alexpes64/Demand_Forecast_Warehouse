"""
Product demand analysis and prediction module.
This module loads data, visualizes statistics, and performs
predictions via a RandomForest model.
"""
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Page configuration
st.set_page_config(page_title="Demand Analysis & Prediction", layout="wide")
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. LOADING AND CLEANING
# ==============================================================================


@st.cache_data
def load_data():
    """
    Loads and cleans data from the CSV file.
    Returns a Pandas DataFrame.
    """
    # Hardcoded path split to respect character limit
    file_path = (
        "C:/Users/alexp/OneDrive/Bureau/TBS/M2/S1/"
        "advenced python/Projet/venv/Historical Product Demand.csv"
    )

    # Direct loading without error handling as requested
    df = pd.read_csv(file_path)

    # Cleaning
    df = df.dropna(subset=['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    df['Order_Demand'] = pd.to_numeric(df['Order_Demand'], errors='coerce')
    df = df.dropna(subset=['Order_Demand'])

    # Filter > 2011
    df = df[df['Date'] > '2011-12-31']

    return df

# ==============================================================================
# 2. MACHINE LEARNING LOGIC
# ==============================================================================


def _add_features(df_in):
    """Adds temporal features to the DataFrame."""
    min_date = df_in['Date'].min()
    df_in['Time'] = (df_in['Date'] - min_date).dt.days
    df_in['Month'] = df_in['Date'].dt.month
    df_in['DayOfWeek'] = df_in['Date'].dt.dayofweek
    df_in['DayOfYear'] = df_in['Date'].dt.dayofyear
    return df_in, ['Time', 'Month', 'DayOfWeek', 'DayOfYear'], min_date


def _train_and_evaluate(target_df, features):
    """Splits data, trains the model, and evaluates performance."""
    split_index = int(len(target_df) * 0.85)
    x_train = target_df[features].iloc[:split_index]
    y_train = target_df['Order_Demand'].iloc[:split_index]
    x_test = target_df[features].iloc[split_index:]
    y_test = target_df['Order_Demand'].iloc[split_index:]
    dates_test = target_df['Date'].iloc[split_index:]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)

    y_pred_test = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return model, rmse, dates_test, y_pred_test


def _predict_future(model, last_date, min_date, features):
    """Generates predictions for the next 30 days."""
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 31)]
    future_df = pd.DataFrame({'Date': future_dates})

    # Recalculate features for the future
    future_df['Time'] = (future_df['Date'] - min_date).dt.days
    future_df['Month'] = future_df['Date'].dt.month
    future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
    future_df['DayOfYear'] = future_df['Date'].dt.dayofyear

    future_forecast = model.predict(future_df[features])
    return future_dates, future_forecast


def run_prediction(df, warehouse, category):
    """
    Trains a RandomForest model for a given warehouse and category.
    Returns results and metrics.
    """
    # Filtering
    target_df = df[
        (df['Warehouse'] == warehouse) & (df['Product_Category'] == category)
    ].copy()
    target_df = target_df.sort_values('Date')

    # Outliers (inline to save a local variable)
    target_df = target_df[
        target_df['Order_Demand'] < target_df['Order_Demand'].quantile(0.98)
    ]

    # Features
    target_df, features, min_date = _add_features(target_df)

    # Training and evaluation (via helper function)
    model, rmse, dates_test, y_pred_test = _train_and_evaluate(
        target_df, features
    )

    # Future prediction (via helper function)
    last_date = target_df['Date'].max()
    future_dates, future_forecast = _predict_future(
        model, last_date, min_date, features
    )

    return {
        'rmse': rmse,
        'target_df': target_df,
        'dates_test': dates_test,
        'y_pred_test': y_pred_test,
        'future_dates': future_dates,
        'future_forecast': future_forecast
    }

# ==============================================================================
# 3. USER INTERFACE
# ==============================================================================


def render_overview(df):
    """Displays Page 1: Overview."""
    st.title("Demand Overview")

    # 1. Global Chart (Line)
    st.subheader("Total Demand by Month")
    df_monthly = df.groupby(
        pd.Grouper(key='Date', freq='M')
    )['Order_Demand'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=df_monthly, x='Date', y='Order_Demand', color='blue', ax=ax
    )
    plt.title('Total Demand by Month')
    plt.xlabel('Date')
    plt.ylabel('Total Orders')
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.markdown("---")

    # 2. Warehouse Chart
    st.subheader("Comparison by Warehouse")
    df_by_warehouse = df.groupby(
        ['Date', 'Warehouse']
    )['Order_Demand'].sum().reset_index()

    sns.set_style("whitegrid")
    grid = sns.relplot(
        data=df_by_warehouse, x='Date', y='Order_Demand', kind="line",
        hue='Warehouse', palette=["orange", "red", "green", "purple"],
        col='Warehouse', col_wrap=2, height=3, aspect=1.5
    )
    st.pyplot(grid.fig)


def render_categories(df):
    """Displays Page 2: Categories by Warehouse."""
    st.title("Product Analysis by Warehouse")

    # Warehouse Selector
    st.sidebar.header("Filters")
    warehouses = sorted(df['Warehouse'].unique())
    selected_wh = st.sidebar.selectbox("Select a warehouse", warehouses)

    # Chart logic (Barplot)
    st.subheader(f"Top Categories for {selected_wh}")

    # Filter and Group
    wh_data = df[df['Warehouse'] == selected_wh]
    wh_grouped = wh_data.groupby(
        ['Date', 'Product_Category']
    )['Order_Demand'].sum().reset_index()
    # Sort (Top)
    wh_sorted = wh_grouped.groupby('Product_Category')['Order_Demand'].sum()
    wh_sorted = wh_sorted.sort_values(ascending=False).reset_index()

    # Dynamic color
    color_map = {
        'Whse_S': "#EF9A9A", 'Whse_A': "#673AB7",
        'Whse_C': "#039BE5", 'Whse_J': "#FB8C00"
    }
    bar_color = color_map.get(selected_wh, "blue")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=wh_sorted.head(15), x='Product_Category',
        y='Order_Demand', color=bar_color, ax=ax2
    )

    plt.title(f'Product Demand - {selected_wh}')
    plt.xlabel('Product Category')
    plt.ylabel('Total Demand')
    plt.xticks(rotation=45)
    st.pyplot(fig2)


def render_prediction(df):
    """Displays Page 3: Prediction."""
    st.title("ML Prediction - Category 019")
    st.info("Specific prediction for **Category_019** via Random Forest.")

    # Warehouse Selector
    st.sidebar.header("Configuration")
    warehouses = sorted(df['Warehouse'].unique())
    ml_wh = st.sidebar.selectbox("Select target warehouse", warehouses)

    # Button
    launch = st.sidebar.button("Start Prediction", type="primary")

    if launch:
        with st.spinner(
            f"Calculation in progress for {ml_wh} / Category_019..."
        ):
            results = run_prediction(df, ml_wh, 'Category_019')

        st.success(f"Model trained! RMSE: {results['rmse']:.2f}")

        # Final Chart
        target_df = results['target_df']
        fig_ml, ax_ml = plt.subplots(figsize=(14, 7))

        # 1. Recent History
        one_year_ago = target_df['Date'].max() - pd.Timedelta(days=365)
        subset_plot = target_df[target_df['Date'] > one_year_ago].copy()
        subset_plot['Rolling_Mean'] = subset_plot['Order_Demand'].rolling(
            window=14
        ).mean()

        sns.lineplot(
            data=subset_plot, x='Date', y='Rolling_Mean',
            color='#2980b9', label='Smoothed History (14d)',
            linewidth=2, ax=ax_ml
        )

        # 2. Test
        ax_ml.plot(
            results['dates_test'], results['y_pred_test'],
            color='green', alpha=0.7, label='Model Validation'
        )

        # 3. Future
        ax_ml.plot(
            results['future_dates'], results['future_forecast'],
            color='#c0392b', linestyle='--', linewidth=2,
            label='Future Forecast (30d)'
        )

        plt.title(f'Precise Prediction: {ml_wh} - Category_019', fontsize=16)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_ml)


def main():
    """Main function of the Streamlit application."""
    # Loading
    with st.spinner('Loading data...'):
        df = load_data()

    # --- SIDEBAR NAVIGATION ---
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to:",
        [
            "1. Overview",
            "2. Categories by Warehouse",
            "3. Prediction (Cat 019)"
        ]
    )

    st.sidebar.markdown("---")

    # Page Routing
    if page == "1. Overview":
        render_overview(df)
    elif page == "2. Categories by Warehouse":
        render_categories(df)
    elif page == "3. Prediction (Cat 019)":
        render_prediction(df)


if __name__ == '__main__':
    main()
