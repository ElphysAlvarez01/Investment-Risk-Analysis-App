import streamlit as st
import pandas as pd
import plotly.express as px

# ---- ADD A BANNER IMAGE ----
st.image("../Image_folder/trading_journal.png", use_container_width=True)

def main():
    st.title("📊 Trade Performance Tracker")

    # ---- FILE UPLOAD ----
    uploaded_file = st.file_uploader("📂 Upload your trade history (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Load Data
        if uploaded_file.name.endswith(".csv"):
            trades_df = pd.read_csv(uploaded_file)
        else:
            trades_df = pd.read_excel(uploaded_file)

        # ---- VALIDATE COLUMN NAMES ----
        expected_columns = ["Date", "Ticker", "Shares", "Entry Price", "Exit Price"]
        
        if not all(col in trades_df.columns for col in expected_columns):
            st.error(f"❌ Invalid file format. Please make sure your file contains these columns: {expected_columns}")
        else:
            # Convert Date to datetime format
            trades_df["Date"] = pd.to_datetime(trades_df["Date"])

            # Calculate PnL for each trade
            trades_df["Return ($)"] = (trades_df["Exit Price"] - trades_df["Entry Price"]) * trades_df["Shares"]
            trades_df["Return (%)"] = ((trades_df["Exit Price"] - trades_df["Entry Price"]) / trades_df["Entry Price"]) * 100

            # Compute Capital Growth
            trades_df["Cumulative Capital"] = 10000 + trades_df["Return ($)"].cumsum()

            # ---- CAPITAL GROWTH CHART ----
            fig = px.line(trades_df, x="Date", y="Cumulative Capital", title="📊 Capital Growth Over Time",
                          markers=True, labels={"Date": "Date", "Cumulative Capital": "Capital ($)"})
            st.plotly_chart(fig)

            # ---- RISK ANALYSIS ----
            max_drawdown = ((trades_df["Cumulative Capital"] - trades_df["Cumulative Capital"].cummax()) / trades_df["Cumulative Capital"].cummax()).min() * 100
            sharpe_ratio = trades_df["Return (%)"].mean() / trades_df["Return (%)"].std()

            # Display Risk Summary
            risk_summary = pd.DataFrame({
                "Max Drawdown (%)": [max_drawdown],
                "Sharpe Ratio": [sharpe_ratio]
            })

            st.write("📉 **Risk vs Performance Analysis**")
            st.dataframe(risk_summary)

if __name__ == "__main__":
    main()

           
