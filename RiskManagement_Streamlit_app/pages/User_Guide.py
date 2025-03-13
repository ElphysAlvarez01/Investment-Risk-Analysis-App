import streamlit as st

# ---- ADD A BANNER IMAGE ----
st.image("../Image_folder/user_guide.png", use_container_width=True)

def main():
    # Page Title
    st.title("📖 User Guide")
    st.subheader("Welcome to the Risk Management Analysis Tool!")
    
    # Introduction
    st.markdown("""
    This guide will help you navigate through the **Risk Management Analysis Tool**, explaining its features and how to use them effectively.

    **Main Features:**
    - 📊 **Risk Analysis Dashboard** – Simulate different trading scenarios and assess risk.
    - 📘 **Trading Journal** – Import and analyze trade data with key performance indicators.
    - ⚠️ **Risk Dashboard** – Monitor risk and drawdown metrics.
    
    Use the navigation on the left to switch between sections.
    """)

    # Section 1: Getting Started
    st.markdown("---")
    st.header("🚀 Getting Started")
    st.markdown("""
    1. **Navigate the App:**  
       - Use the sidebar to access different sections.
       - Click on **"Risk Analysis Dashboard"** to run simulations.
       - Use the **"Trading Journal"** to upload and analyze your trade logs.
    
    2. **Adjust Risk Parameters:**  
       - Modify key parameters such as **account balance, position size, number of trades, and win rate**.
       - Compare different **trading scenarios** side by side.
    
    3. **Visualize Risk Metrics:**  
       - View **capital growth, drawdown charts, box plots, and bell curves**.
       - Analyze **risk-adjusted returns** using Monte Carlo simulations.
    """)

    # Section 2: Using the Trading Journal
    st.markdown("---")
    st.header("📊 Using the Trading Journal")
    st.markdown("""
    The **Trading Journal** allows you to import your trade history for risk analysis and performance tracking.

    **Steps to Use the Trading Journal:**
    1. **Navigate to the "Trading Journal" tab.**
    2. **Upload your trade history as a CSV file.**
    3. The system will **automatically calculate key performance indicators (KPIs)** including:
        - Win Rate (%)
        - Maximum Drawdown (%)
        - Cumulative Profit & Loss ($)
        - Sharpe Ratio (Risk-Adjusted Return)
        - Average Risk-Reward Ratio

    4. Visualize your trading performance using **charts and statistical summaries**.
    """)

    # Section 3: CSV File Requirements
    st.markdown("---")
    st.header("📂 CSV File Requirements")
    st.markdown("""
    To ensure accurate analysis, your CSV file **must** contain the following columns:

    | Column Name         | Description |
    |---------------------|-------------|
    | **Date**           | The date of the trade (YYYY-MM-DD format). |
    | **Symbol**         | The stock or asset traded (e.g., AAPL, BTC, SPY). |
    | **Entry Price**    | The price at which the trade was entered. |
    | **Exit Price**     | The price at which the trade was exited. |
    | **Position Size**  | The number of shares/contracts traded. |
    | **Profit/Loss ($)** | The net profit or loss for the trade. |
    | **Trade Type**     | Whether the trade was a **Buy** or **Sell**. |
    
    **Example CSV File Format:**
    
    ```
    Date,Symbol,Entry Price,Exit Price,Position Size,Profit/Loss ($),Trade Type
    2024-01-10,AAPL,150.00,155.00,100,500,Buy
    2024-01-15,TSLA,700.00,680.00,50,-1000,Sell
    ```
    
    - Ensure **no missing values** in key columns.
    - Use **comma-separated values (.csv) format**.
    - The system will ignore any additional columns not listed above.
    """)

    # Section 4: Risk Management Dashboard
    st.markdown("---")
    st.header("⚠️ Risk Dashboard")
    st.markdown("""
    - Assess **risk exposure** with key performance indicators.
    - Monitor **cumulative drawdowns and volatility trends**.
    - Compare **different trading strategies** under real-world market conditions.
    """)

    # Section 5: Best Practices
    st.markdown("---")
    st.header("📌 Best Practices")
    st.markdown("""
    - Adjust parameters to simulate different **market conditions**.
    - Keep an updated **Trading Journal** to track performance over time.
    - Use **Monte Carlo simulations** to estimate risk probability.
    - Monitor **Sharpe ratios and max drawdowns** to optimize strategy performance.
    """)

    # Section 6: FAQ
    st.markdown("---")
    st.header("❓ Frequently Asked Questions (FAQ)")
    
    with st.expander("1️⃣ How do I upload my trade history?"):
        st.write("Go to the 'Trading Journal' tab, click 'Upload CSV', and select your trade history file.")

    with st.expander("2️⃣ What risk parameters should I adjust?"):
        st.write("You can modify position size, number of trades, average gain/loss, and win rate to model different trading strategies.")

    with st.expander("3️⃣ What is a Monte Carlo simulation?"):
        st.write("Monte Carlo simulations use random sampling to estimate risk outcomes based on different scenarios.")

    with st.expander("4️⃣ What happens if my CSV file has missing data?"):
        st.write("Ensure all required columns are present. Missing data may result in errors or incorrect calculations.")

    with st.expander("5️⃣ Can I analyze my past trades?"):
        st.write("Yes! The Trading Journal allows you to import your trade history and visualize performance trends.")

    # Section 7: Contact and Support
    st.markdown("---")
    st.header("📬 Contact & Support")
    st.markdown("""
    If you have questions or feedback, please reach out via:
    - 📧 **Email:** support@riskmanagementapp.com
    - 🐙 **GitHub Repo:** [Click here](https://github.com/yourrepo)
    """)

if __name__ == "__main__":
    main()

