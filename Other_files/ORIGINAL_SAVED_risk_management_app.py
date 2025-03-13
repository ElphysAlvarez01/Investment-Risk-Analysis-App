import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# ---- EXPAND PAGE WIDTH ----
st.set_page_config(layout="wide")  

# ---- ADD A BANNER IMAGE ----
st.image("../Image_folder/banner.png", use_container_width=True)

# ---- PAGE TITLE ----
st.title("📈 Risk Management Analysis Tool")

# Define colors for each scenario (consistent across all charts)
colors = ["#d00000", "#29bf12", "#008bf8", "#fb5607"]  # Red, Green, Blue, Orange

# ---- UI SEPARATION ----
st.markdown("---")
st.subheader("⚙️ Adjust Risk Parameters for Each Scenario")

# ---- SCENARIO INPUT SLIDERS ----
col1, col2, col3, col4 = st.columns(4)

# Function to create input widgets for a scenario
def scenario_inputs(col, scenario_name, color):
    col.markdown(f"<h3 style='color:{color};'>{scenario_name}</h3>", unsafe_allow_html=True)
    return {
        "account_size": col.number_input(f"💵 Initial Account Balance ($)", min_value=1000, value=10000, key=f"account_{scenario_name}"),
        "position_size_pct": col.number_input(f"💰 Position Size (% of Account)", min_value=1, max_value=100, value=2, key=f"position_{scenario_name}"),
        "num_trades": col.number_input(f"🔄 Number of Trades", min_value=10, max_value=1000, value=50, key=f"trades_{scenario_name}"),
        "avg_gain": col.number_input(f"📈 Avg Gain (%)", min_value=1, max_value=100, value=10, key=f"gain_{scenario_name}"),
        "avg_loss": col.number_input(f"📉 Avg Loss (%)", min_value=1, max_value=100, value=5, key=f"loss_{scenario_name}"),
        "batting_avg": col.number_input(f"🎯 Win Rate (%)", min_value=10, max_value=100, value=55, key=f"winrate_{scenario_name}")
    }

# Get user inputs for three scenarios
scenario1 = scenario_inputs(col1, "Scenario 1", colors[0])
scenario2 = scenario_inputs(col2, "Scenario 2", colors[1])
scenario3 = scenario_inputs(col3, "Scenario 3", colors[2])
scenario4 = scenario_inputs(col4, "Scenario 4", colors[3])

st.markdown("---")
st.subheader("📊 Scenario Performance Metrics")

# ---- FUNCTION FOR RISK MANAGEMENT SIMULATION ----
def risk_management_analysis(params):
    np.random.seed(42)
    capital = params["account_size"]
    trade_results = []
    drawdowns = []
    
    for trade in range(1, params["num_trades"] + 1):
        position_size = (params["position_size_pct"] / 100) * capital
        trade_outcome = np.random.choice([1, 0], p=[params["batting_avg"] / 100, 1 - (params["batting_avg"] / 100)])
        trade_return = ((params["avg_gain"] / 100) * position_size) if trade_outcome == 1 else -((params["avg_loss"] / 100) * position_size)
        capital += trade_return
        drawdown = (capital - max(capital, params["account_size"])) / params["account_size"] * 100
        drawdowns.append(drawdown)
        trade_results.append([trade, capital, trade_return, drawdown])
    
    return pd.DataFrame(trade_results, columns=["Trade #", "Cumulative Capital ($)", "Trade Return ($)", "Drawdown (%)"])

def create_bell_curve(trade_results, scenario_name, color):
    # Ensure correct column name
    fig = px.histogram(trade_results, x="Trade Return ($)", nbins=30, 
                       title=f"{scenario_name} Trade Return Distribution",
                       opacity=0.75, color_discrete_sequence=[color])

    # Compute normal distribution curve (Bell Curve)
    mean = trade_results["Trade Return ($)"].mean()
    std_dev = trade_results["Trade Return ($)"].std()
    
    if std_dev > 0:  # Avoid division by zero
        x_values = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
        y_values = stats.norm.pdf(x_values, mean, std_dev) * len(trade_results) * 3

        # Add normal distribution (bell curve)
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines', name='Normal Distribution', line=dict(color='black')))

    # Add vertical breakeven line at 0%
    fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=max(y_values) if std_dev > 0 else 1, 
                  line=dict(color="red", width=2))

    return fig

# Run simulations for all three scenarios
trade_results_1 = risk_management_analysis(scenario1)
trade_results_2 = risk_management_analysis(scenario2)
trade_results_3 = risk_management_analysis(scenario3)
trade_results_4 = risk_management_analysis(scenario4)

# ---- KPI METRICS ----
def calculate_kpi(trade_results, initial_capital):
    final_balance = trade_results["Cumulative Capital ($)"].iloc[-1]
    percent_change = ((final_balance - initial_capital) / initial_capital) * 100
    return final_balance, percent_change

final1, percent1 = calculate_kpi(trade_results_1, scenario1["account_size"])
final2, percent2 = calculate_kpi(trade_results_2, scenario2["account_size"])
final3, percent3 = calculate_kpi(trade_results_3, scenario3["account_size"])
final4, percent4 = calculate_kpi(trade_results_4, scenario4["account_size"])

# ---- Gradient KPI Box Design ----
kpi_box_style = """
    <style>
        .kpi-box {
    background-color: #222;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    font-size: 18px;
    font-weight: bold;
    color: white;
    border: 1px solid #444;
    box-shadow: 3px 3px 15px rgba(255,255,255,0.2);
            margin-bottom: 10px;
        }
        .positive { color: lightgreen; }
        .negative { color: #ff4747; }
    </style>
"""
st.markdown(kpi_box_style, unsafe_allow_html=True)

# Create KPI Columns
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
        <div class="kpi-box">
            <h4>Scenario 1</h4>
            <p>Final Account Balance: <br> <strong>${final1:,.2f}</strong></p>
            <p>Change: <strong class="{ 'positive' if percent1 >= 0 else 'negative' }">{percent1:.2f}%</strong></p>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
        <div class="kpi-box">
            <h4>Scenario 2</h4>
            <p>Final Account Balance: <br> <strong>${final2:,.2f}</strong></p>
            <p>Change: <strong class="{ 'positive' if percent2 >= 0 else 'negative' }">{percent2:.2f}%</strong></p>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
        <div class="kpi-box">
            <h4>Scenario 3</h4>
            <p>Final Account Balance: <br> <strong>${final3:,.2f}</strong></p>
            <p>Change: <strong class="{ 'positive' if percent3 >= 0 else 'negative' }">{percent3:.2f}%</strong></p>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
        <div class="kpi-box">
            <h4>Scenario 4</h4>
            <p>Final Account Balance: <br> <strong>${final3:,.2f}</strong></p>
            <p>Change: <strong class="{ 'positive' if percent4 >= 0 else 'negative' }">{percent4:.2f}%</strong></p>
        </div>
    """, unsafe_allow_html=True)

# ---- CUSTOM CSS FOR TABS Styled Tabs----
tabs_css = """
    <style>
        .stTabs [data-baseweb="tab-list"] {
            background: linear-gradient(90deg, #222, #444); 
            border-radius: 10px;
            padding: 10px;
        }

        .stTabs [data-baseweb="tab"] {
            font-size: 18px;
            font-weight: bold;
            color: white;
            padding: 12px 20px;
            margin-right: 10px;
            border-radius: 8px;
            transition: 0.3s;
        }

        .stTabs [data-baseweb="tab"]:hover {
            background-color: #666;
            color: #fff;
        }

        .stTabs [aria-selected="true"] {
            background: #ff4b2b !important;
            color: white !important;
            border-radius: 8px;
            font-weight: bold;
        }
    </style>
"""
st.markdown(tabs_css, unsafe_allow_html=True)

# ---- PERFORMANCE METRICS TABS ----
tab1, tab2, tab3, tab4 = st.tabs(["📊 Capital Growth", "📉 Cumulative Drawdown", "🏆 Box Plot", "📊 Bell Curve Analysis"])

# ---- FIXED CHARTS ----
with tab1:
    st.subheader("📊 Capital Growth Over Trades")
    fig1 = go.Figure()
    for df, name, color in zip([trade_results_1, trade_results_2, trade_results_3, trade_results_4], ["Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"], colors):
        fig1.add_trace(go.Scatter(x=df["Trade #"], y=df["Cumulative Capital ($)"], mode="lines", name=name, line=dict(color=color)))
    st.plotly_chart(fig1)

with tab2:
    st.subheader("📉 Cumulative Drawdown Over Time")
    fig2 = go.Figure()
    for df, name, color in zip([trade_results_1, trade_results_2, trade_results_3, trade_results_4], ["Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"], colors):
        fig2.add_trace(go.Scatter(x=df["Trade #"], y=df["Drawdown (%)"], mode="lines", name=name, line=dict(color=color)))
    st.plotly_chart(fig2)

with tab3:
    st.subheader("🏆 Box Plot of Trade Returns")
    
    # Combine trade return data for all scenarios
    trade_returns = pd.concat([
        trade_results_1["Trade Return ($)"], 
        trade_results_2["Trade Return ($)"], 
        trade_results_3["Trade Return ($)"], 
        trade_results_4["Trade Return ($)"]
    ], axis=1, keys=["Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"]) 

    # Melt the DataFrame for Plotly compatibility
    melted_trade_returns = trade_returns.melt(var_name="Scenario", value_name="Trade Return ($)")

    # Apply colors list directly in the Box Plot using categorical ordering
    fig3 = px.box(melted_trade_returns, 
                  x="Scenario", 
                  y="Trade Return ($)", 
                  color="Scenario",
                  color_discrete_sequence=colors)  # ✅ Uses colors list

    st.plotly_chart(fig3)

# ---- TAB 4: BELL CURVE HISTOGRAMS ----
with tab4:
    st.subheader("📊 Bell Curve of Trade Returns (Separated by Scenario)")

    # Create three columns for side-by-side charts
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.plotly_chart(create_bell_curve(trade_results_1, "Scenario 1", colors[0]), use_container_width=True)
    with col2:
        st.plotly_chart(create_bell_curve(trade_results_2, "Scenario 2", colors[1]), use_container_width=True)
    with col3:
        st.plotly_chart(create_bell_curve(trade_results_3, "Scenario 3", colors[2]), use_container_width=True)
    with col4:
        st.plotly_chart(create_bell_curve(trade_results_4, "Scenario 4", colors[3]), use_container_width=True)

    st.write("📌 **Summary:** Each histogram represents the trade return distribution for an individual scenario. The black line represents a normal distribution curve, and the red vertical line indicates breakeven (0%). This helps analyze trade return variability.")

# ---- STRATEGY ANALYSIS SECTION ----
st.markdown("---")
st.subheader("📊 Strategy Analysis")

tab_corr, tab_feature, tab_sensitivity, tab_risk = st.tabs(["📈 Correlation Heatmap", "🔥 Feature Importance", "🎲 Monte Carlo Sensitivity", "📉 Risk vs. Performance"])

# ---- FIXING MISSING CHARTS ----
with tab_corr:
    st.subheader("📈 Correlation Heatmap")
    combined_results = pd.concat([trade_results_1, trade_results_2, trade_results_3, trade_results_4])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(combined_results.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

with tab_feature:
    st.subheader("🔥 Feature Importance")
    X = pd.DataFrame({
        "Avg Gain": [scenario1["avg_gain"], scenario2["avg_gain"], scenario3["avg_gain"], scenario4["avg_gain"]],
        "Avg Loss": [scenario1["avg_loss"], scenario2["avg_loss"], scenario3["avg_loss"], scenario4["avg_loss"]],
        "Win Rate": [scenario1["batting_avg"], scenario2["batting_avg"], scenario3["batting_avg"], scenario4["batting_avg"]],
        "Position Size": [scenario1["position_size_pct"], scenario2["position_size_pct"], scenario3["position_size_pct"], scenario4["position_size_pct"]],
        "Trades": [scenario1["num_trades"], scenario2["num_trades"], scenario3["num_trades"], scenario4["num_trades"]]
    })
    y = [trade_results_1["Cumulative Capital ($)"].iloc[-1], trade_results_2["Cumulative Capital ($)"].iloc[-1], trade_results_3["Cumulative Capital ($)"].iloc[-1], trade_results_4["Cumulative Capital ($)"].iloc[-1]]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    fig = px.bar(importances.sort_values(), title="Feature Importance for Account Success")
    st.plotly_chart(fig)

# ---- TAB 3: MONTE CARLO SENSITIVITY ANALYSIS ----
with tab_sensitivity:
    st.subheader("🎲 Monte Carlo Sensitivity Analysis")
    mc_results = []
    for _ in range(500):
        params = {
            "avg_gain": np.random.uniform(5, 15),
            "avg_loss": np.random.uniform(3, 10),
            "batting_avg": np.random.uniform(40, 80),
            "position_size_pct": np.random.uniform(1, 10),
            "num_trades": np.random.randint(10, 500),
            "account_size": 10000
        }
        final_balance = risk_management_analysis(params)["Cumulative Capital ($)"].iloc[-1]
        mc_results.append([params["avg_gain"], params["avg_loss"], params["batting_avg"], params["position_size_pct"], params["num_trades"], final_balance])

    mc_df = pd.DataFrame(mc_results, columns=["Avg Gain", "Avg Loss", "Win Rate", "Position Size", "Trades", "Final Balance"])
    st.write(mc_df.describe())

with tab_risk:
    st.subheader("📉 Risk vs. Performance Analysis")

    # Compute Maximum Drawdown for Each Scenario
    def calculate_max_drawdown(trade_results):
        cumulative_returns = trade_results["Cumulative Capital ($)"]
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak * 100
        return drawdown.min()  # Worst drawdown (most negative)

    max_drawdown_1 = calculate_max_drawdown(trade_results_1)
    max_drawdown_2 = calculate_max_drawdown(trade_results_2)
    max_drawdown_3 = calculate_max_drawdown(trade_results_3)
    max_drawdown_4 = calculate_max_drawdown(trade_results_4)

    # Compute Sharpe Ratio
    risk_free_rate = 0.02  # Assuming 2% risk-free return
    def calculate_sharpe_ratio(trade_results):
        returns = trade_results["Trade Return ($)"]
        mean_return = returns.mean()
        std_dev_return = returns.std()
        if std_dev_return > 0:
            return (mean_return - risk_free_rate) / std_dev_return
        return 0  # Avoid division by zero

    sharpe_1 = calculate_sharpe_ratio(trade_results_1)
    sharpe_2 = calculate_sharpe_ratio(trade_results_2)
    sharpe_3 = calculate_sharpe_ratio(trade_results_3)
    sharpe_4 = calculate_sharpe_ratio(trade_results_4)

    # Create a DataFrame for Visualization
    risk_performance_df = pd.DataFrame({
        "Scenario": ["Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4"],
        "Max Drawdown (%)": [max_drawdown_1, max_drawdown_2, max_drawdown_3, max_drawdown_4],
        "Sharpe Ratio": [sharpe_1, sharpe_2, sharpe_3, sharpe_4]
    })

    # Scatter Plot for Risk vs. Return
    fig_risk = px.scatter(risk_performance_df, x="Max Drawdown (%)", y="Sharpe Ratio",
                          text="Scenario", color="Scenario",
                          title="Risk vs. Return Trade-Off",
                          color_discrete_map={"Scenario 1": colors[0], "Scenario 2": colors[1], "Scenario 3": colors[2], "Scenario 4": colors[3]},
                          size_max=10)

    # Add annotations
    fig_risk.update_traces(textposition="top center")

    # Display Chart
    st.plotly_chart(fig_risk)

    # Display Risk Performance Metrics
    st.write("📌 **Summary:** This chart visualizes the trade-off between **risk (Max Drawdown)** and **performance (Sharpe Ratio)** for each scenario. A **higher Sharpe Ratio** indicates better risk-adjusted returns, while a **lower Max Drawdown** suggests a more stable strategy.")
    st.dataframe(risk_performance_df)