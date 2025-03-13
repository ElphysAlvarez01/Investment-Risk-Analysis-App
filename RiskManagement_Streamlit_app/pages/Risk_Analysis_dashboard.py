import streamlit as st

# ---- ADD A BANNER IMAGE ----
st.image("pages/Image_folder/Research_dashboard.png", use_container_width=True)


def main():
    st.title("📉 Risk Analysis Page")
    st.write("This page will contain analysis related to risk management XXXXXXXXXXXXXXX.")
    st.write("You can add detailed breakdowns of different risk measures here.")
    
    # 🔗 Add a link to your GitHub
    st.markdown("[📂 View Source Code on GitHub](https://github.com/your_username/risk-management-app)", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

