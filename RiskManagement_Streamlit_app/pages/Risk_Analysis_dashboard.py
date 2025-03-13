import streamlit as st
import os
# ---- ADD A BANNER IMAGE ----

# Correct path based on GitHub folder structure
image_path = os.path.join(os.path.dirname(__file__), "Image_folder", "Research_dashboard.png")

# Check if the image exists before loading
if os.path.exists(image_path):
    st.image(image_path, use_container_width=True)
else:
    st.warning(f"Image not found: {image_path}")

def main():
    st.title("📉 Risk Analysis Page")
    st.write("This page will contain analysis related to risk management XXXXXXXXXXXXXXX.")
    st.write("You can add detailed breakdowns of different risk measures here.")
    
    # 🔗 Add a link to your GitHub
    st.markdown("[📂 View Source Code on GitHub](https://github.com/your_username/risk-management-app)", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

