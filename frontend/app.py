import streamlit as st
import os

# Set page config
st.set_page_config(
    page_title="Research Paper RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Import components and pages (after setting config)
from frontend.components.sidebar import render_sidebar
from frontend.components.chat import render_chat_interface
from frontend.pages.evaluation import render_evaluation_page

def main():
    # Render Sidebar
    render_sidebar()
    
    # Navigation
    # Since Streamlit pages logic can be tricky, we'll use a simple radio button or just tab selection
    tab1, tab2 = st.tabs(["💬 Chat Interface", "📊 Evaluation Dashboard"])
    
    with tab1:
        st.title("Research Paper Assistant")
        render_chat_interface()
        
    with tab2:
        render_evaluation_page()

if __name__ == "__main__":
    main()
