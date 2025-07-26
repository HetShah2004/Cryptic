import streamlit as st
from PIL import Image
import base64

# --- Page Configuration ---
st.set_page_config(page_title="Learn Crypto", page_icon="📚", layout="wide")

# --- Helper Functions ---
def add_bg_from_local(image_file):
    """Sets a local background image."""
    try:
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning("Background image not found.")

def local_css(file_name):
    """Loads a local CSS file."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {file_name}")

# --- Load Assets & Styles ---
local_css("style/style.css")
add_bg_from_local('images/dd.jpg')

# --- Page Title & Introduction ---
st.title("📚 Learn About Cryptocurrency")
st.markdown("Explore these hand-picked videos to get started on your crypto journey, from understanding the basics to learning how to invest.")
st.write("---")

# --- Learning Resources Data ---
# Storing data in a list makes the code cleaner and easier to update
learning_resources = [
    {
        "title": "Cryptocurrency In 5 Minutes | What Is Cryptocurrency?",
        "image_path": "images/what.png", #
        "url": "https://www.youtube.com/watch?v=1YyAzVmP9xQ" # Example URL
    },
    {
        "title": "How Cryptocurrency ACTUALLY works.",
        "image_path": "images/how.png", #
        "url": "https://www.youtube.com/watch?v=bBC-nXj3Ng4" # Example URL
    },
    {
        "title": "How to invest in Crypto Currency!",
        "image_path": "images/f.png", #
        "url": "https://www.youtube.com/watch?v=Yb6825eWv2E" # Example URL
    },
    {
        "title": "How To Invest In Crypto Full Beginners Guide in 2023",
        "image_path": "images/n.png", #
        "url": "https://www.youtube.com/watch?v=CgI-2nS9-uU" # Example URL
    }
]

# --- Display Resources in a Grid ---
# Create a 2x2 grid layout
cols = st.columns(2)

for i, resource in enumerate(learning_resources):
    # Determine which column to place the card in
    col = cols[i % 2]
    
    with col:
        # Using a container with a border for the card effect
        with st.container(border=True):
            try:
                st.image(resource["image_path"])
            except FileNotFoundError:
                st.error(f"Image not found at {resource['image_path']}")