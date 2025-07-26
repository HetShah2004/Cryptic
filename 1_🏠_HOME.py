import streamlit as st
import base64
import pandas as pd
from PIL import Image
from streamlit_lottie import st_lottie
import requests

# --- Page Configuration ---
st.set_page_config(page_title="CRYPTO WEB-APP", page_icon="🪙", layout="wide")

# --- Helper Functions ---

def load_animation(url):
    """Fetches a Lottie animation from a URL."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.exceptions.RequestException:
        return None

def local_css(file_name):
    """Loads a local CSS file."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {file_name}")

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

def round_value(input_value):
    """Rounds a pandas Series value correctly."""
    if input_value.empty:
        return 0.0
    value = input_value.iloc[0]
    if value > 1:
        return float(round(value, 2))
    else:
        return float(round(value, 8))

# --- Load Assets & Styles ---
local_css("style/style.css")
add_bg_from_local('images/hetgabdu.jpg')

# --- Header Section ---
with st.container():
    left_column, right_column = st.columns((1, 10))
    with left_column:
        animation_logo = load_animation("https://assets8.lottiefiles.com/packages/lf20_pxiupds9.json")
        if animation_logo:
            st_lottie(animation_logo, height=100, key="logo_animation")
    with right_column:
        st.markdown('<b class="big-font">CRYPTO DASHBOARD</b>', unsafe_allow_html=True)
st.write("---")

# --- Data Fetching and Caching ---
@st.cache_data(ttl=300) # Cache data for 5 minutes
def get_top_crypto_data():
    """
    Fetches 24hr ticker data from Binance API, filters for USDT pairs,
    and returns the top 200 by trading volume.
    """
    try:
        df = pd.read_json('https://api.binance.com/api/v3/ticker/24hr')
        usdt_df = df[df['symbol'].str.endswith('USDT')].copy()
        usdt_df['quoteVolume'] = pd.to_numeric(usdt_df['quoteVolume'])
        top_200_symbols = usdt_df.sort_values(by='quoteVolume', ascending=False).head(200)
        return top_200_symbols
    except Exception as e:
        st.error(f"Error fetching data from Binance API: {e}")
        return pd.DataFrame()

top_cryptos_df = get_top_crypto_data()

# --- Main Content Area ---
st.header('Live Prices of Top Cryptocurrencies')

# Initialize selected_cryptos to an empty list
selected_cryptos = []

# --- Cryptocurrency Customization moved to Main Page ---
if not top_cryptos_df.empty:
    top_symbols_list = top_cryptos_df['symbol'].tolist()
    
    default_selection = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 
        'DOGEUSDT', 'AVAXUSDT', 'BNBUSDT', 'TRXUSDT', 
        'ADAUSDT', 'MATICUSDT'
    ]
    
    valid_defaults = [coin for coin in default_selection if coin in top_symbols_list]

    with st.expander("Customize Watched Cryptocurrencies", expanded=True):
        selected_cryptos = st.multiselect(
            "Select from the Top 200 most-traded coins:",
            options=top_symbols_list,
            default=valid_defaults,
            label_visibility="collapsed" # Hides the long label for a cleaner look
        )
else:
    st.warning("Could not load the list of top cryptocurrencies.")

st.write("---")

# --- Display Price Metrics ---
if not top_cryptos_df.empty and selected_cryptos:
    cols = st.columns(3)
    
    for i, crypto_symbol in enumerate(selected_cryptos):
        crypto_data = top_cryptos_df[top_cryptos_df.symbol == crypto_symbol]
        
        if not crypto_data.empty:
            price = round_value(crypto_data.weightedAvgPrice)
            percent_change = round_value(crypto_data.priceChangePercent)
            
            with cols[i % 3]:
                st.metric(
                    label=crypto_symbol,
                    value=f"${price:,.4f}",
                    delta=f"{percent_change}%"
                )
else:
    st.info("Please select one or more cryptocurrencies from the customization menu to view live prices.")