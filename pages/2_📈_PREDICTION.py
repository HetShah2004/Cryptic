import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import ta
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="Crypto Forecaster", page_icon="🔮", layout="wide")

# --- Background Helper ---
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover;
        }}
        div[data-testid="stMetric"] {{
            background-color: rgba(38, 39, 48, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    except FileNotFoundError:
        pass  # Skip background if image not found

# Try to add background
add_bg_from_local('images/hetgabdu.jpg')

# --- UI Title & Controls ---
st.title("🔮 Crypto Price Forecaster")
st.markdown("Select a cryptocurrency and a time frame to forecast future price trends.")
st.write("---")
st.subheader("Dashboard Controls")
col1, col2 = st.columns([1, 2])

with col1:
    crypto_name = st.selectbox("Select Cryptocurrency", ("SOL-USD", "DOGE-USD", "MATIC-USD","XRP-USD", "BNB-USD", "ETH-USD", "BTC-USD", "AVAX-USD", "TRX-USD", "ADA-USD"), index=0)

with col2:
    st.markdown("<span style='font-size: 0.9em;'>Select Forecast Period:</span>", unsafe_allow_html=True)
    buttons = {'7 Days': 7, '15 Days': 15, '30 Days': 30}  # Reduced to more realistic timeframes
    b_cols = st.columns(3)

    if 'forecast_days' not in st.session_state:
        st.session_state.forecast_days = 7

    for (label, days), col in zip(buttons.items(), b_cols):
        if col.button(label, use_container_width=True, key=f"btn_{days}"):
            st.session_state.forecast_days = days

forecast_days = st.session_state.forecast_days
st.write("---")

# --- Load Data ---
@st.cache_data(ttl=300)
def load_data(ticker):
    try:
        data = yf.download(ticker, start="2020-01-01", progress=False)  # More recent data
        if not data.empty:
            data.reset_index(inplace=True)
            
            # Handle multi-level columns if they exist
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten multi-level columns by taking the first level
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            
            # Ensure we have the expected columns
            expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if 'Adj Close' in data.columns:
                expected_cols.append('Adj Close')
            
            # Keep only the columns we need
            available_cols = [col for col in expected_cols if col in data.columns]
            data = data[available_cols]
            
            return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
    return None

# --- Feature Engineering Function ---
def create_features(df):
    """Create technical indicators and features"""
    df = df.copy()
    
    # Print column info for debugging
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")
    
    # Ensure all price columns are proper 1D Series and handle any multi-level structure
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        if col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                # If it's a DataFrame (multi-level), take the first column
                df[col] = df[col].iloc[:, 0]
            else:
                # Ensure it's a proper 1D Series
                df[col] = pd.Series(df[col].values.flatten(), index=df.index)
    
    # Verify we have the required columns
    required_cols = ['Close', 'High', 'Low', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return df
    
    # Price-based features
    df['Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Price_Change'] = df['Close'] - df['Close'].shift(1)
    
    # Moving averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_10'] = df['Close'].ewm(span=10).mean()
    
    # Technical indicators - using the properly formatted Series
    try:
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        macd_indicator = ta.trend.MACD(df['Close'])
        df['MACD'] = macd_indicator.macd()
        df['MACD_Signal'] = macd_indicator.macd_signal()
        df['MACD_Histogram'] = macd_indicator.macd_diff()
        
        # Bollinger Bands
        bb_indicator = ta.volatility.BollingerBands(df['Close'], window=20)
        df['BB_Upper'] = bb_indicator.bollinger_hband()
        df['BB_Lower'] = bb_indicator.bollinger_lband()
        df['BB_Middle'] = bb_indicator.bollinger_mavg()
        
        # Handle potential division by zero
        bb_width_denominator = df['BB_Middle'].replace(0, np.nan)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / bb_width_denominator
        
        bb_range = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = np.where(bb_range != 0, 
                                   (df['Close'] - df['BB_Lower']) / bb_range, 
                                   0.5)
        
        # Volatility indicators
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        
        # Momentum indicators
        df['ROC'] = ta.momentum.ROCIndicator(df['Close'], window=10).roc()
        
    except Exception as e:
        st.warning(f"Some technical indicators could not be calculated: {str(e)}")
        # Fallback to basic indicators
        df['RSI'] = 50  # Neutral RSI
        df['MACD'] = 0
        df['MACD_Signal'] = 0
        df['MACD_Histogram'] = 0
        df['BB_Upper'] = df['Close'] * 1.02
        df['BB_Lower'] = df['Close'] * 0.98
        df['BB_Middle'] = df['Close']
        df['BB_Width'] = 0.04
        df['BB_Position'] = 0.5
        df['ATR'] = df['Close'] * 0.02
        df['ROC'] = 0
    
    # Simple volatility and volume indicators
    df['Volatility'] = df['Return'].rolling(window=10).std()
    
    # Volume indicators - with safe division
    df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
    volume_sma_safe = df['Volume_SMA'].replace(0, np.nan)
    df['Volume_Ratio'] = df['Volume'] / volume_sma_safe
    
    # Momentum indicators
    df['Momentum'] = df['Close'] / df['Close'].shift(4) - 1
    
    # Price position relative to recent highs/lows
    df['High_20'] = df['High'].rolling(window=20).max()
    df['Low_20'] = df['Low'].rolling(window=20).min()
    
    # Safe division for price position
    price_range = df['High_20'] - df['Low_20']
    df['Price_Position'] = np.where(price_range != 0,
                                  (df['Close'] - df['Low_20']) / price_range,
                                  0.5)
    
    return df

# --- Advanced Forecasting Function ---
def forecast_iteratively(model, scaler, last_row, feature_cols, forecast_days, recent_prices):
    """Iteratively forecast by updating features at each step"""
    predictions = []
    current_features = last_row[feature_cols].copy()
    
    # Get recent price history for calculating moving averages
    price_history = list(recent_prices[-30:])  # Last 30 days for calculations
    
    for day in range(forecast_days):
        # Scale current features
        current_scaled = scaler.transform([current_features])
        
        # Predict next price
        next_price = model.predict(current_scaled)[0]
        
        # Apply constraints to prevent unrealistic predictions
        last_price = price_history[-1]
        max_daily_change = 0.15  # Maximum 15% daily change
        min_price = last_price * (1 - max_daily_change)
        max_price = last_price * (1 + max_daily_change)
        next_price = np.clip(next_price, min_price, max_price)
        
        predictions.append(next_price)
        price_history.append(next_price)
        
        # Update features for next iteration
        if len(price_history) >= 2:
            current_features['Return'] = (next_price - price_history[-2]) / price_history[-2]
            current_features['Log_Return'] = np.log(next_price / price_history[-2])
            current_features['Price_Change'] = next_price - price_history[-2]
        
        # Update moving averages
        if len(price_history) >= 5:
            current_features['SMA_5'] = np.mean(price_history[-5:])
        if len(price_history) >= 10:
            current_features['SMA_10'] = np.mean(price_history[-10:])
            current_features['EMA_10'] = current_features['EMA_10'] * 0.8182 + next_price * 0.1818
        if len(price_history) >= 20:
            current_features['SMA_20'] = np.mean(price_history[-20:])
        
        # Update other indicators (simplified - in practice, you'd recalculate properly)
        # For RSI, MACD etc., we'll use a dampening factor to prevent extreme swings
        dampening = 0.95
        current_features['RSI'] = current_features['RSI'] * dampening + 50 * (1 - dampening)
        current_features['MACD'] = current_features['MACD'] * dampening
        current_features['BB_Position'] = np.clip(current_features['BB_Position'] * dampening + 0.5 * (1 - dampening), 0, 1)
        current_features['Volatility'] = current_features['Volatility'] * dampening
        current_features['Momentum'] = (next_price / price_history[-min(5, len(price_history))] - 1) if len(price_history) >= 5 else 0
        
    return np.array(predictions)

# --- Load and Process Data ---
data_load_state = st.text(f"Loading data for {crypto_name}...")
df_raw = load_data(crypto_name)
data_load_state.text("")

if df_raw is not None and not df_raw.empty:
    # Create features
    df = create_features(df_raw)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Remove rows with NaN values
    df.dropna(inplace=True)
    
    if len(df) < 100:
        st.error("Insufficient data for reliable predictions. Please try a different cryptocurrency.")
        st.stop()
    
    # Define feature columns (excluding target and date)
    feature_cols = [col for col in df.columns if col not in ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'Adj Close']]
    
    # Prepare data for modeling
    X = df[feature_cols].fillna(method='ffill').fillna(0)  # Handle any remaining NaN
    y = df['Close']
    
    # Additional data cleaning
    X = X.replace([np.inf, -np.inf], 0)  # Replace infinite values
    X = X.fillna(0)  # Fill any remaining NaN values
    
    # Split data (use more recent data for training)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model - using RandomForest for better handling of feature interactions
    model_choice = st.selectbox("Choose Model", ["Random Forest", "SVR"])
    
    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, min_samples_split=5)
    else:
        model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.01)
    
    model.fit(X_train_scaled, y_train)
    
    # Calculate model performance
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    # Generate forecasts
    last_row = df.iloc[-1]
    recent_prices = df['Close'].tail(30).values
    
    with st.spinner("Generating forecasts..."):
        future_predictions = forecast_iteratively(
            model, scaler, last_row, feature_cols, forecast_days, recent_prices
        )
    
    # Create future dates
    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    
    # Calculate prediction confidence intervals (simplified)
    recent_volatility = df['Close'].pct_change().tail(30).std()
    confidence_interval = recent_volatility * np.sqrt(np.arange(1, forecast_days + 1)) * 1.96
    upper_bound = future_predictions * (1 + confidence_interval)
    lower_bound = future_predictions * (1 - confidence_interval)
    
    # --- KPIs ---
    st.subheader(f"Key Performance Indicators (KPIs) for {forecast_days}-Day Forecast")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    last_actual_price = float(y.iloc[-1])
    final_predicted_price = future_predictions[-1]
    percent_change = ((final_predicted_price - last_actual_price) / last_actual_price) * 100
    
    kpi1.metric("Current Price", f"${last_actual_price:,.2f}")
    kpi2.metric(f"Predicted Price ({forecast_days}d)", f"${final_predicted_price:,.2f}", f"{percent_change:+.2f}%")
    kpi3.metric("Model Test Score", f"{test_score:.3f}")
    kpi4.metric("Avg Daily Change", f"{percent_change/forecast_days:+.2f}%")
    
    st.write("---")
    
    # --- Chart ---
    st.subheader(f"Price Forecast for {crypto_name}")
    
    # Show last 60 days of historical data + forecast
    recent_df = df.tail(60)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Historical price
    fig.add_trace(
        go.Scatter(x=recent_df['Date'], y=recent_df['Close'], 
                  name='Historical Price', line=dict(color='#00cc96', width=2)),
        secondary_y=True
    )
    
    # Forecasted price
    fig.add_trace(
        go.Scatter(x=future_dates, y=future_predictions, 
                  name='Forecasted Price', line=dict(color='#ff6b6b', width=2, dash='dash')),
        secondary_y=True
    )
    
    # Confidence interval
    fig.add_trace(
        go.Scatter(x=future_dates, y=upper_bound, fill=None, mode='lines',
                  line_color='rgba(0,0,0,0)', showlegend=False),
        secondary_y=True
    )
    fig.add_trace(
        go.Scatter(x=future_dates, y=lower_bound, fill='tonexty', mode='lines',
                  line_color='rgba(0,0,0,0)', name='Confidence Interval',
                  fillcolor='rgba(255,107,107,0.2)'),
        secondary_y=True
    )
    
    # Volume
    fig.add_trace(
        go.Bar(x=recent_df['Date'], y=recent_df['Volume'], 
               name='Volume', marker_color='rgba(119, 119, 119, 0.3)'),
        secondary_y=False
    )
    
    # Final prediction point
    fig.add_trace(
        go.Scatter(x=[future_dates[-1]], y=[final_predicted_price], 
                  name='Final Prediction', mode='markers',
                  marker=dict(color='red', size=12, symbol='star')),
        secondary_y=True
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=f"{crypto_name} Price Forecast - {model_choice} Model"
    )
    fig.update_yaxes(title_text="Volume", secondary_y=False)
    fig.update_yaxes(title_text="Price (USD)", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model insights
    if model_choice == "Random Forest":
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        st.subheader("Top 10 Most Important Features")
        st.bar_chart(feature_importance.set_index('Feature'))
    
    st.success(f"Model trained on {len(X_train)} samples. Test R² Score: {test_score:.3f}")
    st.warning("**Disclaimer:** This forecast uses historical data and technical indicators. Cryptocurrency markets are highly volatile and unpredictable. This is not financial advice.")
    
    # Show prediction statistics
    with st.expander("Prediction Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Forecast Summary:**")
            st.write(f"- Average predicted price: ${np.mean(future_predictions):,.2f}")
            st.write(f"- Max predicted price: ${np.max(future_predictions):,.2f}")
            st.write(f"- Min predicted price: ${np.min(future_predictions):,.2f}")
            st.write(f"- Prediction volatility: {np.std(future_predictions)/np.mean(future_predictions)*100:.2f}%")
        
        with col2:
            st.write("**Model Performance:**")
            st.write(f"- Training R² Score: {train_score:.3f}")
            st.write(f"- Test R² Score: {test_score:.3f}")
            st.write(f"- Recent volatility: {recent_volatility*100:.2f}%")
            st.write(f"- Features used: {len(feature_cols)}")

else:
    st.error("Could not load financial data. Please check your internet connection and try again later.")