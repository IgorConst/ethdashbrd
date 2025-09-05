import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Check if OpenAI is available
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Crypto Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Global variables
if 'data_source' not in st.session_state:
    st.session_state.data_source = "binance"
if 'selected_coin' not in st.session_state:
    st.session_state.selected_coin = "ETHUSDT"
if 'show_ai_forecast' not in st.session_state:
    st.session_state.show_ai_forecast = False

# Available coins for trading view
CRYPTO_SYMBOLS = {
    "ETHUSDT": "Ethereum",
    "BTCUSDT": "Bitcoin", 
    "BNBUSDT": "BNB",
    "ADAUSDT": "Cardano",
    "SOLUSDT": "Solana"
}

# Create a robust session with retries
def create_robust_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=2,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# Centralized function to get data with multiple fallbacks
def get_data_with_fallbacks(url, params=None, data_type="klines"):
    """Try multiple sources until one works"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    
    session = create_robust_session()
    
    # Try Binance first
    if st.session_state.data_source != "binance_blocked":
        try:
            response = session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            st.session_state.data_source = "binance"
            return response.json(), "binance"
        except Exception as e:
            st.session_state.data_source = "binance_blocked"
    
    # Try CryptoCompare
    try:
        if data_type == "klines":
            symbol = params.get('symbol', 'ETHUSDT')
            fsym = symbol.replace('USDT', '')
            
            cryptocompare_url = "https://min-api.cryptocompare.com/data/v2/histominute"
            cryptocompare_params = {
                'fsym': fsym,
                'tsym': 'USD',
                'limit': 96,
                'aggregate': 15
            }
            
            response = session.get(cryptocompare_url, params=cryptocompare_params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['Response'] == 'Success':
                st.session_state.data_source = "cryptocompare"
                return data['Data']['Data'], "cryptocompare"
                
        elif data_type == "ticker":
            symbol = params.get('symbol', 'ETHUSDT')
            fsym = symbol.replace('USDT', '')
            
            cryptocompare_url = "https://min-api.cryptocompare.com/data/price"
            cryptocompare_params = {'fsym': fsym, 'tsym': 'USD'}
            
            response = session.get(cryptocompare_url, params=cryptocompare_params, headers=headers, timeout=10)
            response.raise_for_status()
            price_data = response.json()
            
            change_url = "https://min-api.cryptocompare.com/data/pricemultifull"
            change_params = {'fsyms': fsym, 'tsyms': 'USD'}
            
            change_response = session.get(change_url, params=change_params, headers=headers, timeout=10)
            change_response.raise_for_status()
            change_data = change_response.json()
            
            formatted_data = {
                'lastPrice': price_data['USD'],
                'priceChangePercent': change_data['RAW'][fsym]['USD']['CHANGEPCT24HOUR'],
                'volume': change_data['RAW'][fsym]['USD']['VOLUME24HOUR']
            }
            
            st.session_state.data_source = "cryptocompare"
            return formatted_data, "cryptocompare"
            
    except Exception as e:
        st.warning(f"CryptoCompare failed: {e}")
    
    # If all APIs fail, return demo data
    return generate_demo_data(data_type), "demo"

def generate_demo_data(data_type, symbol="ETHUSDT"):
    """Generate demo data when all APIs fail"""
    base_price = 2500 if "ETH" in symbol else 50000 if "BTC" in symbol else 300 if "BNB" in symbol else 0.5 if "ADA" in symbol else 100
    
    if data_type == "klines":
        now = datetime.now()
        timestamps = [now - timedelta(minutes=15*i) for i in range(96)]
        timestamps.reverse()
        
        demo_data = []
        
        for i, ts in enumerate(timestamps):
            open_price = base_price + np.random.normal(0, base_price*0.02)
            close_price = open_price + np.random.normal(0, base_price*0.03)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, base_price*0.01))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, base_price*0.01))
            volume = np.random.normal(50000, 10000)
            
            demo_data.append([
                int(ts.timestamp() * 1000),
                open_price, high_price, low_price, close_price,
                volume,
                int(ts.timestamp() * 1000),
                volume * close_price,
                1000,
                50000,
                50000 * close_price,
                0
            ])
        
        return demo_data
        
    elif data_type == "ticker":
        price = base_price + np.random.normal(0, base_price*0.02)
        change = np.random.normal(0, 2)
        volume = 50000 + np.random.normal(0, 10000)
        
        return {
            'lastPrice': price,
            'priceChangePercent': change,
            'volume': volume
        }
    
    return None

# Convert CryptoCompare data to Binance-like format
def convert_cryptocompare_data(data):
    """Convert CryptoCompare format to Binance-like format"""
    converted_data = []
    for item in data:
        converted_data.append([
            item['time'] * 1000,
            item['open'],
            item['high'],
            item['low'],
            item['close'],
            item['volumeto'],
            item['time'] * 1000,
            item['volumefrom'],
            0, 0, 0, 0
        ])
    return converted_data

def get_groq_api_key():
    """Get Groq API key from multiple sources"""
    try:
        if hasattr(st, 'secrets'):
            if 'GROQ_API_KEY' in st.secrets:
                return st.secrets['GROQ_API_KEY']
            elif 'api_keys' in st.secrets and 'GROQ_API_KEY' in st.secrets['api_keys']:
                return st.secrets['api_keys']['GROQ_API_KEY']
    except:
        pass
    
    return os.getenv('GROQ_API_KEY')

# Cache functions
@st.cache_data(ttl=300)
def fetch_candlestick_data(symbol="ETHUSDT"):
    """Fetch candlestick data for selected coin"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': '15m',
            'limit': 96
        }
        
        data, source = get_data_with_fallbacks(url, params, "klines")
        
        if data is None:
            return pd.DataFrame()
        
        if source == "cryptocompare":
            data = convert_cryptocompare_data(data)
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df['data_source'] = source
        return df
        
    except Exception as e:
        st.error(f"Error processing candlestick data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def fetch_crypto_volumes():
    """Fetch volume data for top 5 cryptos"""
    cryptos = ['Bitcoin', 'Ethereum', 'BNB', 'Cardano', 'Solana']
    
    volume_data = {}
    dates = [(datetime.now() - timedelta(days=i)).strftime('%m-%d') for i in range(3)]
    
    for crypto in cryptos:
        volumes = [np.random.normal(1000000, 200000) for _ in range(3)]
        volume_data[crypto] = {
            'dates': dates,
            'volumes': volumes,
            'source': 'demo'
        }
    
    return volume_data

@st.cache_data(ttl=60)
def get_crypto_analysis(symbol="ETHUSDT"):
    """Get current analysis for selected crypto with proper fallbacks"""
    try:
        # First try Binance directly for ticker data
        url = "https://api.binance.com/api/v3/ticker/24hr"
        params = {'symbol': symbol}
        
        # Use the same robust session as candlestick data
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
        
        session = create_robust_session()
        response = session.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # If we get here, Binance worked!
        price = float(data['lastPrice'])
        change = float(data.get('priceChangePercent', 0))
        volume = float(data.get('volume', 50000))
        
        # Get historical data for RSI from the same source
        kline_url = "https://api.binance.com/api/v3/klines"
        kline_params = {
            'symbol': symbol,
            'interval': '1h',
            'limit': 30
        }
        
        kline_response = session.get(kline_url, params=kline_params, headers=headers, timeout=10)
        kline_response.raise_for_status()
        kline_data = kline_response.json()
        
        # Calculate RSI from historical prices
        prices = [float(candle[4]) for candle in kline_data]  # closing prices
        rsi = calculate_rsi(prices)
        
        return {
            'price': price,
            'change_24h': change,
            'volume_24h': volume,
            'rsi': rsi,
            'timestamp': datetime.now().strftime('%H:%M:%S UTC'),
            'source': 'binance'
        }
        
    except Exception as e:
        # If Binance fails, try alternative approach
        st.warning(f"Binance ticker failed, using fallback: {e}")
        return get_crypto_analysis_fallback(symbol)

def get_crypto_analysis_fallback(symbol="ETHUSDT"):
    """Fallback method using candlestick data for metrics"""
    try:
        # Get the latest candlestick data (which we know works)
        df = fetch_candlestick_data(symbol)
        
        if df.empty:
            return None
        
        # Use the most recent candle for metrics
        latest = df.iloc[-1]
        price = latest['close']
        
        # Calculate 24h change from first to last candle
        if len(df) > 1:
            first_price = df.iloc[0]['close']
            change = ((price - first_price) / first_price) * 100
        else:
            change = 0
        
        # Use volume from latest candle (approximate)
        volume = latest['volume'] * 96  # Approximate 24h volume
        
        # Generate RSI from available data
        prices = df['close'].tolist()
        rsi = calculate_rsi(prices)
        
        return {
            'price': price,
            'change_24h': change,
            'volume_24h': volume,
            'rsi': rsi,
            'timestamp': datetime.now().strftime('%H:%M:%S UTC'),
            'source': 'binance_candle'
        }
        
    except Exception as e:
        st.error(f"All analysis methods failed: {e}")
        # Ultimate fallback - demo data
        base_price = 2500 if "ETH" in symbol else 50000 if "BTC" in symbol else 300 if "BNB" in symbol else 0.5 if "ADA" in symbol else 100
        price = base_price + np.random.normal(0, base_price*0.02)
        
        return {
            'price': price,
            'change_24h': np.random.normal(0, 2),
            'volume_24h': 50000 + np.random.normal(0, 10000),
            'rsi': 50,
            'timestamp': datetime.now().strftime('%H:%M:%S UTC'),
            'source': 'demo'
        }

def get_ai_forecast():
    """Get AI-powered forecast using Groq"""
    api_key = get_groq_api_key()
    
    if not api_key:
        return generate_demo_forecast()
    
    try:
        analysis = get_crypto_analysis(st.session_state.selected_coin)
        if not analysis:
            return generate_demo_forecast()
        
        market_context = get_market_context()
        coin_name = CRYPTO_SYMBOLS.get(st.session_state.selected_coin, st.session_state.selected_coin)
        
        prompt = f"""
        As a crypto analyst, provide a concise forecast for {coin_name} based on current data:

        Current Data:
        - Price: ${analysis['price']:,.2f}
        - 24h Change: {analysis['change_24h']:+.2f}%
        - RSI: {analysis['rsi']}
        - Volume: {analysis['volume_24h']:,.0f}
        - Market Context: {market_context}

        Provide concise 4-hour and 24-hour forecasts with key factors and risk level.
        """
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": "You are an expert cryptocurrency analyst providing concise, actionable forecasts."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 300,
                "temperature": 0.3
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            forecast_text = data['choices'][0]['message']['content']
            
            forecast_data = parse_ai_forecast(forecast_text)
            forecast_data['full_analysis'] = forecast_text
            forecast_data['timestamp'] = datetime.now().strftime('%H:%M:%S UTC')
            forecast_data['source'] = 'groq'
            return forecast_data
        else:
            return generate_demo_forecast()
            
    except Exception as e:
        st.warning(f"AI service temporarily unavailable")
        return generate_demo_forecast()

def generate_demo_forecast():
    """Generate demo forecast"""
    coin_name = CRYPTO_SYMBOLS.get(st.session_state.selected_coin, st.session_state.selected_coin)
    return {
        'forecast_4h': 'NEUTRAL',
        'forecast_24h': 'NEUTRAL', 
        'confidence_4h': 50,
        'confidence_24h': 50,
        'risk_level': 'Moderate',
        'full_analysis': f"Demo analysis for {coin_name}: Current market conditions suggest sideways movement. Monitor key support and resistance levels for breakout opportunities.",
        'timestamp': datetime.now().strftime('%H:%M:%S UTC'),
        'source': 'demo'
    }

def get_market_context():
    """Get market context"""
    source = st.session_state.data_source
    if source == "binance":
        return "Normal trading conditions"
    elif source == "cryptocompare":
        return "Using fallback data"
    elif source == "demo":
        return "Using demo data - APIs blocked"
    else:
        return "Market data temporarily unavailable"

def parse_ai_forecast(forecast_text):
    """Parse AI forecast text"""
    forecast_data = {
        'forecast_4h': 'NEUTRAL',
        'forecast_24h': 'NEUTRAL',
        'confidence_4h': 50,
        'confidence_24h': 50,
        'risk_level': 'Moderate'
    }
    
    try:
        lines = forecast_text.lower()
        if 'bullish' in lines and ('4-hour' in lines or '4h' in lines):
            forecast_data['forecast_4h'] = 'BULLISH'
        elif 'bearish' in lines and ('4-hour' in lines or '4h' in lines):
            forecast_data['forecast_4h'] = 'BEARISH'
        
        if 'bullish' in lines and ('24-hour' in lines or '24h' in lines):
            forecast_data['forecast_24h'] = 'BULLISH'
        elif 'bearish' in lines and ('24-hour' in lines or '24h' in lines):
            forecast_data['forecast_24h'] = 'BEARISH'
        
        if 'high' in lines and 'risk' in lines:
            forecast_data['risk_level'] = 'High'
        elif 'low' in lines and 'risk' in lines:
            forecast_data['risk_level'] = 'Low'
    except:
        pass
    
    return forecast_data

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    if len(prices) < period + 1:
        return 50
    
    prices = np.array(prices)
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gains = np.mean(gains[-period:])
    avg_losses = np.mean(losses[-period:])
    
    if avg_losses == 0:
        return 100
    
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 1)

def create_candlestick_chart(df, coin_name):
    """Create candlestick chart with visual indicators"""
    if df.empty:
        return None
    
    is_demo = 'data_source' in df and df['data_source'].iloc[0] == 'demo'
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"{coin_name} - 15min Candlesticks (24h) {'📊 [DEMO]' if is_demo else ''}", 
            f"Volume {'📊 [DEMO]' if is_demo else ''}"
        ),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    if is_demo:
        inc_color = 'rgba(150,150,150,0.7)'
        dec_color = 'rgba(100,100,100,0.7)'
        vol_color = 'rgba(150,150,250,0.4)'
    else:
        inc_color = '#2CA02C'
        dec_color = '#D62728'
        vol_color = 'rgba(0,150,250,0.6)'
    
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=coin_name,
            increasing_line_color=inc_color,
            decreasing_line_color=dec_color
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color=vol_color
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{coin_name} Chart {'📊 [DEMO DATA]' if is_demo else ''}",
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=False
    )
    
    return fig

def create_volume_chart(volume_data):
    """Create volume comparison chart"""
    if not volume_data:
        return None
    
    is_demo = any('source' in volume_data[crypto] and volume_data[crypto]['source'] == 'demo' 
                 for crypto in volume_data)
    
    fig = go.Figure()
    first_crypto = list(volume_data.keys())[0]
    dates = volume_data[first_crypto]['dates']
    
    if is_demo:
        colors = ['#888888', '#aaaaaa', '#cccccc']
    else:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, date in enumerate(dates):
        crypto_names = []
        volumes = []
        
        for crypto, data in volume_data.items():
            crypto_names.append(crypto)
            volumes.append(data['volumes'][i])
        
        fig.add_trace(
            go.Bar(
                name=date,
                x=crypto_names,
                y=volumes,
                marker_color=colors[i % len(colors)],
                opacity=0.6 if is_demo else 0.8
            )
        )
    
    fig.update_layout(
        title=f"Volume Comparison {'📊 [DEMO]' if is_demo else ''}",
        xaxis_title="Cryptocurrency",
        yaxis_title="Volume (Millions)",
        barmode='group',
        height=400
    )
    
    return fig

def display_ai_forecast(forecast):
    """Display AI forecast"""
    if not forecast:
        return
    
    is_demo = forecast.get('source') == 'demo'
    
    st.subheader("🤖 AI Forecasts")
    
    if is_demo:
        st.warning("📊 Using demo forecast data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_4h = forecast['forecast_4h']
        color = "off" if is_demo else "normal"
        if forecast_4h == 'BULLISH':
            st.metric("4-Hour", "🟢 BULLISH", delta_color=color)
        elif forecast_4h == 'BEARISH':
            st.metric("4-Hour", "🔴 BEARISH", delta_color=color)
        else:
            st.metric("4-Hour", "🟡 NEUTRAL", delta_color=color)
    
    with col2:
        forecast_24h = forecast['forecast_24h']
        if forecast_24h == 'BULLISH':
            st.metric("24-Hour", "🟢 BULLISH", delta_color=color)
        elif forecast_24h == 'BEARISH':
            st.metric("24-Hour", "🔴 BEARISH", delta_color=color)  
        else:
            st.metric("24-Hour", "🟡 NEUTRAL", delta_color=color)
    
    st.subheader("🔍 AI Analysis")
    
    with st.expander("View Detailed Analysis", expanded=True):
        if is_demo:
            st.warning("📊 This is demo analysis data")
        st.text(forecast['full_analysis'])
    
    risk_color = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
    risk_emoji = risk_color.get(forecast['risk_level'], "🟡")
    st.metric("Risk Level", f"{risk_emoji} {forecast['risk_level']}", delta_color="off" if is_demo else "normal")
    
    st.caption(f"Analysis updated: {forecast['timestamp']}")

def main():
    st.title("📈 Crypto Analysis Dashboard")
    st.markdown("Multi-coin cryptocurrency analysis with AI-powered forecasting")
    
    # Display current data source
    source = st.session_state.data_source
    source_status = {
        "binance": "✅ Binance API",
        "cryptocompare": "⚠️ CryptoCompare",
        "demo": "📊 Demo Data",
        "binance_blocked": "❌ Binance Blocked"
    }
    
    st.sidebar.info(f"**Data Source**: {source_status.get(source, 'Unknown')}")
    
    if source == "demo":
        st.sidebar.warning("📊 Using demo data - APIs blocked")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Dashboard Controls")
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Coin selection
        st.markdown("---")
        st.subheader("🎯 Select Coin")
        selected = st.selectbox(
            "Choose cryptocurrency:",
            options=list(CRYPTO_SYMBOLS.keys()),
            format_func=lambda x: CRYPTO_SYMBOLS[x],
            index=list(CRYPTO_SYMBOLS.keys()).index(st.session_state.selected_coin)
        )
        st.session_state.selected_coin = selected
        
        # AI Settings
        st.markdown("---")
        st.subheader("🤖 AI Settings")
        
        groq_key = get_groq_api_key()
        if groq_key:
            st.success("Groq API Connected")
            st.info("AI forecasts available on demand")
        else:
            st.warning("Set GROQ_API_KEY for AI forecasts")
            with st.expander("Setup Instructions"):
                st.code("export GROQ_API_KEY='your-groq-key'")
                st.code("# or in Streamlit Cloud Secrets:")
                st.code("GROQ_API_KEY = 'your-groq-key'")
        
        st.info("**Chart Update**: Every 5 minutes")
    
    # Main content
    # Row 1: Chart and current analysis
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Coin selection buttons
        st.subheader("📊 Candlestick Chart")
        
        # Create buttons for coin selection
        cols = st.columns(5)
        for i, (symbol, name) in enumerate(CRYPTO_SYMBOLS.items()):
            if cols[i].button(f"{name.split()[0]}", key=f"btn_{symbol}"):
                st.session_state.selected_coin = symbol
                st.rerun()
        
        # Load chart data
        coin_name = CRYPTO_SYMBOLS.get(st.session_state.selected_coin, st.session_state.selected_coin)
        with st.spinner(f"Loading {coin_name} data..."):
            df = fetch_candlestick_data(st.session_state.selected_coin)
        
        if not df.empty:
            fig = create_candlestick_chart(df, coin_name)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not load chart data")
    
    with col2:
        st.subheader("💹 Current Metrics")
        
        analysis = get_crypto_analysis(st.session_state.selected_coin)
       # In the metrics section, replace the display code with:

        if analysis:
            is_demo = analysis.get('source') == 'demo'
            is_fallback = analysis.get('source') == 'binance_candle'
            price = float(analysis['price'])
            change = float(analysis['change_24h'])
            volume = float(analysis['volume_24h'])
            
            # Style for different data sources
            if is_demo:
                price_style = "color: #ff6b6b; text-decoration: line-through;"
                source_text = "📊 Demo data"
            elif is_fallback:
                price_style = "color: #f9c74f;"  # Yellow for fallback
                source_text = "⚠️ Estimated from chart data"
            else:
                price_style = ""
                source_text = "✅ Live data"
            
            st.markdown(f"""
            <div style="{price_style}">
                <h3 style="margin-bottom: 0;">${price:,.2f}</h3>
                <p style="color: {'#ff6b6b' if is_demo else ('#4ecdc4' if change >= 0 else '#ff6b6b')}; 
                        margin: 0;">
                    {change:+.2f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric(
                label="24h Volume",
                value=f"{volume/1000:.0f}K",
                delta_color="off" if is_demo else "normal"
            )
            
            rsi = analysis['rsi']
            rsi_color = "#ff6b6b" if rsi > 70 else "#4ecdc4" if rsi < 30 else "#f9c74f"
            st.markdown(f"""
            <div style="text-align: center;">
                <p style="margin: 0; font-size: 0.9em;">RSI</p>
                <h3 style="margin: 0; color: {rsi_color};">
                    {rsi} {"🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"}
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"Updated: {analysis['timestamp']}")
            st.caption(source_text) 
    
    # Row 2: AI Forecasting (collapsible)
    st.markdown("---")
    
    # Collapsible AI section
    if st.button("🤖 AI Forecasting ▶", key="ai_toggle"):
        st.session_state.show_ai_forecast = not st.session_state.show_ai_forecast
    
    if st.session_state.show_ai_forecast:
        if st.button("🔄 Generate AI Forecast", type="primary"):
            with st.spinner("Generating AI forecast..."):
                forecast = get_ai_forecast()
            display_ai_forecast(forecast)
        else:
            st.info("Click the button above to generate an AI forecast")
    
    # Row 3: Volume comparison
    st.markdown("---")
    st.subheader("📊 Volume Comparison")
    
    with st.spinner("Loading volume data..."):
        volume_data = fetch_crypto_volumes()
    
    if volume_data:
        fig = create_volume_chart(volume_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.caption("⚠️ Educational purposes only. Not financial advice.")

if __name__ == "__main__":
    main()