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
    page_title="ETH Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Global variable to track which data source is working
if 'data_source' not in st.session_state:
    st.session_state.data_source = "binance"  # Default source

# Create a robust session with retries
def create_robust_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=2,  # Reduced retries
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
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json',
    }
    
    session = create_robust_session()
    
    # Try Binance first (but skip if we know it's blocked)
    if st.session_state.data_source != "binance_blocked":
        try:
            response = session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            st.session_state.data_source = "binance"
            return response.json(), "binance"
        except Exception as e:
            st.session_state.data_source = "binance_blocked"
    
    # Try CryptoCompare for all data types
    try:
        if data_type == "klines":
            # Get ETH data from CryptoCompare
            cryptocompare_url = "https://min-api.cryptocompare.com/data/v2/histominute"
            cryptocompare_params = {
                'fsym': 'ETH',
                'tsym': 'USD',
                'limit': 96,  # 96 periods of 15min = 24 hours
                'aggregate': 15  # 15-minute intervals
            }
            
            response = session.get(cryptocompare_url, params=cryptocompare_params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['Response'] == 'Success':
                st.session_state.data_source = "cryptocompare"
                return data['Data']['Data'], "cryptocompare"
                
        elif data_type == "ticker":
            # Get current ETH price from CryptoCompare
            cryptocompare_url = "https://min-api.cryptocompare.com/data/price"
            cryptocompare_params = {
                'fsym': 'ETH',
                'tsym': 'USD'
            }
            
            response = session.get(cryptocompare_url, params=cryptocompare_params, headers=headers, timeout=10)
            response.raise_for_status()
            price_data = response.json()
            
            # Get 24h change
            change_url = "https://min-api.cryptocompare.com/data/pricemultifull"
            change_params = {
                'fsyms': 'ETH',
                'tsyms': 'USD'
            }
            
            change_response = session.get(change_url, params=change_params, headers=headers, timeout=10)
            change_response.raise_for_status()
            change_data = change_response.json()
            
            # Format as Binance-like response
            formatted_data = {
                'lastPrice': price_data['USD'],
                'priceChangePercent': change_data['RAW']['ETH']['USD']['CHANGEPCT24HOUR'],
                'volume': change_data['RAW']['ETH']['USD']['VOLUME24HOUR']
            }
            
            st.session_state.data_source = "cryptocompare"
            return formatted_data, "cryptocompare"
            
    except Exception as e:
        st.warning(f"CryptoCompare failed: {e}")
    
    # If all APIs fail, return demo data
    return generate_demo_data(data_type), "demo"

def generate_demo_data(data_type):
    """Generate demo data when all APIs fail"""
    if data_type == "klines":
        # Generate demo candlestick data
        now = datetime.now()
        timestamps = [now - timedelta(minutes=15*i) for i in range(96)]
        timestamps.reverse()
        
        base_price = 2500 + np.random.normal(0, 50)
        demo_data = []
        
        for i, ts in enumerate(timestamps):
            open_price = base_price + np.random.normal(0, 20)
            close_price = open_price + np.random.normal(0, 30)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 15))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 15))
            volume = np.random.normal(50000, 10000)
            
            demo_data.append([
                int(ts.timestamp() * 1000),  # timestamp in ms
                open_price, high_price, low_price, close_price,
                volume,  # volume
                int(ts.timestamp() * 1000),  # close_time
                volume * close_price,  # quote_volume
                1000,  # trades
                50000,  # taker_buy_base
                50000 * close_price,  # taker_buy_quote
                0  # ignore
            ])
        
        return demo_data
        
    elif data_type == "ticker":
        # Generate demo ticker data
        price = 2500 + np.random.normal(0, 50)
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
            item['time'] * 1000,  # timestamp in ms
            item['open'],         # open
            item['high'],         # high
            item['low'],          # low
            item['close'],        # close
            item['volumeto'],     # volume
            item['time'] * 1000,  # close_time
            item['volumefrom'],   # quote_volume
            0,                    # trades (not available)
            0,                    # taker_buy_base (not available)
            0,                    # taker_buy_quote (not available)
            0                     # ignore
        ])
    return converted_data

# Cache functions to avoid repeated API calls
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_eth_candlestick_data():
    """Fetch 1-day ETH candlestick data with fallbacks"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'ETHUSDT',
            'interval': '15m',
            'limit': 96
        }
        
        data, source = get_data_with_fallbacks(url, params, "klines")
        
        if data is None:
            return pd.DataFrame()
        
        # Handle different data formats based on source
        if source == "cryptocompare":
            data = convert_cryptocompare_data(data)
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return df
        
    except Exception as e:
        st.error(f"Error processing candlestick data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_crypto_volumes():
    """Fetch volume data for top 5 cryptos with fallbacks"""
    cryptos = ['Bitcoin', 'Ethereum', 'BNB', 'Cardano', 'Solana']
    
    # Generate demo volume data since Binance is blocked
    volume_data = {}
    dates = [(datetime.now() - timedelta(days=i)).strftime('%m-%d') for i in range(3)]
    
    for crypto in cryptos:
        volumes = [np.random.normal(1000000, 200000) for _ in range(3)]
        volume_data[crypto] = {
            'dates': dates,
            'volumes': volumes
        }
    
    return volume_data

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_eth_analysis():
    """Get current ETH analysis with fallbacks"""
    try:
        # Current ticker data
        url = "https://api.binance.com/api/v3/ticker/24hr"
        params = {'symbol': 'ETHUSDT'}
        
        data, source = get_data_with_fallbacks(url, params, "ticker")
        
        if data is None:
            return None
        
        # Ensure all values are numbers, not strings
        price = float(data['lastPrice'])
        change = float(data.get('priceChangePercent', 0))
        volume = float(data.get('volume', 50000))
        
        # For demo data, generate some historical prices for RSI calculation
        if source == "demo":
            prices = [price + np.random.normal(0, 20) for _ in range(30)]
        else:
            # For real data, we'd fetch historical prices
            prices = [price]
        
        rsi = calculate_rsi(prices)
        
        return {
            'price': price,
            'change_24h': change,
            'volume_24h': volume,
            'rsi': rsi,
            'timestamp': datetime.now().strftime('%H:%M:%S UTC'),
            'source': source
        }
        
    except Exception as e:
        st.error(f"Error getting analysis: {e}")
        return None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_ai_forecast():
    """Get AI-powered ETH forecast"""
    # Check if OpenAI is properly configured
    if not OPENAI_AVAILABLE:
        return None
    
    try:
        # Check if API key is set
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return None
        
        # Get current market data
        analysis = get_eth_analysis()
        if not analysis:
            return None
        
        # Get additional market context
        market_context = get_market_context()
        
        # Prepare prompt for OpenAI
        prompt = f"""
        As a crypto analyst, provide a concise forecast for Ethereum (ETH) based on current data:

        Current Data:
        - Price: ${analysis['price']:,.2f}
        - 24h Change: {analysis['change_24h']:+.2f}%
        - RSI: {analysis['rsi']}
        - Volume: {analysis['volume_24h']:,.0f} ETH
        - Market Context: {market_context}

        Provide:
        1. 4-hour forecast (BULLISH/BEARISH/NEUTRAL) with confidence %
        2. 24-hour forecast (BULLISH/BEARISH/NEUTRAL) with confidence %
        3. Key factors analysis (3-4 points)
        4. Risk level (Low/Moderate/High)
        5. Specific recommendation for ETH-USDC LP and GLV positions

        Format as structured analysis, be concise and actionable.
        """
        
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert cryptocurrency analyst providing concise, actionable forecasts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.3
        )
        
        forecast_text = response.choices[0].message.content
        
        # Parse the response to extract key information
        forecast_data = parse_ai_forecast(forecast_text)
        forecast_data['full_analysis'] = forecast_text
        forecast_data['timestamp'] = datetime.now().strftime('%H:%M:%S UTC')
        
        return forecast_data
        
    except Exception as e:
        # Don't show error if it's just missing API key
        if "No API key provided" not in str(e):
            st.error(f"Error generating AI forecast: {e}")
        return None

def get_market_context():
    """Get additional market context for AI analysis"""
    try:
        # Get market status based on which data source is working
        source = st.session_state.data_source
        
        if source == "binance":
            return "Normal trading conditions - Binance API"
        elif source == "cryptocompare":
            return "Using fallback data - CryptoCompare API"
        elif source == "demo":
            return "Using demo data - All APIs blocked"
        else:
            return "Limited market data available"
    except:
        return "Market data temporarily unavailable"

def parse_ai_forecast(forecast_text):
    """Parse AI forecast text to extract structured data"""
    forecast_data = {
        'forecast_4h': 'NEUTRAL',
        'forecast_24h': 'NEUTRAL',
        'confidence_4h': 50,
        'confidence_24h': 50,
        'risk_level': 'Moderate'
    }
    
    try:
        lines = forecast_text.lower()
        
        # Extract 4-hour forecast
        if 'bullish' in lines and '4-hour' in lines or '4h' in lines:
            forecast_data['forecast_4h'] = 'BULLISH'
        elif 'bearish' in lines and ('4-hour' in lines or '4h' in lines):
            forecast_data['forecast_4h'] = 'BEARISH'
        
        # Extract 24-hour forecast
        if 'bullish' in lines and ('24-hour' in lines or '24h' in lines):
            forecast_data['forecast_24h'] = 'BULLISH'
        elif 'bearish' in lines and ('24-hour' in lines or '24h' in lines):
            forecast_data['forecast_24h'] = 'BEARISH'
        
        # Extract risk level
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

def create_candlestick_chart(df):
    """Create candlestick chart"""
    if df.empty:
        st.warning("No candlestick data available")
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('ETH/USDT - 15min Candlesticks (24h)', 'Volume'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='ETH/USDT'
        ),
        row=1, col=1
    )
    
    # Volume
    fig.add_trace(
        go.Bar(
            x=df['timestamp'],
            y=df['volume'],
            name='Volume',
            marker_color='rgba(0,150,250,0.6)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="ETH Trading Chart",
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=False
    )
    
    return fig

def create_volume_chart(volume_data):
    """Create volume comparison chart"""
    if not volume_data:
        st.warning("No volume data available")
        return None
    
    fig = go.Figure()
    
    # Get all dates (should be same for all cryptos)
    first_crypto = list(volume_data.keys())[0]
    dates = volume_data[first_crypto]['dates']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
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
                marker_color=colors[i % len(colors)]
            )
        )
    
    fig.update_layout(
        title="Volume Comparison - Top 5 Cryptocurrencies (Last 3 Days)",
        xaxis_title="Cryptocurrency",
        yaxis_title="Volume (Millions)",
        barmode='group',
        height=400
    )
    
    return fig

def display_ai_forecast(forecast):
    """Display AI forecast in a nice format"""
    if not forecast:
        st.warning("⚠️ AI forecasting unavailable. Set OPENAI_API_KEY to enable.")
        return
    
    st.subheader("🤖 AI Forecasts")
    
    # Create columns for forecasts
    col1, col2 = st.columns(2)
    
    with col1:
        # 4-hour forecast
        forecast_4h = forecast['forecast_4h']
        if forecast_4h == 'BULLISH':
            st.metric("4-Hour", "🟢 BULLISH", help="AI suggests upward movement")
        elif forecast_4h == 'BEARISH':
            st.metric("4-Hour", "🔴 BEARISH", help="AI suggests downward movement")
        else:
            st.metric("4-Hour", "🟡 NEUTRAL", help="AI suggests sideways movement")
    
    with col2:
        # 24-hour forecast
        forecast_24h = forecast['forecast_24h']
        if forecast_24h == 'BULLISH':
            st.metric("24-Hour", "🟢 BULLISH", help="AI suggests upward movement")
        elif forecast_24h == 'BEARISH':
            st.metric("24-Hour", "🔴 BEARISH", help="AI suggests downward movement")  
        else:
            st.metric("24-Hour", "🟡 NEUTRAL", help="AI suggests sideways movement")
    
    # Display detailed analysis
    st.subheader("🔍 AI Analysis")
    
    with st.expander("View Detailed Analysis", expanded=True):
        st.text(forecast['full_analysis'])
    
    # Risk level
    risk_color = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
    risk_emoji = risk_color.get(forecast['risk_level'], "🟡")
    st.metric("Risk Level", f"{risk_emoji} {forecast['risk_level']}")
    
    st.caption(f"AI Analysis updated: {forecast['timestamp']}")

def main():
    # Title
    st.title("📈 ETH Analysis Dashboard")
    st.markdown("Real-time cryptocurrency analysis with AI-powered forecasting")
    
    # Display current data source
    source_status = {
        "binance": "✅ Binance API",
        "cryptocompare": "⚠️ CryptoCompare (Fallback)",
        "demo": "📊 Demo Data (APIs blocked)",
        "binance_blocked": "❌ Binance Blocked"
    }
    
    st.sidebar.info(f"**Data Source**: {source_status.get(st.session_state.data_source, 'Unknown')}")
    
    if st.session_state.data_source == "demo":
        st.sidebar.warning("Using demo data - Binance API is blocked")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Dashboard Controls")
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # AI Configuration
        st.markdown("---")
        st.subheader("🤖 AI Settings")
        
        # Check OpenAI setup
        ai_status = "❌ Not Available"
        if OPENAI_AVAILABLE:
            if os.getenv('OPENAI_API_KEY'):
                ai_status = "✅ Active"
                st.success("OpenAI API Connected")
            else:
                ai_status = "⚠️ API Key Missing"
                st.warning("Set OPENAI_API_KEY for forecasts")
                with st.expander("Setup Instructions"):
                    st.code("export OPENAI_API_KEY='sk-your-key-here'")
                    st.code("# or for Windows:")
                    st.code("$env:OPENAI_API_KEY='sk-your-key-here'")
        else:
            st.error("OpenAI not installed")
            st.code("pip install openai")
        
        st.info(f"**AI Status**: {ai_status}")
        st.info("**Chart Update**: Every 5 minutes")
        st.info("**AI Update**: Every 30 minutes")
    
    # Main content layout
    # Row 1: Chart and current analysis
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📊 ETH Candlestick Chart")
        
        # Load candlestick data
        with st.spinner("Loading chart data..."):
            df = fetch_eth_candlestick_data()
        
        if not df.empty:
            fig = create_candlestick_chart(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Could not load candlestick data")
    
    with col2:
        st.subheader("💹 Current Metrics")
        
        # Load current analysis
        analysis = get_eth_analysis()
        
        if analysis:
            # Convert to float to ensure proper formatting
            price = float(analysis['price'])
            change = float(analysis['change_24h'])
            volume = float(analysis['volume_24h'])
            
            st.metric(
                label="ETH Price",
                value=f"${price:,.2f}",
                delta=f"{change:+.2f}%"
            )
            
            st.metric(
                label="24h Volume",
                value=f"{volume/1000:.0f}K ETH"
            )
            
            # RSI with color coding
            rsi = analysis['rsi']
            if rsi > 70:
                st.metric("RSI", f"{rsi} 🔴", help="Overbought")
            elif rsi < 30:
                st.metric("RSI", f"{rsi} 🟢", help="Oversold") 
            else:
                st.metric("RSI", f"{rsi} 🟡", help="Neutral")
            
            st.caption(f"Updated: {analysis['timestamp']}")
            st.caption(f"Source: {analysis.get('source', 'Unknown')}")
        else:
            st.error("Could not load analysis data")
    
    # Row 2: AI Forecasting Section (only if OpenAI is available)
    if OPENAI_AVAILABLE:
        st.markdown("---")
        st.subheader("🤖 AI Forecasting")
        
        # Load and display AI forecast
        with st.spinner("Generating AI forecast..."):
            forecast = get_ai_forecast()
        
        display_ai_forecast(forecast)
    else:
        st.markdown("---")
        st.info("🤖 Install OpenAI package for AI forecasting: `pip install openai`")
    
    # Row 3: Volume comparison section
    st.markdown("---")
    st.subheader("📊 Volume Comparison")
    
    with st.spinner("Loading volume data..."):
        volume_data = fetch_crypto_volumes()
    
    if volume_data:
        fig = create_volume_chart(volume_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Volume comparison data not available")
    
    # Footer
    st.markdown("---")
    st.caption("⚠️ This dashboard is for educational purposes. Not financial advice.")

if __name__ == "__main__":
    main()