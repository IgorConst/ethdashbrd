import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import time

# Set page config
st.set_page_config(
    page_title="ETH Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Cache functions to avoid repeated API calls
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_eth_candlestick_data():
    """Fetch 1-day ETH candlestick data from Binance"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': 'ETHUSDT',
            'interval': '15m',  # 15-minute candles
            'limit': 96  # 24 hours
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
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
        st.error(f"Error fetching ETH candlestick data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_crypto_volumes():
    """Fetch volume data for top 5 cryptos"""
    cryptos = {
        'BTCUSDT': 'Bitcoin',
        'ETHUSDT': 'Ethereum', 
        'BNBUSDT': 'BNB',
        'ADAUSDT': 'Cardano',
        'SOLUSDT': 'Solana'
    }
    
    volume_data = {}
    
    try:
        for symbol, name in cryptos.items():
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '1d',
                'limit': 3  # Last 3 days
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            volumes = []
            dates = []
            
            for candle in data:
                dates.append(pd.to_datetime(candle[0], unit='ms').strftime('%m-%d'))
                volumes.append(float(candle[5]) / 1_000_000)  # Convert to millions
            
            volume_data[name] = {
                'dates': dates,
                'volumes': volumes
            }
            
            time.sleep(0.1)  # Rate limit protection
        
        return volume_data
        
    except Exception as e:
        st.error(f"Error fetching volume data: {e}")
        return {}

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_eth_analysis():
    """Get current ETH analysis"""
    try:
        # Current ticker data
        url = "https://api.binance.com/api/v3/ticker/24hr"
        params = {'symbol': 'ETHUSDT'}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        ticker_data = response.json()
        
        # Historical data for RSI
        kline_url = "https://api.binance.com/api/v3/klines"
        kline_params = {
            'symbol': 'ETHUSDT',
            'interval': '1h',
            'limit': 30
        }
        
        kline_response = requests.get(kline_url, params=kline_params, timeout=10)
        kline_response.raise_for_status()
        kline_data = kline_response.json()
        
        # Calculate RSI
        prices = [float(candle[4]) for candle in kline_data]
        rsi = calculate_rsi(prices)
        
        current_price = float(ticker_data['lastPrice'])
        price_change = float(ticker_data['priceChangePercent'])
        volume_24h = float(ticker_data['volume'])
        
        return {
            'price': current_price,
            'change_24h': price_change,
            'volume_24h': volume_24h,
            'rsi': rsi,
            'timestamp': datetime.now().strftime('%H:%M:%S UTC')
        }
        
    except Exception as e:
        st.error(f"Error getting analysis: {e}")
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    if len(prices) < period + 1:
        return 50
    
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

def main():
    # Title
    st.title("📈 ETH Analysis Dashboard")
    st.markdown("Real-time cryptocurrency analysis and forecasting")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Dashboard Controls")
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.info("**Data Source**: Binance API")
        st.info("**Update**: Every 5 minutes")
    
    # Main content
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
        st.subheader("🔍 Current Analysis")
        
        # Load current analysis
        analysis = get_eth_analysis()
        
        if analysis:
            st.metric(
                label="ETH Price",
                value=f"${analysis['price']:,.2f}",
                delta=f"{analysis['change_24h']:+.2f}%"
            )
            
            st.metric(
                label="24h Volume",
                value=f"{analysis['volume_24h']/1000:.0f}K ETH"
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
        else:
            st.error("Could not load analysis data")
    
    # Volume comparison section
    st.subheader("📊 Volume Comparison")
    
    with st.spinner("Loading volume data..."):
        volume_data = fetch_crypto_volumes()
    
    if volume_data:
        fig = create_volume_chart(volume_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Could not load volume comparison data")
    
    # Footer
    st.markdown("---")
    st.caption("⚠️ This dashboard is for educational purposes. Not financial advice.")

if __name__ == "__main__":
    main()