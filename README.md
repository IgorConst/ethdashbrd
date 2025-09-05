# Crypto Analysis Dashboard 📈

A real-time cryptocurrency analysis dashboard with AI-powered forecasting capabilities.

## Features

- **Multi-Coin Support**: ETH, BTC, BNB, ADA, SOL
- **Live Candlestick Charts**: 15-minute intervals, 24-hour view
- **Real-time Metrics**: Price, 24h change, volume, RSI
- **AI Forecasting**: On-demand market analysis using Groq AI
- **Volume Comparison**: Top 5 cryptocurrencies comparison
- **Robust Fallbacks**: Automatic demo data when APIs are unavailable

## Data Sources

- **Primary**: Binance API (real-time trading data)
- **Fallback**: CryptoCompare API
- **AI**: Groq API (Llama 3.1 model)
- **Demo Data**: Generated when APIs are blocked

## Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ethdashbrd

Install dependencies

bash
pip install -r requirements.txt
Set up API keys (optional for AI features)

Create .streamlit/secrets.toml:

toml
GROQ_API_KEY = "your-groq-api-key-here"
Run locally

bash
streamlit run dashboard.py
Deployment
Deployed on Streamlit Cloud with automatic GitHub integration:

🔗 Live Demo: https://ethdashbrd-6ks2mtjhawdsuf6rhhdaog.streamlit.app/

Usage
Select Coin: Use buttons or dropdown to switch cryptocurrencies

Refresh Data: Click refresh button for latest data

AI Forecast: Click "Generate AI Forecast" for market analysis

Volume Comparison: View comparative volumes of top cryptocurrencies

Technology Stack
Frontend: Streamlit, Plotly

Data APIs: Binance, CryptoCompare

AI: Groq API (Llama 3.1)

Caching: Streamlit data caching

Error Handling: Robust fallback system

Note
⚠️ Educational purposes only - Not financial advice. Use demo data when APIs are blocked by cloud providers.

License
MIT License - Feel free to use and modify for your projects.

text

This README provides:
- Clear feature overview
- Setup instructions
- Deployment information
- Usage guidelines
- Technology stack
- Important disclaimers

It's concise but covers all essential information for users and developers!