import streamlit as st

# Simple test page
st.title("🧪 Streamlit Test")
st.write("If you can see this, Streamlit is working!")
st.success("✅ Success - Basic Streamlit functionality works")

# Test if we can import other libraries
try:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import requests
    
    st.success("✅ All required libraries imported successfully")
    
    # Test basic functionality
    st.subheader("📊 Test Chart")
    
    # Simple test data
    import random
    data = [random.randint(1, 100) for _ in range(10)]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(y=data, name='Test Data'))
    fig.update_layout(title="Test Chart", height=400)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("✅ Plotly charts working")
    
    # Test API call
    st.subheader("🔌 Test API Connection")
    
    try:
        response = requests.get("https://api.binance.com/api/v3/ping", timeout=5)
        if response.status_code == 200:
            st.success("✅ Binance API connection working")
        else:
            st.error(f"❌ Binance API error: {response.status_code}")
    except Exception as e:
        st.error(f"❌ API connection failed: {e}")

except ImportError as e:
    st.error(f"❌ Missing library: {e}")
    st.info("Run: pip install plotly pandas numpy requests")

except Exception as e:
    st.error(f"❌ Unexpected error: {e}")