import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io

from data_engine import fetch_stock_data, add_technical_indicators, get_company_info
from model_engine import LSTMModel
from scoring_engine import calculate_investment_score

# Page Config
st.set_page_config(page_title="AI Stock Predictor & Analyst", layout="wide")

# Title
st.title("📈 AI-Powered Stock Price Prediction & Investment Analysis")
st.markdown("### Using LSTM Neural Networks + Technical Indicators + Advanced Scoring")

# Sidebar
st.sidebar.header("Prediction Settings")

# Asset Class Selection
asset_class = st.sidebar.radio("Select Asset Class", ["US Stocks", "Singapore (SGX)", "Cryptocurrencies", "Commodities"])

# Ticker Databases
us_tickers = {
    "Technology": {"AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "GOOGL": "Alphabet Inc.", "NVDA": "NVIDIA Corporation", "TSLA": "Tesla, Inc.", "META": "Meta Platforms, Inc.", "AVGO": "Broadcom Inc.", "ORCL": "Oracle Corporation", "ADBE": "Adobe Inc.", "CRM": "Salesforce, Inc."},
    "Finance": {"JPM": "JPMorgan Chase & Co.", "BAC": "Bank of America Corp.", "GS": "Goldman Sachs Group", "MS": "Morgan Stanley", "V": "Visa Inc.", "MA": "Mastercard Incorporated", "WFC": "Wells Fargo & Company", "C": "Citigroup Inc.", "BLK": "BlackRock, Inc.", "AXP": "American Express Company"},
    "Healthcare": {"JNJ": "Johnson & Johnson", "PFE": "Pfizer Inc.", "UNH": "UnitedHealth Group", "ABBV": "AbbVie Inc.", "MRK": "Merck & Co., Inc.", "LLY": "Eli Lilly and Company", "TMO": "Thermo Fisher Scientific", "DHR": "Danaher Corporation", "ABT": "Abbott Laboratories", "BMY": "Bristol-Myers Squibb"},
    "Consumer": {"AMZN": "Amazon.com, Inc.", "WMT": "Walmart Inc.", "KO": "Coca-Cola Company", "PEP": "PepsiCo, Inc.", "NKE": "NIKE, Inc.", "COST": "Costco Wholesale Corp.", "MCD": "McDonald's Corporation", "PG": "Procter & Gamble Co.", "HD": "Home Depot, Inc.", "DIS": "Walt Disney Company"},
    "Energy": {"XOM": "Exxon Mobil Corporation", "CVX": "Chevron Corporation", "COP": "ConocoPhillips", "SLB": "Schlumberger Limited", "EOG": "EOG Resources, Inc.", "MPC": "Marathon Petroleum Corp.", "PSX": "Phillips 66", "VLO": "Valero Energy Corp.", "OXY": "Occidental Petroleum", "HES": "Hess Corporation"}
}

sg_tickers = {
    "Finance": {"D05.SI": "DBS Group Holdings", "O39.SI": "OCBC Bank", "U11.SI": "United Overseas Bank (UOB)", "S68.SI": "Singapore Exchange (SGX)", "G13.SI": "Genting Singapore"},
    "REITs": {"C38U.SI": "CapitaLand Integrated Commercial Trust", "A17U.SI": "Ascendas REIT", "M44U.SI": "Mapletree Logistics Trust", "N2IU.SI": "Mapletree Pan Asia Commercial Trust", "ME8U.SI": "Mapletree Industrial Trust", "K71U.SI": "Keppel REIT", "AJBU.SI": "Frasers Centrepoint Trust"},
    "Technology/Telco": {"Z74.SI": "Singtel", "V03.SI": "Venture Corporation", "U96.SI": "Sembcorp Industries", "CC3.SI": "StarHub", "A13.SI": "Ascent Bridge"},
    "Consumer/Industrial": {"C31.SI": "CapitaLand Investment", "U06.SI": "UOL Group", "F34.SI": "Wilmar International", "C07.SI": "Jardine Cycle & Carriage", "J36.SI": "Jardine Matheson", "BS6.SI": "Yangzijiang Shipbuilding", "BN4.SI": "Keppel Ltd.", "S63.SI": "Singapore Technologies Engineering"}
}

crypto_tickers = {
    "Major Cryptos": {
        "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "BNB-USD": "Binance Coin", "SOL-USD": "Solana", "XRP-USD": "XRP",
        "ADA-USD": "Cardano", "DOGE-USD": "Dogecoin", "AVAX-USD": "Avalanche", "DOT-USD": "Polkadot", "TRX-USD": "TRON",
        "LINK-USD": "Chainlink", "MATIC-USD": "Polygon", "SHIB-USD": "Shiba Inu", "LTC-USD": "Litecoin", "BCH-USD": "Bitcoin Cash",
        "UNI-USD": "Uniswap", "NEAR-USD": "NEAR Protocol", "LEO-USD": "UNUS SED LEO", "DAI-USD": "Dai", "STX-USD": "Stacks"
    }
}

commodity_tickers = {
    "Metals": {
        "GC=F": "Gold", "SI=F": "Silver", "PL=F": "Platinum", "HG=F": "Copper", "PA=F": "Palladium"
    },
    "Energy": {
        "CL=F": "Crude Oil (WTI)", "BZ=F": "Brent Crude Oil", "NG=F": "Natural Gas", "RB=F": "RBOB Gasoline"
    },
    "Uranium": {
        "URA": "Global X Uranium ETF (Proxy)", "CCJ": "Cameco Corporation (Uranium Miner)"
    }
}

# Selection Logic
if asset_class == "US Stocks":
    selected_sector = st.sidebar.selectbox("Select Sector", list(us_tickers.keys()))
    stock_dict = us_tickers[selected_sector]
elif asset_class == "Singapore (SGX)":
    selected_sector = st.sidebar.selectbox("Select Sector", list(sg_tickers.keys()))
    stock_dict = sg_tickers[selected_sector]
elif asset_class == "Cryptocurrencies":
    selected_sector = st.sidebar.selectbox("Select Category", list(crypto_tickers.keys()))
    stock_dict = crypto_tickers[selected_sector]
else: # Commodities
    selected_sector = st.sidebar.selectbox("Select Category", list(commodity_tickers.keys()))
    stock_dict = commodity_tickers[selected_sector]

# Create labels and select stock
stock_options = [f"{ticker} - {name}" for ticker, name in stock_dict.items()]
selected_option = st.sidebar.selectbox("Select Asset", stock_options)
selected_stock = selected_option.split(" - ")[0]

# Manual Entry Option
manual_ticker = st.sidebar.text_input("Or Enter Custom Ticker (e.g., AMD, BTC-USD, GC=F)")
if manual_ticker:
    selected_stock = manual_ticker.upper()

# Date Selection
end_date = datetime.now()
start_date = end_date - timedelta(days=365*2)
start_date_input = st.sidebar.date_input("Start Date", start_date)
end_date_input = st.sidebar.date_input("End Date", end_date)

# Model Parameters
st.sidebar.subheader("Model Parameters")
epochs = st.sidebar.slider("Epochs", 5, 50, 5)
lookback = st.sidebar.slider("Lookback Days", 30, 90, 60)
future_days = st.sidebar.slider("Predict Next N Days", 5, 30, 15)

# Main App Logic
if st.sidebar.button("Run Analysis"):
    with st.spinner(f"Fetching data for {selected_stock}..."):
        df = fetch_stock_data(selected_stock, start_date_input, end_date_input)
        info = get_company_info(selected_stock)
        
    if df is not None and not df.empty:
        df = add_technical_indicators(df)
        
        # 1. Asset Data Preview
        st.subheader(f"Asset Data Preview: {selected_stock} ({info.get('longName', selected_stock)})")
        st.markdown(f"**Category:** {selected_sector} | **Type:** {asset_class}")
        st.dataframe(df.tail(10))
        
        # 2. Interactive Charts
        st.subheader("Technical Analysis Dashboard")
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close Price"), row=1, col=1)
        if 'BB_High' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], name="BB High", line=dict(dash='dash', color='rgba(173, 216, 230, 0.5)')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], name="BB Low", line=dict(dash='dash', color='rgba(173, 216, 230, 0.5)')), row=1, col=1)
        if 'SMA_50' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name="SMA 50"), row=1, col=1)
        if 'RSI' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name="RSI"), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        if 'MACD' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name="MACD"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], name="Signal"), row=3, col=1)
        fig.update_layout(height=800, title_text=f"{selected_stock} Technical Indicators")
        st.plotly_chart(fig, use_container_width=True)
        
        # 3. LSTM Prediction
        st.subheader("AI Price Prediction (LSTM)")
        lstm = LSTMModel(lookback=lookback)
        close_prices = df['Close'].values
        with st.spinner("Training LSTM Model..."):
            lstm.train(close_prices, epochs=epochs)
            hist_preds = lstm.get_historical_predictions(close_prices)
            future_preds = lstm.predict_future(close_prices, days_to_predict=future_days)
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=df.index[lookback:], y=close_prices[lookback:], name="Actual Price"))
        fig_pred.add_trace(go.Scatter(x=df.index[lookback:], y=hist_preds.flatten(), name="Model Fit"))
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(future_days)]
        fig_pred.add_trace(go.Scatter(x=future_dates, y=future_preds.flatten(), name="Future Prediction", line=dict(dash='dash', color='red')))
        fig_pred.update_layout(title=f"{selected_stock} Price Prediction", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # 4. Investment Scoring (Note: Fundamental scoring might be limited for Crypto/Commodities)
        st.subheader("Comprehensive Investment Analysis")
        total_score, rating, factor_scores = calculate_investment_score(info, df)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Overall Investment Score", f"{total_score:.2f}/100")
            st.markdown(f"### Recommendation: **{rating}**")
            if asset_class in ["US Stocks", "Singapore (SGX)"]:
                metrics_data = {
                    "Metric": ["Market Cap", "P/E Ratio", "P/B Ratio", "ROE", "52W High", "52W Low"],
                    "Value": [
                        f"${info.get('marketCap', 0):,.0f}",
                        f"{info.get('trailingPE', 0):.2f}",
                        f"{info.get('priceToBook', 0):.2f}",
                        f"{info.get('returnOnEquity', 0)*100:.2f}%" if isinstance(info.get('returnOnEquity'), (int, float)) else "N/A",
                        f"${info.get('fiftyTwoWeekHigh', 0):.2f}",
                        f"${info.get('fiftyTwoWeekLow', 0):.2f}"
                    ]
                }
                st.table(pd.DataFrame(metrics_data))
            else:
                st.info("Fundamental metrics (P/E, ROE) are not applicable for Cryptocurrencies or Commodities. Analysis is based on Momentum, Risk, and Liquidity.")
        with col2:
            fig_score = go.Figure(go.Bar(x=list(factor_scores.values()), y=list(factor_scores.keys()), orientation='h', marker_color='skyblue'))
            fig_score.update_layout(title="Factor Score Breakdown", xaxis_title="Score", xaxis_range=[0, 100])
            st.plotly_chart(fig_score, use_container_width=True)
            
        # 5. Export Data
        st.subheader("Export Analysis")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Historical Data & Indicators')
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Price': future_preds.flatten()})
            future_df.to_excel(writer, sheet_name='Future Predictions', index=False)
            scores_df = pd.DataFrame(list(factor_scores.items()), columns=['Factor', 'Score'])
            scores_df.to_excel(writer, sheet_name='Investment Scores', index=False)
        st.download_button(label="Download Analysis as Excel", data=output.getvalue(), file_name=f"{selected_stock}_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.error("Could not fetch data. Please check the ticker symbol or date range.")
else:
    st.info("Select an asset and click 'Run Analysis' in the sidebar to begin.")
