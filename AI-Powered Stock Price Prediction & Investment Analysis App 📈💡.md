# 📈 AI-Powered Stock Predictor & Investment Analyst

A comprehensive End-to-End financial analysis dashboard built with Streamlit, LSTM Neural Networks, and the 8-Factor Investment Scoring Model.

## 🚀 Key Features

- **Multi-Asset Support:** Analyze US Stocks, Singapore (SGX) Stocks, Top 20 Cryptocurrencies, and Major Commodities (Gold, Oil, Uranium, etc.).
- **AI Forecasting:** LSTM (Long Short-Term Memory) neural network for future price prediction (5–30 days).
- **Technical Analysis:** 15+ indicators including RSI, MACD, Bollinger Bands, ADX, OBV, and more.
- **8-Factor Scoring:** Comprehensive investment rating (Strong Buy to Strong Sell) based on Financials, Valuation, Growth, Risk, Momentum, Liquidity, Management, and Industry.
- **Interactive UI:** Dynamic Plotly charts with full company names and sector-based navigation.
- **Data Export:** Download full analysis reports as Excel files.

---

## 🐳 Deployment with Docker (Recommended)

The easiest way to run the app without worrying about Python versions or library conflicts.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### Steps
1. **Download the project files** into a folder.
2. **Open your terminal** in that folder.
3. **Run the command:**
   ```bash
   docker-compose up --build
   ```
4. **Access the app:** Open your browser to `http://localhost:8501`.

---

## 🐍 Manual Setup (Python 3.9 - 3.12)

If you prefer to run it directly on your machine, use a virtual environment to avoid conflicts (especially if you have Python 3.14).

### Using Conda
```bash
# 1. Create environment
conda create -n stock_app python=3.11 -y

# 2. Activate
conda activate stock_app

# 3. Install dependencies
python -m pip install -r requirements.txt

# 4. Run
streamlit run app.py
```

### Using venv
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Project Structure

- `app.py`: Main Streamlit UI and dashboard logic.
- `data_engine.py`: Data fetching (yFinance) and technical indicator calculations.
- `model_engine.py`: Optimized LSTM model training and prediction.
- `scoring_engine.py`: 8-factor investment scoring logic.
- `Dockerfile` & `docker-compose.yml`: Containerization settings.
- `requirements.txt`: Python library dependencies.

---

## 💡 Tips for Accuracy
- **Epochs:** Increase epochs (20-50) for deeper pattern recognition, or keep them low (5-10) for faster results.
- **Lookback:** Adjust lookback days to change how much historical context the AI considers for each prediction.
- **Model Fit:** Always compare the "Model Fit" line to the "Actual Price" to see how well the AI has learned that specific asset's behavior.

---

## ⚠️ Disclaimer
This application is for **educational and experimental purposes only**. Financial markets involve significant risk. Always perform your own due diligence before making investment decisions.
