import numpy as np

def calculate_investment_score(info, df):
    """
    Calculate a comprehensive investment score (0-100) based on 8 factors.
    Factors: Financial Performance, Valuation, Growth Potential, Risk, Momentum, 
             Liquidity, Management Quality, Industry Strength.
    """
    scores = {}
    
    # 1. Financial Performance (Profitability)
    roe = info.get('returnOnEquity', 0)
    profit_margin = info.get('profitMargins', 0)
    scores['Financial Performance'] = np.clip((roe * 100 * 2 + profit_margin * 100 * 2), 0, 100)

    # 2. Valuation
    pe = info.get('trailingPE', 20)
    pb = info.get('priceToBook', 2)
    # Lower is usually better for valuation, so we invert
    pe_score = max(0, 100 - (pe * 2))
    pb_score = max(0, 100 - (pb * 10))
    scores['Valuation'] = (pe_score + pb_score) / 2

    # 3. Growth Potential
    rev_growth = info.get('revenueGrowth', 0)
    earn_growth = info.get('earningsGrowth', 0)
    scores['Growth Potential'] = np.clip((rev_growth * 100 * 2 + earn_growth * 100 * 2), 0, 100)

    # 4. Risk
    beta = info.get('beta', 1)
    debt_to_equity = info.get('debtToEquity', 100)
    # Lower beta and lower debt are better
    beta_score = max(0, 100 - (abs(beta - 1) * 50))
    debt_score = max(0, 100 - (debt_to_equity / 2))
    scores['Risk'] = (beta_score + debt_score) / 2

    # 5. Momentum (Technical)
    if not df.empty:
        # Use .iloc[-1] and handle potential series/dataframe
        cp_val = df['Close'].iloc[-1]
        current_price = float(cp_val.iloc[0] if hasattr(cp_val, 'iloc') else cp_val)
        
        s20_val = df['SMA_20'].iloc[-1] if 'SMA_20' in df.columns else current_price
        sma_20 = float(s20_val.iloc[0] if hasattr(s20_val, 'iloc') else s20_val)
        
        rsi_val = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
        rsi = float(rsi_val.iloc[0] if hasattr(rsi_val, 'iloc') else rsi_val)
        momentum_score = 50
        if current_price > sma_20: momentum_score += 25
        if 40 < rsi < 70: momentum_score += 25
        scores['Momentum'] = momentum_score
    else:
        scores['Momentum'] = 50

    # 6. Liquidity
    avg_vol = info.get('averageVolume', 1000000)
    scores['Liquidity'] = np.clip(np.log10(avg_vol + 1) * 10, 0, 100)

    # 7. Management Quality (Proxy via ROA and Cash)
    roa = info.get('returnOnAssets', 0)
    scores['Management Quality'] = np.clip(roa * 100 * 5, 0, 100)

    # 8. Industry Strength (Simplified proxy)
    scores['Industry Strength'] = 70 # Defaulting to a neutral-positive score

    # Weighted Average
    weights = {
        'Financial Performance': 0.15,
        'Valuation': 0.15,
        'Growth Potential': 0.15,
        'Risk': 0.15,
        'Momentum': 0.10,
        'Liquidity': 0.10,
        'Management Quality': 0.10,
        'Industry Strength': 0.10
    }
    
    total_score = sum(scores[f] * weights[f] for f in scores)
    
    # Recommendation
    if total_score >= 80: rating = "Strong Buy"
    elif total_score >= 60: rating = "Buy"
    elif total_score >= 40: rating = "Hold"
    elif total_score >= 20: rating = "Sell"
    else: rating = "Strong Sell"
    
    return total_score, rating, scores
