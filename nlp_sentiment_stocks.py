"""
=============================================================
 Real-Time NLP Sentiment & Stock Market Correlation Engine
 Author : Thejas Sharma
 Data   : Yahoo Finance API (yfinance) + Synthetic news headlines
          For real news: NewsAPI.org (free tier) or
          Kaggle: "Financial News and Stock Price Integration"
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# NLP
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("⚠️  vaderSentiment not installed — using rule-based sentiment fallback")
    print("   Install: pip install vaderSentiment")

# Stock data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance not installed — using synthetic stock data")
    print("   Install: pip install yfinance")

from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

plt.style.use('seaborn-v0_8-darkgrid')
NAVY, GOLD, RED, GREEN = '#1F3864', '#C9A84C', '#e74c3c', '#2ecc71'

COMPANIES = {
    'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Alphabet',
    'AMZN': 'Amazon', 'META': 'Meta'
}

# ─────────────────────────────────────────────
# 1. GENERATE / LOAD HEADLINES
# ─────────────────────────────────────────────
POSITIVE_TEMPLATES = [
    "{co} beats Q{q} earnings expectations, revenue up {n}%",
    "{co} announces record-breaking sales of ${n}B this quarter",
    "{co} launches revolutionary AI product, stock surges",
    "{co} expands to new markets, analysts upgrade to Buy",
    "{co} reports strong user growth of {n}M new customers",
    "{co} CEO announces major strategic partnership deal",
    "{co} dividend increase signals strong financial health",
]
NEGATIVE_TEMPLATES = [
    "{co} misses earnings forecast, shares fall sharply",
    "{co} faces regulatory investigation over data practices",
    "{co} announces {n}% workforce reduction amid cost cuts",
    "{co} Q{q} revenue disappoints, guidance lowered",
    "{co} loses major contract worth ${n}B to competitor",
    "{co} executive departure raises investor concerns",
    "{co} product recall impacts quarterly profit margins",
]
NEUTRAL_TEMPLATES = [
    "{co} holds annual shareholder meeting, no major changes",
    "{co} appoints new board member from finance sector",
    "{co} files quarterly 10-Q report with SEC",
    "{co} announces minor product update for enterprise clients",
    "{co} participates in industry conference next month",
]

def generate_headlines(n_days=180):
    np.random.seed(42)
    records = []
    base_date = datetime.today() - timedelta(days=n_days)

    for day_offset in range(n_days):
        date = base_date + timedelta(days=day_offset)
        if date.weekday() >= 5:
            continue  # skip weekends
        for ticker, name in COMPANIES.items():
            n_articles = np.random.randint(2, 6)
            for _ in range(n_articles):
                sentiment_type = np.random.choice(['positive','negative','neutral'],
                                                   p=[0.40, 0.30, 0.30])
                tmpl = np.random.choice(
                    POSITIVE_TEMPLATES if sentiment_type == 'positive'
                    else NEGATIVE_TEMPLATES if sentiment_type == 'negative'
                    else NEUTRAL_TEMPLATES
                )
                headline = tmpl.format(
                    co=name, q=np.random.randint(1,5),
                    n=np.random.randint(5, 40)
                )
                records.append({'date': date.date(), 'ticker': ticker,
                                'company': name, 'headline': headline,
                                'true_sentiment': sentiment_type})

    df = pd.DataFrame(records)
    print(f"✅ Generated {len(df):,} headlines across {df['date'].nunique()} trading days")
    return df


# ─────────────────────────────────────────────
# 2. SENTIMENT ANALYSIS
# ─────────────────────────────────────────────
def rule_based_sentiment(text):
    """Simple fallback if VADER not available."""
    pos_words = ['beat','record','strong','growth','surge','expand','launch','partner','increase','profit']
    neg_words = ['miss','fall','cut','reduce','lose','fail','decline','concern','recall','investigation']
    text_lower = text.lower()
    pos = sum(w in text_lower for w in pos_words)
    neg = sum(w in text_lower for w in neg_words)
    if pos > neg: return 0.5 + pos * 0.1
    elif neg > pos: return -0.5 - neg * 0.1
    return 0.0


def analyse_sentiment(df):
    if VADER_AVAILABLE:
        analyzer = SentimentIntensityAnalyzer()
        df['sentiment_score'] = df['headline'].apply(
            lambda h: analyzer.polarity_scores(h)['compound'])
        print("✅ VADER sentiment analysis complete")
    else:
        df['sentiment_score'] = df['headline'].apply(rule_based_sentiment)
        print("✅ Rule-based sentiment analysis complete (fallback)")

    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda s: 'Positive' if s > 0.05 else ('Negative' if s < -0.05 else 'Neutral'))

    # Daily aggregate sentiment per ticker
    daily_sent = df.groupby(['date','ticker']).agg(
        avg_sentiment  = ('sentiment_score', 'mean'),
        n_articles     = ('headline',        'count'),
        pos_ratio      = ('sentiment_label', lambda x: (x=='Positive').mean())
    ).reset_index()
    daily_sent['date'] = pd.to_datetime(daily_sent['date'])
    print(f"✅ Daily sentiment aggregated: {len(daily_sent):,} rows")
    return daily_sent


# ─────────────────────────────────────────────
# 3. STOCK PRICE DATA
# ─────────────────────────────────────────────
def get_stock_data(tickers, start_date, end_date):
    if YFINANCE_AVAILABLE:
        try:
            print("📡 Fetching real stock data from Yahoo Finance...")
            raw = yf.download(list(tickers.keys()), start=start_date,
                              end=end_date, progress=False)['Close']
            records = []
            for ticker in tickers:
                if ticker in raw.columns:
                    sub = raw[ticker].dropna().reset_index()
                    sub.columns = ['date', 'close']
                    sub['ticker'] = ticker
                    sub['return_1d'] = sub['close'].pct_change() * 100
                    records.append(sub)
            df_stock = pd.concat(records)
            print(f"✅ Real stock data loaded: {df_stock.shape}")
            return df_stock
        except Exception as e:
            print(f"⚠️  Yahoo Finance error ({e}) — using synthetic stock data")

    print("📊 Generating synthetic stock price data...")
    np.random.seed(99)
    records = []
    dates = pd.bdate_range(start_date, end_date)
    base_prices = {'AAPL':175, 'MSFT':380, 'GOOGL':155, 'AMZN':185, 'META':500}
    for ticker in tickers:
        price = base_prices.get(ticker, 200)
        prices = [price]
        for _ in range(len(dates)-1):
            shock = np.random.normal(0.0003, 0.015)
            prices.append(prices[-1] * (1 + shock))
        sub = pd.DataFrame({'date': dates, 'close': prices, 'ticker': ticker})
        sub['return_1d'] = sub['close'].pct_change() * 100
        records.append(sub)
    df_stock = pd.concat(records)
    print(f"✅ Synthetic stock data: {df_stock.shape}")
    return df_stock


# ─────────────────────────────────────────────
# 4. CORRELATION ANALYSIS
# ─────────────────────────────────────────────
def correlation_analysis(daily_sent, df_stock):
    merged = daily_sent.merge(df_stock[['date','ticker','return_1d']], on=['date','ticker'])
    merged = merged.dropna()

    # Lead-lag: sentiment today vs return tomorrow
    merged = merged.sort_values(['ticker','date'])
    merged['return_next_day'] = merged.groupby('ticker')['return_1d'].shift(-1)
    merged = merged.dropna(subset=['return_next_day'])

    correlations = []
    for ticker in COMPANIES:
        sub = merged[merged['ticker'] == ticker]
        if len(sub) < 20: continue
        r_same, p_same = stats.pearsonr(sub['avg_sentiment'], sub['return_1d'])
        r_lead, p_lead = stats.pearsonr(sub['avg_sentiment'], sub['return_next_day'])
        correlations.append({
            'ticker':    ticker,
            'company':   COMPANIES[ticker],
            'r_same_day':   round(r_same, 3),
            'r_next_day':   round(r_lead, 3),
            'p_value_lead': round(p_lead,  4),
            'significant':  p_lead < 0.05,
            'n_observations': len(sub)
        })
        print(f"  {ticker:5s} — Same-day r={r_same:.3f} | Lead-1d r={r_lead:.3f} | p={p_lead:.4f} {'✅' if p_lead<0.05 else ''}")

    corr_df = pd.DataFrame(correlations)
    print(f"\n✅ Significant correlations: {corr_df['significant'].sum()} / {len(corr_df)}")
    return merged, corr_df


# ─────────────────────────────────────────────
# 5. DASHBOARD
# ─────────────────────────────────────────────
def create_dashboard(daily_sent, df_stock, merged, corr_df):
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor('#F8F9FA')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)
    fig.text(0.5, 0.97, 'NLP SENTIMENT & STOCK MARKET CORRELATION ENGINE',
             ha='center', fontsize=15, fontweight='bold', color=NAVY)
    fig.text(0.5, 0.945, 'Thejas Sharma  |  Data & Business Analyst',
             ha='center', fontsize=10, color='gray')

    ticker_ex = list(COMPANIES.keys())[0]
    colors_list = [NAVY, GOLD, RED, GREEN, '#9b59b6']

    # ── 1. Sentiment over time ──
    ax1 = fig.add_subplot(gs[0, :2])
    for i, (ticker, color) in enumerate(zip(COMPANIES.keys(), colors_list)):
        sub = daily_sent[daily_sent['ticker']==ticker].sort_values('date')
        rolled = sub.set_index('date')['avg_sentiment'].rolling('7D').mean()
        ax1.plot(rolled.index, rolled.values, label=ticker, color=color, lw=1.8)
    ax1.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax1.set_title('7-Day Rolling Sentiment Score by Company', fontweight='bold')
    ax1.set_ylabel('Avg Sentiment Score')
    ax1.legend(fontsize=9, ncol=5)
    ax1.fill_between(rolled.index, 0, rolled.values, where=rolled.values>0,
                     alpha=0.07, color=GREEN)
    ax1.fill_between(rolled.index, 0, rolled.values, where=rolled.values<0,
                     alpha=0.07, color=RED)

    # ── 2. Correlation heatmap ──
    ax2 = fig.add_subplot(gs[0, 2])
    corr_pivot = corr_df.set_index('company')[['r_same_day','r_next_day']]
    corr_pivot.columns = ['Same Day', 'Next Day (Lead)']
    sns.heatmap(corr_pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                ax=ax2, linewidths=0.5, vmin=-0.5, vmax=0.5)
    ax2.set_title('Sentiment→Return Correlation', fontweight='bold')

    # ── 3. Scatter sentiment vs return ──
    ax3 = fig.add_subplot(gs[1, 0])
    sub_ex = merged[merged['ticker']==ticker_ex]
    ax3.scatter(sub_ex['avg_sentiment'], sub_ex['return_next_day'],
                alpha=0.4, color=NAVY, s=20)
    m, b = np.polyfit(sub_ex['avg_sentiment'], sub_ex['return_next_day'], 1)
    x_line = np.linspace(sub_ex['avg_sentiment'].min(), sub_ex['avg_sentiment'].max(), 100)
    ax3.plot(x_line, m*x_line+b, color=GOLD, lw=2)
    ax3.set_title(f'{ticker_ex}: Sentiment vs Next-Day Return', fontweight='bold')
    ax3.set_xlabel('Sentiment Score')
    ax3.set_ylabel('Next-Day Return (%)')

    # ── 4. Stock price with sentiment overlay ──
    ax4 = fig.add_subplot(gs[1, 1:])
    stock_ex = df_stock[df_stock['ticker']==ticker_ex].sort_values('date')
    sent_ex  = daily_sent[daily_sent['ticker']==ticker_ex].sort_values('date')
    ax4b = ax4.twinx()
    ax4.plot(stock_ex['date'], stock_ex['close'], color=NAVY, lw=2, label='Stock Price')
    ax4b.fill_between(sent_ex['date'], 0, sent_ex['avg_sentiment'],
                      where=sent_ex['avg_sentiment']>0, alpha=0.25, color=GREEN, label='Pos Sentiment')
    ax4b.fill_between(sent_ex['date'], 0, sent_ex['avg_sentiment'],
                      where=sent_ex['avg_sentiment']<0, alpha=0.25, color=RED, label='Neg Sentiment')
    ax4.set_title(f'{ticker_ex} Stock Price vs News Sentiment', fontweight='bold')
    ax4.set_ylabel('Stock Price ($)', color=NAVY)
    ax4b.set_ylabel('Sentiment Score', color=GREEN)
    ax4.tick_params(axis='x', rotation=30)

    # ── 5. Sentiment label distribution ──
    ax5 = fig.add_subplot(gs[2, 0])
    sent_counts = daily_sent.merge(
        pd.DataFrame([{'ticker': t, 'company': c} for t, c in COMPANIES.items()]),
        on='ticker'
    )
    # overall
    overall = daily_sent.copy()
    overall['sentiment_label'] = overall['avg_sentiment'].apply(
        lambda s: 'Positive' if s>0.05 else ('Negative' if s<-0.05 else 'Neutral'))
    vc = overall['sentiment_label'].value_counts()
    ax5.bar(vc.index, vc.values, color=[GREEN, RED, GOLD][:len(vc)])
    ax5.set_title('Overall Sentiment Distribution', fontweight='bold')
    ax5.set_ylabel('Days Count')

    # ── 6. Lead-lag bar chart ──
    ax6 = fig.add_subplot(gs[2, 1:])
    x = np.arange(len(corr_df))
    w = 0.35
    bars1 = ax6.bar(x-w/2, corr_df['r_same_day'],   width=w, label='Same Day',        color=NAVY,  alpha=0.8)
    bars2 = ax6.bar(x+w/2, corr_df['r_next_day'],   width=w, label='Lead 1 Day',       color=GOLD,  alpha=0.8)
    ax6.axhline(0, color='black', lw=0.8, alpha=0.5)
    ax6.set_xticks(x)
    ax6.set_xticklabels(corr_df['ticker'], fontsize=10)
    ax6.set_title('Sentiment–Return Correlation by Company', fontweight='bold')
    ax6.set_ylabel('Pearson r')
    ax6.legend()

    plt.savefig('sentiment_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#F8F9FA')
    plt.close()
    print("\n📊 Dashboard saved → sentiment_dashboard.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 NLP SENTIMENT & STOCK CORRELATION ENGINE — THEJAS SHARMA")
    print("="*60)

    end_date   = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=180)).strftime('%Y-%m-%d')

    headlines  = generate_headlines(n_days=180)
    daily_sent = analyse_sentiment(headlines)

    print("\n📈 Fetching stock data...")
    df_stock = get_stock_data(COMPANIES, start_date, end_date)

    print("\n🔗 Running correlation analysis...")
    merged, corr_df = correlation_analysis(daily_sent, df_stock)

    print("\n🎨 Building dashboard...")
    create_dashboard(daily_sent, df_stock, merged, corr_df)

    print("\n✅ Done! Results saved. Upload to GitHub.")
