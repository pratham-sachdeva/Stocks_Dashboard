#!/usr/bin/env python
# coding: utf-8

# In[141]:


import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import requests
import numpy as np
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# In[142]:


# Fetch the list of S&P 500 components from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_data = pd.read_html(url)[0]

# Extract the ticker symbols
sp500_tickers = sp500_data['Symbol'].tolist()

url2 = "https://en.wikipedia.org/wiki/NASDAQ-100"
nasdaq100_data = pd.read_html(url2)[4]

# Extract the ticker symbols
nasdaq100_tickers = nasdaq100_data['Ticker'].tolist()

url3 = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
djia_data = pd.read_html(url3)[1]

# Extract the ticker symbols
djia_tickers = djia_data['Symbol'].tolist()

url4 = "https://en.wikipedia.org/wiki/FTSE_100_Index"
ftse100_data = pd.read_html(url4)[4]

# Extract the ticker symbols
ftse100_tickers = ftse100_data['Ticker'].tolist()

# URL of the BSE Sensex Wikipedia page
url5 = "https://en.wikipedia.org/wiki/BSE_Sensex"
bse_sensex_data = pd.read_html(url5)[1]


# Extract the ticker symbols
bse_sensex_tickers = bse_sensex_data['Symbol'].tolist()


# In[143]:


# List of available indexes
available_indexes = ["S&P 500", "NASDAQ 100", "DOWJONES", "FTSE 100", "BSE SENSEX"]

# Sidebar: Index Selection
index_selection = st.sidebar.selectbox("Select an Index", available_indexes)


# In[144]:


index_mapping = {
    "S&P 500": "^GSPC",
    "NASDAQ 100": "^NDX",
    "DOWJONES": "^DJI",
    "FTSE 100": "^FTSE",
    "BSE SENSEX": "^BSESN"
}


# In[145]:


index_symbol = index_mapping.get(index_selection)


# In[146]:


@st.cache_data
def get_stock_list(index_name):
    
    if index_name == "S&P 500":
        
        stock_list = sp500_data
    
    elif index_name == "NASDAQ 100":
        
        stock_list = nasdaq100_tickers
        
    elif index_name == "DOWJONES":
        
        stock_list = djia_tickers
        
    elif index_name == "FTSE 100":
        
        stock_list = ftse100_tickers
    
    elif index_name == "BSE SENSEX":
        
        stock_list = bse_sensex_tickers
    
    
    else:
        stock_list = []
    
    return stock_list


# In[147]:


ticker_list = get_stock_list(index_selection)


# In[148]:


st.markdown("<h1 style='text-align: center;'>Stock Market Analysis Dashboard</h1>", unsafe_allow_html=True)


# In[149]:


@st.cache_data
def get_stock_data(stock_symbol, start_date, end_date):
    
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data


# In[150]:


@st.cache_data
def get_index_data(index_symbol, timeframe):
    
    # Define end date as today
    end_date = datetime.now()
        
    # Calculate start date based on the selected timeframe
    if timeframe == '5 Day':
        start_date = end_date - timedelta(days=5)
    elif timeframe == '1 Week':
        start_date = end_date - timedelta(weeks=1)
    elif timeframe == '1 Month':
        start_date = end_date - timedelta(weeks=4)
    elif timeframe == '6 Months':
        start_date = end_date - timedelta(weeks=26)
    elif timeframe == 'YTD':
        start_date = datetime(end_date.year, 1, 1)
    elif timeframe == '1 Year':
        start_date = end_date - timedelta(weeks=52)
    elif timeframe == '5 Year':
        start_date = end_date - timedelta(weeks=260)
    
    index_data = yf.Ticker(index_symbol).history(start=start_date, end=end_date)

    return index_data


# In[151]:


@st.cache_data
def get_currency(stock_symbol):
    stock_data_other = yf.Ticker(stock_symbol)
    info = stock_data_other.info['longName']
    currency = stock_data_other.info['currency']
    return info,currency


# In[152]:


datetime.today()


# In[ ]:


st.sidebar.title("Select Parameters")

selected_stock = st.sidebar.selectbox("Select a stock symbol", ticker_list)
default_start_date = datetime.today() - timedelta(weeks=52)
start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date")

# Ensure the start date is before the end date
if start_date >= end_date:
    st.error("Error: Start date must be before end date.")
    
# Fetch data and display price chart
stock_data = get_stock_data(selected_stock, start_date, end_date)
other_data = get_currency(selected_stock)

st.header(f"{other_data[0]} ({selected_stock}) Stock Price")

if stock_data.empty:
    st.write("Data not available for this stock symbol in the specified date range.")
else:
    fig = go.Figure(data=go.Candlestick(x=stock_data.index,
                                       open=stock_data['Open'],
                                       high=stock_data['High'],
                                       low=stock_data['Low'],
                                       close=stock_data['Close']))
    
    fig.update_layout(yaxis_title=f'Price ({other_data[1]})',
                      xaxis_title='Date')
    
    st.plotly_chart(fig)
    


# In[190]:


warning_text = "<span style='color:red;'>Bias Warning</span>"

# Display the warning text
st.markdown(warning_text, unsafe_allow_html=True)

with st.expander("About Confirmation Bias"):
    st.write("""
    **Beware of Confirmation Bias:** 🧠

    Confirmation bias is a powerful psychological tendency where we unconsciously seek out information that supports our existing beliefs. In the world of investing, this means you might be tempted to choose a timeframe that aligns with your preconceived notions about a stock.

    📈 If you're feeling bullish, you might prefer shorter timeframes, hoping to find recent price increases.
    📉 If you're bearish, you might lean towards longer timeframes, seeking evidence of long-term declines.

    But remember, true investment wisdom comes from an objective assessment of all available data. Don't fall into the trap of cherry-picking data to support your outlook. Consider various timeframes to make well-informed decisions. 📊💼

    Stay rational and keep confirmation bias at bay!
    """)


# In[154]:


st.subheader(f"{other_data[0]} Stock Summary")
if not stock_data.empty:
    # Calculate 1-Year Change
    one_year_change = ((stock_data["Adj Close"][-1] / stock_data["Adj Close"][0]) - 1) * 100
    
    average_vol_3m = stock_data["Volume"].tail(63).mean()

    prev_close = stock_data["Close"][-2]
    open_price = stock_data["Open"][-1]
    volume = stock_data["Volume"][-1]
    day_range = f"{stock_data['Low'][-1]:,.2f}-{stock_data['High'][-1]:,.2f}"
    fifty_two_week_range = f"{stock_data['Low'].min():,.2f}-{stock_data['High'].max():,.2f}"
    
    stock_summary_data = {
        "Prev. Close": [f"{prev_close:,.2f}"],
        "Open": [f"{open_price:,.2f}"],
        "1-Year Change": [f"{one_year_change:.2f}%"],
        "Volume": [f"{volume:,.0f}"],
        "Average Vol. (3m)": [f"{average_vol_3m:,.0f}"],
        "Day's Range": [day_range],
        "52 wk Range": [fifty_two_week_range]
    }

    # Convert the dictionary to a DataFrame
    df_stock_summary = pd.DataFrame.from_dict(stock_summary_data)

    # Display the summary information in a table
    st.table(df_stock_summary)
else:
    st.write("Stock data is not available. Please select a valid stock.")


# In[155]:


pricing_data, fundamental_data, news  = st.tabs(["Pricing Data", "Fundamental Data", "Top News"])


# In[156]:


with pricing_data:
    st.subheader(f'Price Movements for {selected_stock}')
    updated_data = stock_data
    updated_data["% Change"] = stock_data["Adj Close"] / stock_data["Adj Close"].shift(1) - 1
    st.write(updated_data)
    
    annual_return = updated_data["% Change"].mean()*252*100
    annual_return_color = "green" if annual_return >= 0 else "red"
    st.markdown(f"Annual Return: <span style='color:{annual_return_color}'>{round(annual_return, 2)}%</span>", unsafe_allow_html=True)
    
    stdev = np.std(updated_data["% Change"]) * np.sqrt(252)
    stdev_color = "green" if stdev >= 0 else "red"
    st.markdown(f"Standard Deviation is: <span style='color:{stdev_color}'>{round(stdev * 100, 2)}%</span>", unsafe_allow_html=True)
    
    fig = go.Figure()
    
    st.subheader(f"{other_data[0]} % Price Change")

    # Create a condition to determine the color of bars (green for positive and red for negative)
    positive_mask = updated_data['% Change'] >= 0
    negative_mask = updated_data['% Change'] < 0

    # Add bars for positive values
    fig.add_trace(go.Bar(
        x=updated_data.index[positive_mask],
        y=updated_data['% Change'][positive_mask],
        name=f"{selected_stock} % Change (Positive)",
        marker_color='rgb(34, 139, 34)'
    ))

    # Add bars for negative values (inverted)
    fig.add_trace(go.Bar(
        x=updated_data.index[negative_mask],
        y=updated_data['% Change'][negative_mask],
        name=f"{selected_stock} % Change (Negative)",
        marker_color='rgb(220, 20, 60)'
    ))

    # Customize the chart layout
    fig.update_layout(xaxis_title='Date',
                      yaxis_title='% Price Change',
                      barmode='relative',
                      legend=dict(x=0, y=1.2))

    # Display the chart in the Streamlit app
    st.plotly_chart(fig)


# In[159]:


analyzer = SentimentIntensityAnalyzer()


# In[160]:


@st.cache_data
def print_stock_news(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    news = stock.news
    top_news = []
    for item in news[:5]:
        title = item['title']
        link = item['link']
        publish_date = item['providerPublishTime']
        
        # Analyze sentiment of the news title
        sentiment = analyzer.polarity_scores(title)
        
        news_info = {
            "title": title,
            "link": link,
            "published_date": publish_date,
            "sentiment": sentiment['compound']  # Compound sentiment score
        }
        top_news.append(news_info)
    return top_news


# In[161]:


if selected_stock:
    top_5_news = print_stock_news(selected_stock)


# In[192]:


with news:
    
    warning_text = "<span style='color:red;'>Bias Warning</span>"

    # Display the warning text
    st.markdown(warning_text, unsafe_allow_html=True)

    with st.expander("About Anchoring Bias"):
        st.write("""
        **Anchoring Bias Alert:** 🧠

        When you encounter sentiment analysis, especially if it's the first analysis you see, it can set an initial reference point or 'anchor' for your expectations. Subsequent sentiment analyses or news might then be evaluated in relation to this anchor.

        For example, if you read a highly positive sentiment analysis initially, you may anchor your expectations to that positivity. Later news, even if it's still positive but not as optimistic, could be perceived as disappointing. This bias can influence your investment decisions.

        Stay mindful of your anchors, and consider each piece of information independently to make informed decisions. Don't let an early anchor overly influence your judgment.
        """)
    st.subheader(f'Top News for {selected_stock}')
    for i, news_item in enumerate(top_5_news):

        st.subheader(f'News {i+1}')
        st.write("Title:", news_item['title'])
        st.write("Link:", news_item['link'])

        sentiment_score = news_item['sentiment']
        sentiment_color = "green" if sentiment_score > 0 else "red" if sentiment_score < 0 else "white"

        st.write("Sentiment Score:", f"<font color='{sentiment_color}'>{sentiment_score}</font>", unsafe_allow_html=True)
        
        


# In[163]:


@st.cache_data
def get_fundamental_metrics(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    info = stock.info

    # Map API response fields to desired metrics
    fundamental_metrics = {
        "Market Cap": info.get("marketCap"),
        "Forward P/E": info.get("forwardPE"),
        "Trailing P/E": info.get("trailingPE"),
        "Dividend Yield": info.get("dividendYield") * 100 if info.get("dividendYield") else None,
        "Earnings Per Share (EPS)": info.get("trailingEps"),
        "Beta": info.get("beta")
    }

    return fundamental_metrics


# In[164]:


with fundamental_data:
    st.subheader(f"Fundamental Data for {selected_stock}")

    fundamental_metrics = get_fundamental_metrics(selected_stock)

    for metric, value in fundamental_metrics.items():
        st.write(f"{metric}: {value}")


# In[165]:


st.header("Stock Price Comparison")

# Get user input for two stock symbols from the list of tickers

selected_stock2 = st.selectbox("Select the second stock symbol", ticker_list)

# Fetch data for second stock
stock_data2 = get_stock_data(selected_stock2, start_date, end_date)



if stock_data.empty or stock_data2.empty:
    st.write("Data not available for one or more selected stock symbols in the specified date range.")

else:
    # Create an area chart for the first stock
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index,
                             y=stock_data['Close'],
                             mode='lines',
                             fill='tozeroy',  # Create an area chart
                             name=f"{selected_stock} Closing Price"))

    # Create an area chart for the closing prices of the second stock
    fig.add_trace(go.Scatter(x=stock_data2.index,
                            y=stock_data2['Close'],
                            mode='lines',
                            fill='tozeroy',  # Create an area chart
                            name=f"{selected_stock2} Closing Price"))

    fig.update_layout(title='Stock Price Comparison',
                        xaxis_title='Date',
                        yaxis_title=f'Price ({other_data[1]})',
                        legend=dict(x=0, y=1.2))

    st.plotly_chart(fig)    


# In[166]:


st.header(f"{index_selection} Index Performance")

index_timeframe = ['5 Day', '1 Week', '1 Month', '6 Months', 'YTD', '1 Year', '5 Year']
selected_index_timeframe = st.selectbox("Select a timeframe for Index Price", index_timeframe)

# Fetch historical price data for the selected index
index_his_data = get_index_data(index_symbol, selected_index_timeframe)


if index_his_data.empty:
    st.write("Data not available for the selected index.")
else:
    # Create a line chart for the index price
    fig = go.Figure(data=go.Candlestick(x=index_his_data.index,
                                       open=index_his_data['Open'],
                                       high=index_his_data['High'],
                                       low=index_his_data['Low'],
                                       close=index_his_data['Close']))

    # Customize the chart layout
    fig.update_layout(xaxis_title='Date',
                      yaxis_title=f'Price ({other_data[1]})',
                      legend=dict(x=0, y=1.2))

    # Display the chart in the Streamlit app
    st.plotly_chart(fig)


# In[167]:


def fetch_market_cap_data(index_tickers):
    market_cap_data = {}
    sector_data = {}
    
    for ticker in index_tickers:
        try:
            stock_info = yf.Ticker(ticker).info
            if "marketCap" in stock_info and stock_info["marketCap"]:
                if "longName" in stock_info and stock_info["longName"]:
                    market_cap_data[ticker] = {
                        "LongName": stock_info["longName"],
                        "MarketCap": float(stock_info["marketCap"])
                    }
                if "sector" in stock_info and stock_info["sector"]:
                    sector_data[ticker] = {
                        "Sector": stock_info["sector"]
                    }
        except requests.exceptions.HTTPError as e:
            pass
        
    market_cap_df = pd.DataFrame(market_cap_data.values(), index=market_cap_data.keys())
    sector_df = pd.DataFrame(sector_data.values(), index=sector_data.keys())
    
    # Merge the two DataFrames on the index (ticker symbol)
    merged_df = market_cap_df.merge(sector_df, left_index=True, right_index=True)
    
    return merged_df


# In[168]:


new_tickerlist_nasdaq = ['AAPL','MSFT', 'GOOG', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ASML']


# In[169]:


new_tickerlist_sp500 = ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'LLY', 'V']


# In[170]:


new_tickerlist_dowjones = ['AAPL', 'MSFT', 'V', 'UNH', 'JNJ', 'JPM', 'WMT', 'PG', 'HD', 'CVX']


# In[171]:


new_tickerlist_ftse100 = ['AZN', 'SHEL', 'BA', 'BP', 'RIO', 'GSK', 'JD', 'ADM', 'CRH', 'HLN']


# In[172]:


new_tickerlist_bse = ['RELIANCE.BO', 'TCS.BO', 'HDFCBANK.BO', 'ICICIBANK.BO', 'HINDUNILVR.BO', 'INFY.BO', 'ITC.BO', 'SBIN.BO',
                      'BHARTIARTL.BO', 'BAJFINANCE.BO']


# In[173]:



if index_selection == "S&P 500":
    ticker_list_2 = new_tickerlist_sp500
    market_cap_df = fetch_market_cap_data(ticker_list_2)
    top_10_stocks = market_cap_df.nlargest(10, "MarketCap")

elif index_selection == "NASDAQ 100":
    ticker_list_2 = new_tickerlist_nasdaq
    market_cap_df = fetch_market_cap_data(ticker_list_2)
    top_10_stocks = market_cap_df.nlargest(10, "MarketCap")

elif index_selection == "DOWJONES":
    ticker_list_2 = new_tickerlist_dowjones
    market_cap_df = fetch_market_cap_data(ticker_list_2)
    top_10_stocks = market_cap_df.nlargest(10, "MarketCap")

elif index_selection == "FTSE 100":
    ticker_list_2 = new_tickerlist_ftse100
    market_cap_df = fetch_market_cap_data(ticker_list_2)
    top_10_stocks = market_cap_df.nlargest(10, "MarketCap")

elif index_selection == "BSE SENSEX":
    ticker_list_2 = new_tickerlist_bse
    market_cap_df = fetch_market_cap_data(ticker_list_2)
    top_10_stocks = market_cap_df.nlargest(10, "MarketCap")
    
else:
    ticker_list_2 = []


# In[ ]:


st.header(f'Top Stocks by Market Cap - {index_selection}')
# Plot a treemap using Plotly Express
fig = px.treemap(top_10_stocks, path=['LongName'], values='MarketCap',
                 color='MarketCap')
st.plotly_chart(fig)


# In[194]:


warning_text = "<span style='color:red;'>Bias Warning</span>"

# Display the warning text
st.markdown(warning_text, unsafe_allow_html=True)

with st.expander("About Herd Mentality"):
    st.write("""
    **Herd Mentality Alert:** 🐑

    Herd mentality is a powerful psychological force that can influence investment decisions. When you see the largest market cap stocks in the Treemap, there might be a temptation to follow the crowd and invest in these stocks. They are often seen as safer and more reliable due to their size.

    However, it's essential to remember that what works for the herd may not always be the best choice for your unique financial goals and risk tolerance. Investment decisions should be based on a careful assessment of your own objectives and research, rather than blindly following the crowd.

    Stay independent in your choices and consider a diversified approach. This can help you avoid the pitfalls of herd mentality and make decisions that align with your personal financial strategy.
""")


# In[175]:


index_data = fetch_market_cap_data(ticker_list_2)


# In[176]:



sector_performance = index_data.groupby('Sector')['MarketCap'].sum().reset_index()

st.subheader(f'Sector-wise Performance for Top stocks - {index_selection}')
fig = px.bar(sector_performance, x='Sector', y='MarketCap',
             labels={'Sector': ' ', 'MarketCap': 'Total Market Cap'},
             color='Sector')

fig.update_xaxes(categoryorder='total descending')

st.plotly_chart(fig)


# In[177]:


def plot_sma_vs_closing_price(stock_symbol, start, end):
    # Retrieve stock data using yfinance
    stock_data = yf.download(stock_symbol, start, end)
    
    # Calculate Simple Moving Average (SMA)
    sma_period = 20
    stock_data['SMA'] = stock_data['Close'].rolling(window=sma_period).mean()
    
   # Create a Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA'], mode='lines', name=f'SMA {sma_period}'))
    
    # Customize the layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title= f'Price ({other_data[1]})',
        title=f'{stock_symbol} Closing Price vs. SMA',
        legend=dict(x=0, y=1)
    )
    
    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)


# In[178]:


def plot_ema_vs_closing_price(stock_symbol, start, end):
    # Retrieve stock data using yfinance
    stock_data = yf.download(stock_symbol, start=start, end=end)
    
    ema_period = 20
    # Calculate Exponential Moving Average (EMA)
    stock_data['EMA'] = stock_data['Close'].ewm(span=ema_period).mean()
    
    # Create a Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['EMA'], mode='lines', name=f'EMA {ema_period}'))
    
    # Customize the layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title= f'Price ({other_data[1]})',
        title=f'{stock_symbol} Closing Price vs. EMA',
        legend=dict(x=0, y=1)
    )
    
    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)


# In[179]:


st.header("Trend Analysis using Indicators")
indicator_type = st.selectbox("Select Indicator Type", ["sma", "ema"])
                                               


# In[180]:


if indicator_type == "sma":
    sma_plot = plot_sma_vs_closing_price(selected_stock, start_date, end_date)
    
elif indicator_type == 'ema':
    ema_plot = plot_ema_vs_closing_price(selected_stock, start_date, end_date)       


# ## Simulation

# In[181]:


def monte_carlo_simulation(returns, initial_investment, num_years, num_simulations):
    portfolio_values = np.zeros((num_simulations, num_years * 252))
    for i in range(num_simulations):
        # Convert returns to DataFrame for random sampling
        returns_df = returns.to_frame()
        sampled_returns = returns_df.sample(num_years * 252, replace=True, axis=0)
        price_paths = initial_investment * np.cumprod(1 + sampled_returns)
        portfolio_values[i, :] = np.sum(price_paths, axis=1)
    return portfolio_values


# In[182]:


st.header("Stock Portfolio Monte Carlo Simulation")
st.write('''Monte Carlo Simulation gives a wide range of potential future outcomes by running a large number of random simulations.
            Provides valuable insights into risk and uncertainty.''')

initial_investment = st.number_input("Initial Portfolio Value:", min_value=1000, value=10000)
num_years = st.number_input("Number of Simulation Years:", min_value=1, value=5)
num_simulations = st.number_input("Number of Simulations:", min_value=1, value=10)


# In[183]:


stock_returns = stock_data['Adj Close'].pct_change().dropna()


# In[184]:


simulation_results = monte_carlo_simulation(stock_returns, initial_investment, num_years, num_simulations)


# In[185]:


fig_simulation = go.Figure()

for i in range(num_simulations):
    # Convert the range object to a list using the list() function
    x_values = list(range(num_years * 252))
    fig_simulation.add_trace(go.Scatter(x=x_values, y=simulation_results[i, :], mode='lines', name=f'Simulation {i+1}'))

fig_simulation.update_layout(
    title='Portfolio Value Simulation',
    xaxis_title='Trading Days',
    yaxis_title='Portfolio Value',
)

st.plotly_chart(fig_simulation)


# In[1]:


warning_text = "<span style='color:red;'>Bias Warning</span>"

# Display the warning text
st.markdown(warning_text, unsafe_allow_html=True)

with st.expander("About Cognitive Bias"):
    st.write("""
    **Overreaction to Extreme Data Alert:** 📈📉

    When you encounter extreme scenarios in a Monte Carlo simulation or financial analysis, it can evoke intense emotional responses. These outliers may trigger fear or greed, leading to impulsive decisions that deviate from your usual risk tolerance.

    For instance, in a Monte Carlo simulation, if you come across a rare scenario where investments could potentially lose a significant portion of their value, it might induce panic. In such moments, there's a risk of making hasty decisions, like selling investments hastily.

    Remember that extreme scenarios are often outliers and not representative of the norm. Maintain a rational and long-term perspective, and avoid letting momentary emotions guide your actions. Strategic and well-informed decisions lead to better financial outcomes." 🧠💼
""")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




