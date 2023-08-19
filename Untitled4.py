#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


import pandas as pd

# Fetch the list of S&P 500 components from Wikipedia
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_data = pd.read_html(url)[0]

# Extract the ticker symbols
sp500_tickers = sp500_data['Symbol'].tolist()


# In[4]:


url = "https://en.wikipedia.org/wiki/NASDAQ-100"
nasdaq100_data = pd.read_html(url)[4]

# Extract the ticker symbols
nasdaq100_tickers = nasdaq100_data['Ticker'].tolist()


# In[5]:


url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
djia_data = pd.read_html(url)[1]

# Extract the ticker symbols
djia_tickers = djia_data['Symbol'].tolist()


# In[6]:


url = "https://en.wikipedia.org/wiki/FTSE_100_Index"
ftse100_data = pd.read_html(url)[4]

# Extract the ticker symbols
ftse100_tickers = ftse100_data['Ticker'].tolist()


# In[7]:


# List of available indexes
available_indexes = ["S&P 500", "NASDAQ 100", "DOWJONES", "FTSE 100"]

# Sidebar: Index Selection
index_selection = st.sidebar.selectbox("Select an Index", available_indexes)


# In[8]:


def get_stock_list(index_name):
    
    if index_name == "S&P 500":
        
        stock_list = sp500_data
    
    elif index_name == "NASDAQ 100":
        
        stock_list = nasdaq100_tickers
        
    elif index_name == "DOWJONES":
        
        stock_list = djia_tickers
        
    elif index_name == "FTSE 100":
        
        stock_list = ftse100_tickers
            
    else:
        stock_list = []
    
    return stock_list


# In[9]:


ticker_list = get_stock_list(index_selection)


# In[40]:


st.markdown("<h1 style='text-align: center;'>Financial Dashboard</h1>", unsafe_allow_html=True)


# In[11]:


# Function to get historical stock data
def get_stock_data(stock_symbol, start_date, end_date):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    return stock_data


# In[12]:


def get_currency(stock_symbol):
    stock_data_other = yf.Ticker(stock_symbol)
    info = stock_data_other.info['longName']
    currency = stock_data_other.info['currency']
    
    return info,currency 
    


# In[13]:


a = get_currency("AAPL")


# In[14]:


print(a[0])


# In[15]:


# Get user input for stock symbol from a list of tickers

st.sidebar.title("Select Stock Symbols")
selected_stock = st.sidebar.selectbox("Select a stock symbol", ticker_list)
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date")

# Ensure the start date is before the end date
if start_date >= end_date:
    st.error("Error: Start date must be before end date.")
    
# Fetch data and display price chart
stock_data = get_stock_data(selected_stock, start_date, end_date)
other_data = get_currency(selected_stock)

st.subheader(f"{other_data[0]} ({selected_stock}) Stock Price")

if stock_data.empty:
    st.write("Data not available for this stock symbol in the specified date range.")
else:
    fig = go.Figure(data=go.Candlestick(x=stock_data.index,
                                       open=stock_data['Open'],
                                       high=stock_data['High'],
                                       low=stock_data['Low'],
                                       close=stock_data['Close']))
    
    fig.update_layout(yaxis_title=f'Price ({other_data[1]})')
    
    st.plotly_chart(fig)


# In[44]:





# In[19]:


st.title("Stock Price Comparison")

# Get user input for two stock symbols from the list of tickers

selected_stock2 = st.selectbox("Select the second stock symbol", ticker_list)

# Fetch data for second stock
stock_data2 = get_stock_data(selected_stock2, start_date, end_date)


if stock_data.empty or stock_data2.empty:
    st.write("Data not available for one or more selected stock symbols in the specified date range.")
else:
    # Create a candlestick chart for the first stock
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index,
                             y=stock_data['Close'],
                             mode='lines',
                             name=f"{selected_stock} Closing Price"))

    # Create a line chart for the closing prices of the second stock
    fig.add_trace(go.Scatter(x=stock_data2.index,
                            y=stock_data2['Close'],
                            mode='lines',
                            name=f"{selected_stock2} Closing Price"))

    fig.update_layout(title='Stock Price Comparison',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        yaxis2=dict(title='Closing Price', overlaying='y', side='right'),
                        legend=dict(x=0, y=1.2))

    st.plotly_chart(fig)




    


# In[45]:


stock_summary, pricing_data, fundamental_data, news  = st.tabs(["Stock Summary","Pricing Data", "Fundamental Data", "Top News"])


# In[47]:


with stock_summary:
    if not stock_data.empty:
        # Calculate 1-Year Change
        one_year_change = ((stock_data["Adj Close"][-1] / stock_data["Adj Close"][0]) - 1) * 100
    
        # Calculate average volume for the last 3 months
        average_vol_3m = stock_data["Volume"].tail(63).mean()

        prev_close = stock_data["Close"][-2]
        open_price = stock_data["Open"][-1]
        volume = stock_data["Volume"][-1]
        day_range = f"{stock_data['Low'][-1]:,.2f}-{stock_data['High'][-1]:,.2f}"
        fifty_two_week_range = f"{stock_data['Low'].min():,.2f}-{stock_data['High'].max():,.2f}"
    
        st.subheader(f"Stock Summary: {other_data[0]}")
        st.markdown(f"**Prev. Close:** ${prev_close:,.2f}")
        st.markdown(f"**Open:** ${open_price:,.2f}")
        st.markdown(f"**1-Year Change:** {one_year_change:.2f}%")
        st.markdown(f"**Volume:** {volume:,.0f}")
        st.markdown(f"**Average Vol. (3m):** {average_vol_3m:,.0f}")
        st.markdown(f"**Day's Range:** {day_range}")
        st.markdown(f"**52 wk Range:** {fifty_two_week_range}")


# In[21]:


with pricing_data:
    st.header('Price Movements')
    updated_data = stock_data
    updated_data["% Change"] = stock_data["Adj Close"] / stock_data["Adj Close"].shift(1) - 1
    st.write(updated_data)
    
    annual_return = updated_data["% Change"].mean()*252*100
    annual_return_color = "green" if annual_return >= 0 else "red"
    st.markdown(f"Annual Return: <span style='color:{annual_return_color}'>{round(annual_return, 2)}%</span>", unsafe_allow_html=True)
    
    stdev = np.std(updated_data["% Change"]) * np.sqrt(252)
    stdev_color = "green" if stdev >= 0 else "red"
    st.markdown(f"Standard Deviation is: <span style='color:{stdev_color}'>{round(stdev * 100, 2)}%</span>", unsafe_allow_html=True)


# In[22]:


def print_stock_news(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    news = stock.news
    top_news = []
    for item in news[:5]:
        title = item['title']
        link = item['link']
        publish_date = item['providerPublishTime']
        news_info = {
            "title": title,
            "link": link,
            "published_date": publish_date
        }
        top_news.append(news_info)
    return top_news


# In[23]:


if selected_stock:
    top_5_news = print_stock_news(selected_stock)


# In[24]:


with news:
    st.subheader(f'Top News for ({selected_stock})')
    for i, news_item in enumerate(top_5_news):
        st.subheader(f'News {i+1}')
        st.write("Title:", news_item['title'])
        st.write("Link:", news_item['link'])
        
                     


# In[25]:


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


# In[26]:



with fundamental_data:
    st.header("Fundamental Data")
    st.subheader(f"Fundamental Data for ({selected_stock})")

    fundamental_metrics = get_fundamental_metrics(selected_stock)

    for metric, value in fundamental_metrics.items():
        st.write(f"{metric}: {value}")


# In[ ]:





# In[27]:


from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries


# def get_alpha_vantage_indicator_data(stock_symbol, indicator_type, api_key):
#     
#     ts = TimeSeries(key=api_key, output_format='pandas')
#     data_ts, meta_data_ts = ts.get_daily(stock_symbol, outputsize='full')
#     
#     period = 60
#     
#     ti = TechIndicators(key=api_key, output_format='pandas')
#     
#     if indicator_type == "sma":
#         data, meta_data = ti.get_sma(symbol=stock_symbol,interval='daily', time_period = period, series_type='close')
#     elif indicator_type == "ema":
#         data, meta_data = ti.get_ema(symbol=stock_symbol,interval='daily', time_period = period, series_type='close')
#     elif indicator_type == "rsi":
#         data, meta_data = ti.get_rsi(symbol=stock_symbol,interval='daily', time_period = period, series_type='close')
#         
#     df1 = data
#     df2 = data_ts['4. close'].iloc[period-1::]
#     df2.index = df1.index
#     total_df = pd.concat([df1,df2], axis=1)
#     
#     return total_df

# with st.sidebar:
#     indicator_type = st.selectbox("Select Indicator Type", ["sma", "ema", "rsi"])
#     

# indicator_column_mapping = {
#         "sma": "SMA",  
#         "ema": "EMA",
#         "rsi": "RSI"
#         
#     }

# indicator_column_name = indicator_column_mapping.get(indicator_type)

# with st.header("Technical Indicator Analysis"):
#         
#     indicator_data = get_alpha_vantage_indicator_data(selected_stock, indicator_type, api_key= 'FUR59JMM1VD9MNKI')
#     
#     if not indicator_data.empty:
#             print(indicator_data)
#             
#             fig = go.Figure()
#             fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'],
#                                    mode='lines', name=f"Closing Price - {selected_stock}"))
#             fig.add_trace(go.Scatter(x=indicator_data.index, y=indicator_data[indicator_column_name],
#                                  mode='lines', name=f"{indicator_type.upper()} - {selected_stock}"))
#             fig.update_layout(title=f"{indicator_type.upper()} vs Closing Price for ({selected_stock})",
#                           xaxis_title='Date', yaxis_title='Value')
#             st.plotly_chart(fig)
#             
#     else:
#         st.write("Indicator data not available for this stock symbol.")

# stock_performance = {}
# 
# for ticker in ticker_list:
#     stock_p_data = yf.download(ticker, start=start_date, end=end_date)
#     if not stock_data.empty:
#         percent_change = (stock_p_data['Adj Close'].iloc[-1] / stock_p_data['Adj Close'].iloc[0] - 1) * 100
#         stock_performance[ticker] = percent_change
#         
# top_performing_stocks = sorted(stock_performance.items(), key=lambda x: x[1], reverse=True)[:10]
# top_stocks = [stock[0] for stock in top_performing_stocks]
# 
# 
# # Fetch historical stock data for the top performing stocks
# top_stocks_data = {}
# for ticker in top_stocks:
#     stock_p_data = yf.download(ticker, start=start_date, end=end_date)
#     top_stocks_data[ticker] = stock_p_data

# # Create a line chart to visualize the performance
# fig = go.Figure()
# for ticker, stock_p_data in top_stocks_data.items():
#     fig.add_trace(go.Scatter(x=stock_p_data.index, y=stock_p_data['Adj Close'], mode='lines', name=ticker))
# 
# fig.update_layout(title=f"Top 10 Performing Stocks in {index_selection}",
#                   xaxis_title='Date', yaxis_title='Price',
#                   legend=dict(x=0, y=1.2))
# st.plotly_chart(fig)

# In[28]:


def fetch_market_cap_data(index_tickers):
    market_cap_data = {}
    for ticker in index_tickers:
        try:
            stock_info = yf.Ticker(ticker).info
            if "marketCap" in stock_info and stock_info["marketCap"]:
                if "longName" in stock_info and stock_info["longName"]:
                    market_cap_data[ticker] = {
                        "LongName": stock_info["longName"],
                        "MarketCap": float(stock_info["marketCap"])
                    }
        except requests.exceptions.HTTPError as e:
            pass
        
    df = pd.DataFrame(market_cap_data.values(), index=market_cap_data.keys())
    return df


# In[29]:


new_tickerlist_nasdaq = ['AAPL','MSFT', 'GOOG', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ASML']


# In[30]:


new_tickerlist_sp500 = ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'LLY', 'V']


# In[31]:


new_tickerlist_dowjones = ['AAPL', 'MSFT', 'V', 'UNH', 'JNJ', 'JPM', 'WMT', 'PG', 'HD', 'CVX']


# In[32]:


new_tickerlist_ftse100 = ['AZN', 'SHEL', 'BA', 'BP', 'RIO', 'GSK', 'JD', 'ADM', 'CRH', 'HLN']


# In[33]:



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
    
else:
    ticker_list_2 = []


# In[50]:


st.header(f'Top Stocks by Market Cap {index_selection}')
# Plot a treemap using Plotly Express
fig = px.treemap(top_10_stocks, path=['LongName'], values='MarketCap',
                 color='MarketCap')
st.plotly_chart(fig)


# In[ ]:





# In[35]:


def plot_sma_vs_closing_price(stock_symbol, start, end):
    # Retrieve stock data using yfinance
    stock_data = yf.download(stock_symbol, start, end)
    
    # Calculate Simple Moving Average (SMA)
    sma_period = 60
    stock_data['SMA'] = stock_data['Close'].rolling(window=sma_period).mean()
    
   # Create a Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA'], mode='lines', name=f'SMA {sma_period}'))
    
    # Customize the layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Price',
        title=f'{stock_symbol} Closing Price vs. SMA',
        legend=dict(x=0, y=1)
    )
    
    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)


# In[36]:



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
        yaxis_title='Price',
        title=f'{stock_symbol} Closing Price vs. EMA',
        legend=dict(x=0, y=1)
    )
    
    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)


# In[37]:


with st.sidebar: indicator_type = st.selectbox("Select Indicator Type", ["sma", "ema"])
                                               


# indicator_column_mapping = { "sma": "SMA",
#                             "ema": "EMA"}
# 
# indicator_column_name = indicator_column_mapping.get(indicator_type)

# In[38]:


if indicator_type == "sma":
    st.title("Stock SMA vs. Closing Price")
    sma_plot = plot_sma_vs_closing_price(selected_stock, start_date, end_date)
    
elif indicator_type == 'ema':
        st.title("Stock EMA vs. Closing Price")
        ema_plot = plot_ema_vs_closing_price(selected_stock, start_date, end_date)       


# In[ ]:




