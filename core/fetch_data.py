from yfinance import Ticker, download
from webull import paper_webull as pwb
from datetime import datetime, date, time
from backtesting import Backtest, Strategy
from pandas_market_calendars import get_calendar
from yfinance.exceptions import YFRateLimitError

def fetch_data(ticker,get_dataframe=True):
    """
    Fetches historical stock data for a given ticker using yfinance or Webull.
    """
    print(f"Fetching data for {ticker}...")
    try:
        df = Ticker(ticker).history(period='max')
        if not df.empty:
            print(f"Successfully fetched data for {ticker} using yfinance.")
            return df
    except YFRateLimitError:
        print(f"yfinance rate limit hit for {ticker}. Attempting alternative method.")
        df = download(ticker, multi_level_index = False, auto_adjust=True, period='max', progress=False)
        if not df.empty:
            print(f"Successfully fetched data for {ticker} using yfinance download.")
            return df
        print(f"yfinance failed for {ticker}. Attempting to fetch data using Webull paper trading API.")
        start_date = to_datetime(pwb().get_ticker_info(ticker)['inceptionDate']).date() if 'inceptionDate' in pwb().get_ticker_info(ticker) else to_datetime('01-01-2000').date()
        today_date = date.today()
        dates = [(start_date+Timedelta(days=365*(i))) for i in range(int(ceil((today_date - start_date)/Timedelta(days=365))))]
        days = (today_date - dates[-1]).days
        if days < 365 and days > 0: dates[-1] = today_date
        dictionary = {'Start Date: ' + dates[i].strftime('%Y-%m-%d') : pwb().get_bars(stock=ticker,
        count=len(get_calendar('NYSE').schedule(dates[i],dates[i+1])), interval='d1',
        timeStamp=int(datetime.combine(dates[i+1]+Timedelta(days=1),
time(0,0)).timestamp())) for i in range(len(dates)) if i!=(len(dates)-1)}
        if get_dataframe:
            if dictionary:
                df = concat(list(dictionary.values())).reset_index().drop_duplicates().set_index('timestamp')
                print(f"Successfully fetched data for {ticker} using Webull.")
                return df
            else:
                print(f"No data fetched for {ticker} using Webull.")
                return DataFrame()
    print(f'No data found for {ticker}.')
    return DataFrame()