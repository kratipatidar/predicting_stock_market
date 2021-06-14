from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

# here we scrape the data from FinViz website
finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['GME', 'TSLA']

# defining a dictionary for the news tables and getting the URLs for each company
news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker
    # now we request the data using the URLs
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    # saving the scraped data to a dictionary, by mapping using suitable ticker symbol
    html = BeautifulSoup(response, 'html')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

# next we will mine the required data from scraped data

## first we define an empty list
parsed_data = []

## next we extract the news titles and date
for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.get_text()
        date_data = row.td.text.split(' ')

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]

        ## appending all the extracted data to the list
        parsed_data.append([ticker, date, time, title])

# creating a dataframe out of the parsed data
df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])

# calling the sentiment analysis function
vader = SentimentIntensityAnalyzer()

# applying the sentiment analysis to our data
f = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['title'].apply(f)

# altering the dataframe for visualization purposes
df['date'] = pd.to_datetime(df.date).dt.date
mean_df = df.groupby(['ticker', 'date']).mean()
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound', axis='columns').transpose()
print(mean_df)

# finally we plot our results
plt.figure(figsize=(16, 12))
mean_df.plot(kind='bar', grid=True, color=['orange', 'green'])
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Compound Scores')
plt.title('Compound Scores of each Company - Using FinViz Data')
plt.show()
