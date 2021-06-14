from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from psaw import PushshiftAPI
import datetime

# First we get all the posts off reddit
api = PushshiftAPI()

start_time = int(datetime.datetime(2020, 1, 1).timestamp())

submissions = list(api.search_submissions(after=start_time,
                                          subreddit='wallstreetbets',
                                          filter=['url', 'author', 'title', 'subreddit'],
                                          limit=20000))

# Storing the relevant posts into a dataframe
post_date = []
post_title = []
post_ticker = []
post_tickers_first = []
for submission in submissions:
    words = submission.title.split()
    cashtags = list(set(filter(lambda word: word.lower().startswith('$'), words)))

    if len(cashtags) > 0:
        post_ticker.append(cashtags)
        post_title.append(submission.title)
        post_date.append(submission.created_utc)

print(post_date)
print(post_title)
print(post_ticker)

for ticker in post_ticker:
    post_tickers_first.append(ticker[0])

# Making a dataframe out of collected posts
reddit_posts = pd.DataFrame(columns=['date', 'ticker', 'title'])
reddit_posts['date'] = post_date
reddit_posts['ticker'] = post_tickers_first
reddit_posts['title'] = post_title

# Changing certain columns
reddit_posts['date'] = pd.to_datetime(reddit_posts['date'], unit='s')
reddit_posts['date'] = pd.to_datetime(reddit_posts.date).dt.date

# Filtering this dataframe on the relevant tickers we want
reddit_posts_2 = reddit_posts.loc[reddit_posts.ticker.isin(['$GME', '$TSLA'])]
print(reddit_posts_2)

# Conducting Sentiment Analysis on post titles
vader = SentimentIntensityAnalyzer()
f = lambda title: vader.polarity_scores(title)['compound']

reddit_posts_2['compound'] = reddit_posts_2['title'].apply(f)
mean_df = reddit_posts_2.groupby(['ticker', 'date']).mean()
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound', axis='columns').transpose()

# Plotting the Sentiment Analysis Results
plt.figure(figsize=(16, 12))
mean_df.plot(kind='bar', grid=True, color=['orange', 'green'])
plt.legend(loc='best')
plt.xlabel('Date')
plt.ylabel('Compound Scores')
plt.title('Compound Scores of each Company - Using Reddit Data')
plt.show()


