## Predicting the Stock Market

### Background

Social media has become an integral part of our society in the 21st century. It is accessible through almost any electronic medium people come in contact with, tracking every facet of our lives for the world to see. This research centers on the situation which occurred between users from a social media site called “Reddit” and wall street hedge fund companies. Reddit is a social media platform where users can share thoughts and opinions on thousands of topics among its users. Users of the site decided to go against wall street for what they deemed unfair practices in regards to stock market trading. The situation that ensued was that the user base of Reddit bought stock from a video game retailer named GameStop, in order to artificially increase its value, shacking up how trading was viewed and done for wall street traders. The aftermath left many losing billions of dollars and new interest in how large groups can affect society through social media networking. In this project we create a neural network and perform sentiment analysis on Reddit and FinViz (a financial news website) to identify trends prevalent on these sites and their impact on the financial markets in regards to trading stocks.

### Introduction

The main objective is to obtain a greater understanding on how neural networks are built and how they function for predictive modeling. The project aims to utilize data from reddit, a social media platform, and financial data to create a neural network which can predict the stock market trends based on reddit sentiment analysis. This means we have a neural network that will predict the stock market price based on historical data and a separate sentiment analysis based on reddit posts and finviz news articles. Reddit is an online forum which utilizes long form text posts with images and video capabilities to allow users to express their feelings on a variety of topics. Reddit itself is structured by different forums based on topics, called “subreddits”, and calls its submissions “posts”. The financial data was downloaded from Yahoo Finance, with a free to use license as long as it is for research purposes. We also obtained financial news data from FinViz to supplement our sentiment analysis on reddit posts. We qualify our sentiment analysis to the companies GameStop and Tesla, since these companies tend to be frequently featured in a lot of online posts and news articles. For our financial analysis centered on developing a neural network, we also consider Apple and AMC (American Multi-Cinema) along with GME (GameStop) and Tesla to evaluate our model’s performance.

### Goal and Prediction

Our goal, along with modeling a deep learning model that can use historic data to accurately predict stock prices was to see if a trending financial topic on reddit would have a true impact on the stock market. Our prediction, utilizing past and recent occurrences, is that there will be a direct correlation between the fluctuation in the stock market based on our sentiment analysis results.

### Methodology and Analysis

We divide our analysis into two parts as follows:
#### Part-1: Sentiment Analysis
We conducted sentiment analysis on Reddit posts and news articles on FinViz website specifically based on our companies of interest - GameStop (GME) and Tesla (TSLA). We achieved this through the following steps:
1. For analyzing reddit posts,we first used the PushShiftAPI to scrape the relevant posts off reddit. We were only able to obtain posts dated from mid- April due to the API and platform limitations.
2. For the analysis of news articles on FinViz,we used BeautifulSoup to extract the corresponding article titles and consequently save them in a data structure.
3. We then mined the date,titles and company tickers(stock symbols,for instance GME will be illustrated by '$GME') of these posts for further analysis.
4. Next, we filtered this mined data to get the data for our companies of interest- GameStop and Tesla. This had to be done specifically for data from reddit since in the case involving finviz, we were able to directly scrape data specific for GME and Tesla.
5. Consequently, we called the sentiment intensity analyzer function of the 'vader' package in 'nltk' to calculate compound scores of each post title. The higher the compound score, the higher the tendency is of the post being positive. We found the mean of these scores for each company on each date to get an overview of the general sentiment towards the companies' stocks.
6. Finally, we plotted the analysis results to better observe the sentiment trends across the companies and the two platforms.

#### Part-2: Financial Analysis

Following our Sentiment Analysis, we move on to building an artificial Recurrent Neural Network using Long Short Term Memory (LSTM) to predict the closing prices of GME (GameStop), TSLA (Tesla). The reason we use RNN (a feedback network) with LSTM is that this network is capable of persisting important information in its memory.
We also tested our model performance on two other companies, namely, AMC (American Multi-Cinema) and AAPl (Apple) to check the performance of our neural network. We use past 60-day stock prices to predict the next day closing price. The following steps illustrate the process of building a deep learning model:
1. First, we used pandas data-reader to scrape stock price data off Yahoo.
2. Next, we plotted the closing prices to observe the general trend followed by the stock prices of a specific company.
3. We pre-processed the training data in order to make it suitable for feeding into the neural network. Our training dataset comprised stock prices from 2012 to 2019 for each company of interest.
4. We then built and compiled our neural network model,and consequently trained it on our training dataset.
5. We then tested this model on our test dataset(code shown below),which contains the stock data from the year 2020 to April 25, 2021.
6. Following this, we calculated the Root Mean Square Error(RMSE) of our model for testing its performance across different datasets specific to each company.
7. Finally, we generated plots(using the code attached)for the actual and predicted values to observe how close or far away our predictions are.

### Results

The neural network model that we built utilizes purely financial data in order to make its predictions. Currently, we utilize our sentiment analysis to visually identify negative sentiments for a specific date and see what correlations there are to our financial algorithm. This has yielded good results to conclude that there is a correlation between negative sentiment and a dip in the stock market.

To ensure that our financial model performed reliably, we utilized Apple (AAPL) stock data in order to compare a stock that was not affected by the Reddit phenomenon. Per our analysis, we saw that AAPL was not affected by the movement, and we had low RMSE scores for our AAPL analysis, which furthered the confidence of our analysis. The two stocks most affected by the Reddit phenomenon was GameStop (GME) and Tesla (TSLA). In our results we saw erratic behaviour all through the time period we analyzed, which correlated with our sentiment analysis observations.

### Conclusion

Social media and online networks have changed how people interact with each other and their environment. Ever since the collective efforts by redditors to displace the financial markets occurred, changes have occurred on how the financial industry views online groups and social media. It prioritized many industries to understand social networks in a deeper maner and placed more power on the users who are a part of them.

We created a neural network which utilized financial data to make predictions into the next day. We further created a sentiment analysis to analyze the sentiment from the users of Reddit and other articles from a financial site called FinViz. We concluded that our neural network did function accurately and we were able to gain insights into stock market predictions based on our sentiment analysis. Our team ran into several limitations with hardware to process our neural network, limitations in data we could obtain from social media sites and combining our neural network and sentiment analysis into one algorithm. The potential to apply this kind of research to other topics such as video game sales or trending car sales is something that could be further explored.
