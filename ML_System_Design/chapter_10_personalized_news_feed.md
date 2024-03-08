## Chapter 10: Personalized News Feed

### 0. Clarification
- the system retrieves unseen posts or posts with unseen comments
- ranks them based on how engaging they are to user
- take no longer than 200 ms
- objective of the system is to increase user engagement
    - engagement includes, impression, click, like, share, comments...

### 1. Frame as ML task
1.1 Define the ML objective
- maximize the num of specific implicit reactions, such as dwell time or user clicks
- maximize the num of specific explicit reactions, such as like or share
- maximize a weightd score based on both implicit and explicit reactions

| Reaction | click | like | comment | share | friendship request | hide | block|
|----------|----------|----------|----------|----------|----------|----------|----------|
| Weight   | 1 | 5 | 10 | 20 | 30| -20 | -30|


1.2 Specify input and output: user => personalized news feed system => a ranked list of posts based on engagement

1.3 Choose the right ML category:
- Pointwise Learning to Rank (LTR) to rank posts by engagement scores (between a user and a post)
- A binary classifier to predict the prob of various reactions for a <user, post> pair

```math
\text{Engagement score} = \sum_{reaction \in reactions} p_{\text{reaction}} * w_{\text{reaction}} 
```

### 2. Data Preparation
2.1 Data Engineering
- Users
- Posts

| Post ID | Author ID | Texual content | hashtags | mentions | image or videos | timestamp | 

- User-post interactions
- **Friendship**

| User ID 1 | User ID 2 | Time when friendshio was formed | Close friend | Family memebr | 

2.2 Feature Engineering
- post features
    - textual content - BERT
    - image or video - ResNet, CLIP, SimCLR
    - reactions
        - num of likes, comments, share, use bucketing and normalization
    - hashtags
        - tokenize (Viterbi), token to IDs (feature hashing), vectorization (TFIDF, word2Vec)
    - post's age (bucket + one-hot)
- user features 
    - demographic: age, gender, country
    - contextual: device, time of day
    - **user-post historical interactions**: a list id post IDs that this user engaged with => extract features from these posts then average
    - **being mentioned in the post**
- **user-author affinities**
    - This is among the most important features in predicting a user's engagement on FB!
    - Like/click/comment rate
    - Length of friendship
    - close friends and family

### 3. Model Development
3.1 Model selection 
- N independent DNNs for each reaction
    - expensive to train
    - for less freq reactions, not enough data
- Multi-task DNNs
    - check chapter 5 harmful content detection
    - learn similarities between tasks and reduce computation
- Improve model for passive users by adding prediciton head for
    - dwell-time: time user spends on a post
    - skip: a user spends less than t seconds (0.5s)

3.2 Model training
- construct pos/neg dataset for each reaction type
- loss function, BCL for classification, MAE, MSE, Huber loss for regression

### 4. Evaluation
4.1 Offline metrics:
- Precision, recall, F1, ROC-AUC

4.2 Online metrics:
- CTR
- Reaction rate
- Total time spend on news feed per day/week/month
- User satisfication rate from user survey

### 5. Service
5.1 Data preparation pipeline
- compute batch and online features
- continuously generate training data from new posts and interactions

5.3 prediction pipeline
- retrieval service: effectively retrieve unseen posts
- ranking service
- re-rank (demotion misinfo and creepy and user filter), diversify



### Reference
- [x] [News Feed Ranking in Facebook](https://engineering.fb.com/2021/01/26/ml-applications/news-feed-ranking/)
- [x] [Twitter's news feed system](https://blog.x.com/engineering/en_us/topics/insights/2017/using-deep-learning-at-scale-in-twitters-timelines#:~:text=Using%20Deep%20Learning%20at%20Scale%20in%20Twitter%E2%80%99s%20Timelines,companies.%20...%203%20Impact%20...%204%20Acknowledgements%20)
- [x] [LinkedIn's News Feed system](https://www.linkedin.com/blog/engineering/feed/understanding-feed-dwell-time)
- [ ] [BERT paper]
- [x] [ResNet model]
- [x] [CLIP model: learns visual concepts from natural language supervision](https://openai.com/research/clip)
- [ ] [Viterbi algorithm for hashtag segmentation](https://en.wikipedia.org/wiki/Viterbi_algorithm)
- [x] [TF-IDF: a measure of importance of a word to a document in a collection or corpus](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [x] [Word2Vec](https://en.wikipedia.org/wiki/Word2vec)
- [x] [Serving a billion personalized new feed](https://www.youtube.com/watch?v=Xpx5RYNTQvg)
- [x] [MSE loss](https://en.wikipedia.org/wiki/Mean_squared_error)
- [x] [MAE loss](https://en.wikipedia.org/wiki/Mean_absolute_error)
- [x] [Huber loss](https://en.wikipedia.org/wiki/Huber_loss)
- [x] [Binary cross entropy loss](https://en.wikipedia.org/wiki/Cross-entropy)
- [ ] [A news feed system design](https://liuzhenglaichn.gitbook.io/system-design/news-feed/design-a-news-feed-system)
- [ ] [Predict viral tweets](https://towardsdatascience.com/using-data-science-to-predict-viral-tweets-615b0acc2e1e)
- [ ] [Cold start problem in recommendation systems](https://en.wikipedia.org/wiki/Cold_start_(recommender_systems))
- [ ] [Positional bias](https://eugeneyan.com/writing/position-bias/)
- [x] [Determine retraining frequency](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html#continual-learning)