## Chapter 7: Event Recommendation System

### 0. Clarification
- Personalized event recommendation system
- Event is ephoemral one-time occurance, a user cannot register after the event is finished
- User can register, invited, and form friendships
- Construct training data based on user interaction.
- goal is to increase ticket sale, free or paid

### 1. Frame as ML task
1.1 Define the ML objective: maximize the number of event registrations.
1.2 Specify input and output: user => top k events ranked by relevance
1.3 Choose the right ML category
- Simple rule based, popular events => baseline model
- Embedding based models: content based or collaborative filtering
- Reformulating it into a ranking problem, LTR
    - pointwise LTR: <query, item> => relevance score
    - pairwise LTR: <query, item1, item2> => item2 > item2
    - listwise LTR: ...

We use pointwise as it is simpler. We take <user, event> and use binary classifier to predict the prob of user will register it.

### 2. Data Preparation
2.1 Data Engineering
- Users
- Events

| Event ID | Host user ID | Category/subcategory | Description | Price | Location | Data/Time |
|..........|..........|..........|..........|..........|..........|..........|

- Friendships

| User Id 1 | UserID 2 | Timestamp|
|..........|..........|..........|

- User-event interaction

| User ID | Event ID | Interaciton type | Interaction value| Location | Timestamp |
|..........|..........|..........|..........|..........|..........|

2.2 Feature Engineering

Note that event is short-lived and not many historical interactions available for a given event. Therefore, cold-start and new-item problem. Therefore, we put effort into feature engineering. 
- Location-related features
    - transit score & transit score similarity (with user)
    - walk score & walk score similarity
    - bike score & score similarity
    - same city with the user?
    - same country with the user?
    - distance between event loc and user loc
    - distance similarity (diff between event loc and user prev distance median)
- Time-related features
    - the remaining time till the event begins
    - remaining time similarity
    - estimated travel time and similarity
    - user historical event weekday profile => day similarity and hour similarity
- Social-related features
    - number of people registered
    - ratio of total regist to total impression
    - registered friend similarity (diff between num of reg friends and prev reg events)
    - num of friends who invited this user to the event
    - num of fellow users who invited this user to the event
    - is the event's host a friend of the user
    - how often this user attend prev events hosted by this host?
    - host popularity
- User-related features
    - age, gender
- Event-related features
    - price
    - price similarity
    - event description similarity (TFIDF)
    - event tags, category

Some potential points to talk about
- batch/static vs streaming/dynamic features
- feature computation efficiency: instead of pass similarity, pass user loc and event loc to model and learn it automatically.
- use a decay factor: more weights on user's recent interactions
- use embedding learning: convert each user and event into an embedding vector
- create features from users' attribute may create bias, like age and gender

### 3. Model Development
3.1 Model selection 
- LR
- DT
- bagging: averaging reduce variance
- boosting: reduce both v and b (GBDT) => baseline
- NN

3.2 Model training

Construct Dataset => <user, event> pair, and label => imbalanced

Choose loss: CEL

### 4. Evaluation
4.1 Offline metrics:
- Precision@k, Recall@k
- mAP, MRR, nDCG

4.2 Online metrics:
- CTR
- conversion rate = total num of regis/total num of impression
- bookmark rate
- share rate
- revenue lift 

### 5. Serving
- online learning pipeline: requires continuously fine-tune as cold-start and new-items
- prediction pipeline
    - event filtering: use location, or user specified types to quickly reduce from 1M
    - ranking: use the pred prob



- [x] [Data leakage](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [x] [Online training frequency](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html#towards-continual-learning)
