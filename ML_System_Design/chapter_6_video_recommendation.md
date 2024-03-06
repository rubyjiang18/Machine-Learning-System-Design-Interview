## Chapter 6: Video Recommendation System

### 0. Clarifying requirements
- homepage video recommendation system that recommend personalized videos to users as they load the homepage
- construct the dataset based on user interaction
- 10B videos, load takes <= 200 ms

### 1. Frame as ML task
1.1 Define the ML objective
- business goal is to increase user engagement
- ML objectives
    - maximize the num of user clicks (click-bait)
    - maximize the num of completed videos (recom short videos that are quick to complete)
    - maximize total watch time
    - **maximize the number of relevant videos**
        - define relevance based on implicit and explicit user interactions (click like, watch at least half)
1.2 Specify input and output: user => video rec system => recommended videos
1.3 Choose the right ML category

Three common types of personalized recommendation system:
- Content-based filtering
- Collaborative filtering
- Hybrid filtering

##### Content-based filtering
Find similar videos to the videos the user engaged.

Pros
- ability to recommend new videos
- ability to capture the unique interest of users

Cons
- difficult to discover a user's new interests
- requires domain knowledge, i.e., we need to engineer video features manully

##### Collaborative filtering (CF)
CF user user-user similarity and video-video similarity to recom new videos. Intuitive is that find similar users (based on prev interactions) are interesed in similar videos.

Pros
- does not need video features
- easy to discover new area of interest
- efficient than content-based

Cons
- Cold-start problem
- cannot handle niche interests

##### Hybrid filtering
This method combines both CF and content-based filtering sequentially or in parallel, usually sequential hybrid filtering. This leads to better results as it not only reply on user historical interactions but also video features.

We use 
- CF-based model as the first stage candidate generator
- followed by a content-based model as second stage to recom videos

### 2. Data Preparation
2.1 Data Engineering

2.2 Feature Engineerinf


### Reference
- [x] [Youtube recommendation system](https://blog.youtube/inside-youtube/on-youtubes-recommendation-system/)
- [x] [DNN for Youtube recommendation](http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)
- [ ] [CBOW](https://arxiv.org/pdf/1301.3781.pdf)
- [x] [BERT paper](https://arxiv.org/abs/1810.04805)
- [x] [Matrix factorization](https://developers.google.com/machine-learning/recommendation/collaborative/matrix)
- [x] [Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
- [x] [WALS optimization](https://fairyonice.github.io/Learn-about-collaborative-filtering-and-weighted-alternating-least-square-with-tensorflow.html)
- [x] [**Instagram multi-stage recommendation system**](https://instagram-engineering.com/powered-by-ai-instagrams-explore-recommender-system-7ca901d2a882)
- [x] [Exploration and exploitation trade-off](https://en.wikipedia.org/wiki/Multi-armed_bandit)
- [x] [Bias in AI and recommendation systems](https://www.searchenginejournal.com/biases-search-recommender-systems/339319/)
- [ ] [Ethical concerns in recommendation systems](https://link.springer.com/article/10.1007/s00146-020-00950-y)
- [ ] [Seasonality in recommendation systems](https://www.computer.org/csdl/proceedings-article/big-data/2019/09005954/1hJsfgT0qL6)
- [ ] [A multitask ranking system](https://daiwk.github.io/assets/youtube-multitask.pdf)
- [ ] [Benefit from a negative feedback](https://arxiv.org/pdf/1607.04228.pdf)
