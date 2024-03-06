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
- Videos

| Video ID | Length | Manual tags | Manual titles | Likes | Views | Language |

- Users

| User ID | Username | Age | Gender | City | Country | Email | Language | Time zone | 

- User-video interactions

| User ID | Video ID | Timestamp | Interaction type | Interaction value | Location |

2.2 Feature Engineerinf

- Video features
    - video ID => embedding layer is learned during model trainig
    - video length => people prefer videos of diff duration
    - language => embedding layer
    - title and tags => statistical or light-waight ML-based for tags, and BRRT for titles
- User features
    - user demographic
    - user historical interactions
        - search history: avg all prev search query embedding
        - liked videos: video ID mao to embed vectors using the embed layer, then avg 
        - watched videos and impression: same as liked videos
    - contexual information: time of day, day of week, device

### 3. Model Development
3.1 and 3.2 Model selection and training

- Matrix factorization
    - Feedback matrix with combination of explicit feedback and implicit feedback
    - MF decompose it into a user embedding and item embedding, such that distance represent relevance.
    - training: random initialize 2 embed matrices, then iteratively optimize the embedding to decrease the loss the pred score matrix and feedback matrix. (SGD or WALS, short for weighted alternating least squares)
    - Loss is a weighted comb of squared distance over observed <user, video> pair and unobserved pairs. Use weighted due to the matrix being sparse
    ```math
    L = \sum_{(i,j in obs)} (A_{ij} - U_i * V_j) ^ 2 + W \sum_{(i,j not in obs)} (A_{ij} - U_i * V_j) ^ 2
    ```
    - Inference: output a dot product for similarity measure
    - As MF only use interaction data, not video features, it is commonly used in **collabortive filtering**.
    - Pros:
        - fast training speed
        - fast service speed
    - Cons:
        - only reply on user-video interaction
        - diff at handling new users

- Two-tower Neural Network:
    - A video encoder, a user encoder
    - The dist between embeddings in the shared embed space represent their relevance
    - Construct dataset: <user, video> pairs, construct positive and negative pairs (random video or disliked videos).
    - Loss function: we frame it as a classification task, therefore, cross-enntropy loss
    - Inference: Approximate NN to find the top k most similar/relevant videos embeddings efficiently.
    - Good for both CF (when CF video encoder just embed video ID to a vector) and content-based. 
    - Pros: user user features and can handle new users.
    - Cons: slow serving, training expensive.

### 4. Evaluation
4.1 Offline metrics:
- Precision@k
- mAP
- Diversity

4.2 Online metrics:
- CTR
- num of completed videos
- total watch time
- explicit user feedback

### 5. Serving
In a two-stage design, a **lightweight model** to quickly narrow down (candidate gene, do not use video feature); then a havier model that accurately scores and rank (both user and video features); then rerank.

- k candidate generation model using shallow two-tower NN for efficiency
- score use deep two-tower NN for accuracy
- reranking(region restricted, freshness, misinformation, duplicate, fairness and bias)

Common challenges:
- serving speed
- precision
- diversity (multiple candidate generator)
- fairness: Make separate models for underserved groups; Track metrics (for example, accuracy and absolute error) on each demographic to watch for biases.
- cold-start problem
    - new users
        - quick fill profiles
        - calc an average user
        
    - new videos
        - video metadata 
        - no interaction: multi-arm bandit, display it to random user to collect interaction
        - then fine-tune the NN using the new interaction



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
