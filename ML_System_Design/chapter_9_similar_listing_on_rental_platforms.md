## Chapter 9: Similar Listings on Vacation Rental Platforms

### 0. Requirements
- Recommend items similar to those a user is currently viewing
- defination of similar: same neighborhood, city, price range, etc
- same recommend lists for both signed in user and anonymous users
- do not use user rated demo data

### 1. Frame as ML task
1.1 Define the ML objective: accurately predict which listing the user will click nest, given the listing the user is currently viewing

1.2 Specify input and output: user currently viewing listing => a ranked list of listings based on click prob

1.3 Choose the right ML category:

- Similar to Instgram "Explore" feature, Airbnb listing feature, and Word2Vec approch, **learn item embedding using co-occurences of items in user's browsing histories**. 
- We build a system by training a model that maps each listing to an embedding vector, so if two listings co-occur in user's browsing history, their embedding vectors are in close proximity in the embedding space.
- To recommend, we use ANN.

### 2. Data Preparation
2.1 Data Engineering

- Users
- Listings

| Listing ID | Host ID | Price | Sq ft | Rating | Type | City | Num of beds | Max guests|

- User-listing interactions

| ID | User ID | Listing ID | Position of the listing in the displayed list | Interaction Type | Source | Timestamp|

Interaction Type can be "click" or "book", source can be "search feature" or "similar listing feature".

2.2 Feature Engineering

- the model only use browsing histories (called sessions) for training
- A session is a sequence of clicked listing IDs, followed by an eventualy booked listing, without interruption.

| Session ID | Clicked listing IDs | Eventually booked listing ID | 
|----------|----------|----------|
| 1 | 1, 5, 4, 9 | 26 |
| 2 | 6, 8, 9, 21, 6, 15, 6 | 5|


### 3. Model Development
3.1 Model Selection
- To learn embedding, we need NN, we use a shallow one to learn listing embedding.

3.2 Model training
- The training starts by initializing listing embedding to random vectors
- These embeddings are learned gradually by **reading through search sessions**, using the sliding window method
- As window slide, the embed of central listing is updated to be similar to the embed of other listings in the window, and dissimilar from listings outside the window
- Retrain daily on the newly constructed data

3.3 Construct the dataset
- positive pairs are <central list, any list within the window>
- negative pairs are <central list, randomly sampled listings>

3.4 choose the loss function
- compute similarity between two embeddings
- apply sigmoid to distance to a prob value in [0,1]
- use cross-entropy as a standard classification loss

```math
argmax \sum_{(c,p) \in D_p} \log(\frac{1}{1+e^{-E_p*E_c}}) + \sum_{(c,n) \in D_n} \log(\frac{1}{1+e^{E_n*E_c}})
```

3.5 Can we improve the loss function to learn better embed?
- **Add the eventually booked listing as a global context**
- **Add negative pairs from the same region/neighborhood to the training data**


```math
argmax \sum_{(c,p) \in D_p} \log(\frac{1}{1+e^{-E_p*E_c}}) + \sum_{(c,n) \in D_n} \log(\frac{1}{1+e^{E_n*E_c}}) + \sum_{(c,b) \in D_{booked}} \log(\frac{1}{1+e^{-E_p*E_c}}) + \sum_{(c,n) \in D_{hard}} \log(\frac{1}{1+e^{E_n*E_c}})
```

### 4. Evaluation
To evaluate what embedding learns
- To evaluate if geographical similarity is encoded we performed k-means clustering on learned embeddings.
- Next, we evaluated average cosine similarities between listings of different types (Entire Home, Private Room, Shared Room) and price ranges and confirmed that cosine similarities between listings of same type and price ranges are much higher compared to similarities between listings of different type and price ranges. Therefore, we can conclude that those two listing characteristics are well 
 

4.1 Offline metrics:

Test how  good the new model is at prediciting the **eventually-booked listing**, based on the latest user click. We create a metric call **average rank of the eventually booked listing**.

Steps:
    - given a search session consiting of 
    ```
    L0 (first clicked listing) | L1, L2, L3, L4, L5, L6 (final booked)
    ```
    - use modelto get embedding
    - re-rank the L1 to L6 based on similarity score with L0
    - If L6 (final booked) was ranked high, it indicates a good model
    - We average the rank of final booked listing across all sesscions in the val dataset to compute this metric


4.2 Online metrics:
- CTR = num clicked listings / num of recommended listings
- Session book rate = num of session turned into booking / total num of sessions


### 5. Serving
5.1 Training pipeline

5.2 Indexing pipeline

5.3 Prediciton pipeline
- embedding fetcher service
    - if model seen listing, fetch embed from the index table
    - if model not seen, use heuristics (To create embeddings for a new listing we find 3 geographically closest listings that do have embeddings, and are of same listing type and price range as the new listing, and calculate their mean vector.) to handle new listings; when collected enough interaction data, retrain model
- nearnest neighbor service
- re-rank: apply user filters and certain constrains


### Reference
- [x] [Instgram's Explore recommender system](https://ai.meta.com/blog/powered-by-ai-instagrams-explore-recommender-system/)
    - Ig2vec treats account IDs that a user interacts with — e.g., a person likes media from an account — as a sequence of words in a sentence.
    - to narrow down candidates, train a super-lightweight model that learns from and tries to approximate our main ranking models as much as possible (ranking distillation model)
- [x] [Airbnb listing embedding in search ranking](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e)
- [ ] [Word2Vec]()
- [ ] [Negative sampling technique](https://www.baeldung.com/cs/nlps-word2vec-negative-sampling)
- [ ] [Positional bias](https://eugeneyan.com/writing/position-bias/)
- [ ] [Random walk]()
- [ ] [Random walk with restarts]()
- [ ] [Seasonality in recommendation systems](https://sci-hub.ru/10.1109/bigdata47090.2019.9005954)


