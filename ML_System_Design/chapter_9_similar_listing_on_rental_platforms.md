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
Loss = \sum_{(c,p) \in D_p} \log(\frac{1}{1+e^{-E_p*E_c}}) + \sum_{(c,n) \in D_n} \log(\frac{1}{1+e^{E_n*E_c}})
```
