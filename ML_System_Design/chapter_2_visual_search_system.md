## Chapter 2: Visual Search System

### 0. Clarifying requirements
- A visual search system that retrieves images similar to the user query image, rank them based on similarity to the query image. 
- The platform only supports images, no video, no text queries allowed. 
- For simplicity, no personalization.

### 1. Frame as an ML task
- Define the ML objective: accurately retrieve images that are visually similar to the user query image.
- Specify system's input and output
- Choose the right ML category: ranking problem and representation learning

### 2. Data Preparation
2.1 Data Engineering
- **Images**: ID, Owner ID, Uoload time, Manual tags
- **Users**: ID, Username, Age, Gender, City, Country, Email
- **User-image interaction**: Impression, Click (Collect, share, like... if allowed)
    UserID, Query Image ID, Display Image ID, Position in the displayed list, Interaction Type, Location, Timestamp

2.2 Feature Engineering
For images, resize, scale, normalization, consistent color mode...

### 3. Model Development
3.1 Model selection: CNN and transformer based

3.2 Model training: constrastive learning that train the model to distinguish similar and dissimilar images.

3.3 Construct training dataset
To label positive images:
- Use human judgement: expensive and time consuming
- Use user interaction (clicks) as a proxy for similarity: noisy and sparse training data
- Artificially create a similar image from the query image, known as self-supervision: replys on data augmentation, such as SimCLR and MoCo, no manual work required, not noisy. Only drawback is that constructed training data differs from the real data.

3.3 Choose the loss function to measure the quality of produced embedding
Constrastive loss
- Compute **similarity** between the query image and the embeddings of other images
- Apply **softmax** to the computed distances => values sum up to 1 
```math
P(y=i|x) = \frac{e^{x_i}}{\sum_{j}e^{x_j}}
```
- **Cross entropy** to measure how close the pred prob are to the ground truth
```math
L(y, \hat{y}) = -\sum_{i=1}^{N} y_i \log(\hat{y}_i) = \sum_{i=1}^{N} y_i \log(\frac{1}{\hat{y}_i})
```

### 4. Evaluation
4.1 Offline metrics - recommendation system
- Mean Reciprocal Rank (MRR)

It considers only the rank of the first relevant item in the output list, and does not measure the precision and ranking quality of a ranked list.
```math
MRR = \frac{1}{m}\sum_{i=1}^{m}\frac{1}{ranki}
```
- Recall@K

```math
recall@K = \frac{\text{num of relevant items among the top k items in the output list}}{\text{total num of relevant items in the entire dataset}}
```

Recall@K is not helpful when denominator is very high, where our goal is to retrieve a handful of the most similar images. It does measure the ranking quality of the model. Do not use.
- Precision@K

```math
Precision@K = \frac{\text{num of relevant items among the top k items in the output list}}{k}
```

This metric measures how precise the output lists are, but it does not consider the ranking quality. Do not use.
- Mean average precision (mAP)

This metric first computes the AP for each output list, and then averages AP values.
```math
AP = \frac{\sum^{k}_{i=1} \text{Precision@i if i'th item is relevant to the user}}{\text{total num of relevant items}}
```

Overall ranking quality of the list is considered. However, it is designed for **binary relevance**. For continuous relevance score, nDCG is better.

- nDCG

This metric measures the ranking quality of an output list compared tothe ideal ranking.

DCG calcuates the cumulative gain of items in a list by summing up the relevance score of each item. Then the score is accumulated from the top of the output list to the bottom, with the score of each result discounted at lower ranks.
```math
DCG_p = \sum_{i=1}^{p} \frac{rel_i}{\log2(i+1)}
```
where $rel_i$ is the ground truth rel score of the image ranked at locaiton i.

nDCG is a normalized version of DCG as DCG can take any values, normalize it by the DCG of an ideal ranking list.
```math
nDCG_p = \frac{DCG_p}{IDCG_p}
```
Its primary shortcoming is that deriving the ground truth rel scores is not always possible. But in our case, in eval set, we use similarity score, we can use NDCG.

4.2 Online metric
- CTR
```math
CTR = \frac{\text{num of clicked images}}{\text{total num of suggested images}}
```
- Avg daily, weekly, monthly time spent on the suggested images.

### 5. Serving - Prediction & Indexing
5.1 Prediction pipeline
- Embedding generation 
- Nearest neighbor service 
    - Exact nearest neighbor (linear search): search the entir index table. O(NxD)
    - Approximate nearest neighbor: O(logN x D)
        - Tree base ANN: iteratively adding new criterion to each node.
        - Locality sensitive hasing (LSH) based ANN: These hash functions map points in close proximity to each other into the same bucket.
        - Clustering based ANN: group points based on similarity.
- Re-ranking service 
 
Business level logic and policies. Filter inapproriate results (private images, near duplicates).

5.2 Indexing pipeline
- Indexing service: store the embed of the entire image in an index table.

### 6. Follow up
- How to use image metadata, such as tags, to improve search results. See chapter 3
- How to support the ability to search images by a textual query. See chapter 4

### Reference
- [x] [Visual search at pinterest](https://arxiv.org/pdf/1505.07647.pdf)
- [x] [Unifying visual embeddings for search at Pinterest](https://medium.com/pinterest-engineering/unifying-visual-embeddings-for-visual-search-at-pinterest-74ea7ea103f0)
- [x] [Representation learning](https://en.wikipedia.org/wiki/Feature_learning)
- [x] [ResNet paper](https://arxiv.org/abs/1512.03385)
- [x] [Transformer paper - Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [x] [Vision transformer paper](https://arxiv.org/abs/2010.11929)
- [x] [SimCLR - A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- [ ] [MoCo paper](https://arxiv.org/pdf/1911.05722.pdf) 
- [x] [Constractive Representation Learning Methods](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)
- [x] [Dot product](https://en.wikipedia.org/wiki/Dot_product)
- [x] [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [x] [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance)
- [x] [Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)
- [x] [Curse of dimensionality issue in ML](https://www.mygreatlearning.com/blog/understanding-curse-of-dimensionality/)
- [x] [Cross-entropy loss](https://en.wikipedia.org/wiki/Cross-entropy)
- [ ] [Vector quantization](http://www.ws.binghamton.edu/fowler/fowler%20personal%20page/EE523_files/Ch_10_1%20VQ%20Description%20(PPT).pdf)
- [ ] [Product quantization](https://towardsdatascience.com/product-quantization-for-similarity-search-2f1f67c5fddd)
- [ ] [R-Trees](https://en.wikipedia.org/wiki/R-tree)
- [x] [KD-Trees](https://kanoki.org/2020/08/05/find-nearest-neighbor-using-kd-tree/)
- [ ] [Annoy](https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6)
- [ ] [Locality-sensitive hashing](http://web.stanford.edu/class/cs246/slides/03-lsh.pdf)
- [ ] [Fais library](https://github.com/facebookresearch/faiss)
- [ ] [ScaNN library](https://github.com/google-research/google-research/tree/master/scann)
- [x] [Cotent moderation with ML](https://appen.com/blog/content-moderation/)
- [x] [Bias in AI and recommendation systems](https://www.searchenginejournal.com/biases-search-recommender-systems/339319/#close)
- [x] [Positional bias](https://eugeneyan.com/writing/position-bias/)
- [x] [Smart crop](https://blog.twitter.com/engineering/en_us/topics/infrastructure/2018/Smart-Auto-Cropping-of-Images)
- [ ] [Better search with gnns](https://arxiv.org/pdf/2010.01666.pdf)
- [x] [Active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning))
- [ ] [Human-in-the-loop ML](https://arxiv.org/pdf/2108.00941.pdf)

