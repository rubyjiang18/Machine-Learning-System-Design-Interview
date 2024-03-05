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