## Chapter 4: Youtube Video Search

### 0. Clarification
- A search system for videos
- Input is text query only, output is a list of relevant videos
- We use both the video's visual content and textual data
- As opposed to recommendation system, no personalization is required
- Given 10M <video, text query> pairs for training

### 1. Frame as ML task
1.1 Define the ML objective: retrieve and rank videos based on relevance to the text query
1.2 Specify input and output
1.3 Choose the right ML category
Basically do text search and visual search seperately then fuse the results.
- Visual search: rank based on similarity => Representation learning
    - Two seperate encoders for text query and video
    - Calc similarity score using two embed then rank
- Text search: videos with the most similar titles, descriptions, or tags to the text query are shown as output
    - Inverted index is a common tech for creating the text-based search component. 
    - Not based on ML, no training cost
    - Elasticsearch

### 2. Data Preparation
2.1 Data Engineering

Not much engineering: Videl name/index | query | split type

2.2 Feature Engineering

- Prepare text data: 
    - Text normalization or clean up (lowercase, remove punctutation, trim whitespaces, decompose combined graphemes into a combination of simple ones, strip accents, lemmatization and stemming)
    - Tokenization: breaking text into smaller units/tokens (word, subword, character tokenization)
    - Token to IDs: 
        - lookup table: 1:1 mapping maps token to ID
        - hashing (feature hashing or hashing trick): memory efficient method that use a hash function to obtain IDs, without keeping a lookup table. Still maps tokens to ID.
        - pros and cons of these 2 methods
- Prepare video data
    - Decode frame => sample frames => resize => scaling/normalization/correct colors => frames.npy

### 3. Model Development
3.1 Model selection
- Text encoder
    - statistical methods
        - BoW: convert a sentence into a fixed length vector, and models sentence-word occurance.
            - Pros: Fast
            - Cons: Does not consider order of words, does not capture semantic and contextual meaning of sentence, representation vector is sparse
        - TF-IDF: reflect how important a word is to a document in a collection or corpus. It create the same sentence-word matrix as BoW, but normalize the matrix by the freq of words.

        ```math
        tfidf(t, d, D) = tf(t, d) * idf(t, D)
        ```

            - Pros: better than BoW as it gives less weights to freq words.
            - Cons: an extra normalization step, no order of words, no semantic, spars
    - ML based methods: solves the above issues.
        - Embedding (lookup) layer: map each ID to an embed vector using loopup table. **This is a very effective and simple solution to convert sparse feature, such as IDs, to  a fixed-sized embed.**
        - Word2Vec: shallow NN and use the co-occurences of words in a local context to learn word embed. CBOW and Skip-gram.
        - Transformer-based architectures: considers the context of the words in a sentence when converting to embed. Diff from Word2Vec, it produces diff embed for the same word depending on the context. BERT, GPT, BLOOM.
- Video encoder
    - Video level models: process a whole video and create embed. Use 3d convolutions or Transformers. Comp Expensive.
    - Frame level models
        - preprocess a video and sample frames => Run model on sampled frames to create embed => Aggregate (avg) to generate video embed (ViT)
        - Pros: Faster and comp less expensive.
        - Cons: do no understand temporal aspect of the videos, such as actions or motions, but not as crucial.

3.2 Model training: constrastive learning approach
- For each input video, we create 1 positive text query, (n-1) negative text queries
- Encode each video and n text queries
- Compute similarity scores (S1 = Ev * E1, ..., Sn = Ev * En)
- Softmax to sum to 1
- Cross Entropy for loss

### 4. Evaluation
4.1 Offline metric
- MRR
- Recall@k
- Precision@k and mAP
- nDCG

4.2 Online metric
- CTR
- Video completion rate
- Total watch time of search results by week, month, year

### 5. Serving
5.1 Prediction pipeline
- visual search => hundres of videos
    - Use NN or ANN to find the most similar video embed to the text embed
- textual search => hundres of videos
    - Use elastic search to find videos with **titles or tags overlap with text query**
- fusing layer => tens of videos
    - one option is to jut re-rank based on weighted sum of their predicted relevance scores
    - another is to adopt an additional model to re-rank the videos
- re-ranking service => search results
    - incorporating business-level logic and policies

5.2 Indexing pipeline
- Video indexing pipeline: encoder create embed for each video and then indexed for NN.
- Text indexing pipeline: elasticsearch for indexing titles, tags, auto-tags.

### 6. Other talking points
6.1 Multi-stage design
6.2 Use more video features, popularity, freshness
6.3 Instead of using annotated dataset, use interactions - continual learning
6.4 Use ML to find titles and tags which are semantically similar to query text

### Reference
- [ ] [Elasticsearch](https://www.tutorialspoint.com/elasticsearch/elasticsearch_query_dsl.htm)
- [x] [Preprocessing text data](https://huggingface.co/docs/transformers/preprocessing)
- [x] [NFKD normalization](http://unicode.org/reports/tr15/)
- [x] [What is Tokenization summary](https://huggingface.co/docs/transformers/tokenizer_summary)
- [x] [Hash collision](https://en.wikipedia.org/wiki/Hash_collision)
- [ ] [Deep Learning for NLP](http://cs224d.stanford.edu/lecture_notes/notes1.pdf)
- [x] [TI-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [x] [Word2Vec models](https://www.tensorflow.org/text/tutorials/word2vec)
- [ ] [Continuous bag of words](https://www.kdnuggets.com/2018/04/implementing-deep-learning-methods-feature-engineering-text-data-cbow.html)
- [ ] [Skip-gram model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
- [x] [BERT model](https://arxiv.org/abs/1810.04805)
- [ ] [GPT3 model](https://arxiv.org/abs/2005.14165)
- [x] [BLOOM model](https://bigscience.huggingface.co/blog/bloom)
- [x] [Transformer implementation from scratch](https://peterbloem.nl/blog/transformers)
- [x] [3D convolutions](https://www.kaggle.com/code/shivamb/3d-convolutions-understanding-use-case/notebook)
- [ ] [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [x] [Query understanding for search engines](https://www.linkedin.com/pulse/ai-query-understanding-daniel-tunkelang/)
- [ ] [Multimodel video representation learning](https://arxiv.org/abs/2012.04124)
- [ ] [Multimodel language models](https://arxiv.org/abs/2107.00676)
- [ ] [Near-duplicate video detection](https://arxiv.org/abs/2005.07356)
- [ ] [Generalizable search relevence](https://livebook.manning.com/book/ai-powered-search/chapter-10/v-10/1)
- [x] [**Freshness in search and recommendation systems**](https://developers.google.com/machine-learning/recommendation/dnn/re-ranking)
- [ ] [Semantic product search by Amazon](https://arxiv.org/pdf/1907.00937.pdf)
- [ ] [Ranking relevance in Yahoo search](https://www.kdd.org/kdd2016/papers/files/adf0361-yinA.pdf)
- [ ] [Semantic product search in E-Commerce](https://arxiv.org/abs/2008.08180)