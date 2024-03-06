# Machine Learning System Design Interview Reading List
Welcome to the [Machine Learning System Design Interview](https://www.amazon.com/Machine-Learning-System-Design-Interview/dp/1736049127/ref=sr_1_1?crid=1N7OYRAM2K046&keywords=machine+learning+system+design+interview&qid=1698602213&sprefix=%2Caps%2C173&sr=8-1) Reading List! This curated list is from the Reference Materials at each chapter of the above listed book.

Another good MLSD resources can be found in this [repo](https://github.com/alirezadir/Machine-Learning-Interviews). Be sure to check it out.

## Chapter 1: Introduction and Overview

## Chapter 2: Visual Search System

## Chapter 3: Google Stree View Blurring System

## Chapter 4: Youtube Video Search
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
- [x] [Freshness in search and recommendation systems](https://developers.google.com/machine-learning/recommendation/dnn/re-ranking)
- [ ] [Semantic product search by Amazon](https://arxiv.org/pdf/1907.00937.pdf)
- [ ] [Ranking relevance in Yahoo search](https://www.kdd.org/kdd2016/papers/files/adf0361-yinA.pdf)
- [ ] [Semantic product search in E-Commerce](https://arxiv.org/abs/2008.08180)

## Chapter 5: Harmful Content Detection
- [x] [Facebook's inauthentic behavior](https://transparency.fb.com/policies/community-standards/inauthentic-behavior/)
- [x] [LinkedIn's professional community policies](https://www.linkedin.com/legal/professional-community-policies)
- [x] [Twitter's civic integrity policy](https://help.twitter.com/en/rules-and-policies/election-integrity-policy)
- [x] [Facebooks integrity survey](https://arxiv.org/abs/2009.10311)
- [x] [Pinterest's violation detection system](https://medium.com/pinterest-engineering/how-pinterest-fights-misinformation-hate-speech-and-self-harm-content-with-machine-learning-1806b73b40ef)
- [x] [Abusive deteciton at LinkedIn-isolation forests (unsupervised)](https://engineering.linkedin.com/blog/2019/isolation-forest)
- [x] [WPIE (Whole Post Integrity Embeddings) method](https://ai.meta.com/blog/community-standards-report/)
- [x] [BERT (Bidirectional Encoder Representations from Transformers-pretraining for language understanding) paper](https://arxiv.org/abs/1810.04805)
- [x] [Multilingual DistilBert](https://huggingface.co/distilbert-base-multilingual-cased)
- [ ] [Multilingual language models](https://arxiv.org/abs/2107.00676)
- [x] [CLIP model](https://openai.com/research/clip)
- [x] [SimCLR paper](https://arxiv.org/abs/2002.05709)
- [ ] [VideoMoCo paper](https://arxiv.org/abs/2103.05905)
- [x] [Hyperparameter tunning](https://cloud.google.com/ai-platform/training/docs/hyperparameter-tuning-overview)
- [x] [Overfitting](https://en.wikipedia.org/wiki/Overfitting)
- [x] [Focal loss](https://amaarora.github.io/posts/2020-06-29-FocalLoss.html)
- [ ] [Graidient blending in multimodel systems](https://arxiv.org/abs/1905.12681)
- [x] [ROC curves vs precision-recall curve](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)
- [x] [Introduced bias by human labeling](https://labelyourdata.com/articles/bias-in-machine-learning)
- [ ] [Facebook's approach to quickly tackling trending harmful content](https://ai.meta.com/blog/harmful-content-can-evolve-quickly-our-new-ai-system-adapts-to-tackle-it/)
- [ ] [Facebook's TIES approach](https://arxiv.org/abs/2002.07917)
- [ ] [Temporal interaction embedding](https://www.facebook.com/atscaleevents/videos/730968530723238)
- [ ] [Building and scaling human review system](https://www.facebook.com/atscaleevents/videos/1201751883328695)
- [ ] Abusive account detection framework
- [x] [Borderline contents](https://transparency.fb.com/features/approach-to-ranking/content-distribution-guidelines/content-likely-violating-our-community-standards)
- [x] [Efficient harmful content detection - Few-Shot Learner](https://about.fb.com/news/2021/12/metas-new-ai-system-tackles-harmful-content/)
- [ ] [Linear Transformer paper](https://arxiv.org/abs/2006.16236)
- [ ] [Efficient AI models to detect hate speech](https://ai.meta.com/blog/how-facebook-uses-super-efficient-ai-models-to-detect-hate-speech/)

## Chapter 6: Video Recommendation System
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

## Chapter 7: Event Recommendation System
- [x] [Data leakage](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [x] [Online training frequency](https://huyenchip.com/2022/01/02/real-time-machine-learning-challenges-and-solutions.html#towards-continual-learning)

## Chapter 8: Ad Click Prediction on Social Platforms
- [ ] [Addressing delayed feedback](https://arxiv.org/pdf/1907.06558.pdf)
- [ ] [AdTech basics](https://advertising.amazon.com/library/guides/what-is-adtech)
- [x] [SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- [x] [Feature crossing](https://developers.google.com/machine-learning/crash-course/feature-crosses/video-lecture#:~:text=Build%20an%20understanding%20of%20feature%20crosses.,1%20Implement%20feature%20crosses%20in%20TensorFlow.)
- [x] [Feature Generation with Gradient Boosted Decision Trees](https://towardsdatascience.com/feature-generation-with-gradient-boosted-decision-trees-21d4946d6ab5)
- [ ] DCN paper
- [ ] DCN V2 paper
- [ ] Microsoft's deep crossing entwork paper
- [ ] Factorization Machines
- [ ] Deep Factorization Machines
- [x] [Kaggle's winning solution in ad click prediction](https://www.youtube.com/watch?v=4Go5crRVyuU)
- [x] [Data leakage](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [x] [Time-based dataset splitting](https://www.linkedin.com/pulse/time-based-splitting-determining-train-test-data-come-manraj-chalokia/)
- [ ] [Modal calibration](https://machinelearningmastery.com/calibrated-classification-model-in-scikit-learn/)
- [ ] [Field-aware Factorization Machines](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)
- [ ] [catastrophic forgetting in continual learning](https://www.cs.uic.edu/~liub/lifelong-learning/continual-learning.pdf)


## Chapter 9: Similar Listings on Vacation Rental Platforms

## Chapter 10: Personalized News Feed
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


## Chapter 11: People You May Know
- [x] [Clustering in ML](https://developers.google.com/machine-learning/clustering/overview)
- [x] [PYMK on Facebook](https://www.youtube.com/watch?v=Xpx5RYNTQvg&t=1823s)
- [ ] [Graph convolutional neural netwoeks](https://tkipf.github.io/graph-convolutional-networks/)
- [ ] [GraphSage paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf)
- [ ] [Graph attention networks](https://arxiv.org/abs/1710.10903)
- [ ] [Graph isomorphism network](https://arxiv.org/abs/1810.00826)
- [ ] [Graph neural networks](https://distill.pub/2021/gnn-intro/)
- [ ] [Personalized random walk](https://www.youtube.com/watch?v=HbzQzUaJ_9I)
- [x] [LinkedIn's PYMK system](https://engineering.linkedin.com/blog/2021/optimizing-pymk-for-equity-in-network-creation)
- [ ] [Addressing delayed feedback](https://arxiv.org/pdf/1907.06558.pdf)