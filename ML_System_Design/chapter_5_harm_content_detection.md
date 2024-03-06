## Chapter 5: Harmful Content Detection

### 0. Clarification
- Harmful content: posts with violence, adult content, self-harm, drug, hate-speech, etc.
- Bad actors/inauthentic behaviors: face accounts, spam, phishing, organized unethical activities, unsafe behaviors

We design a system to 
- proactively monitors new posts 
- detects harmful contents
- remove or demotes them if they violates the platform's guidelines
- informs users why the post was identified as harmful
- a post can be text, images, videos, or any combinations, and can be in diff languages
- user can report harmful posts

### 1. Frame as ML task
1.1 Define the ML objective: accurately predict harmful posts => safer platform

1.2 Specify input and output: post as input, output prob of being harmful. To combine heterogeneous data:
- Late fusion
    - pros: train, eval, test each model independently
    - cons: seperte training data for each modality, cannot capture the combined harmful
- Early fusion
    - pros: one training dataset, one model to maintain, and considers the combined effect of diff modalities
    - cons: learning is more diffcult as the model need to learn the complex relationship between modality, which requires sufficient training data.

we use early fusion.

1.3 Choose the right ML category
- single binary classifier
- one binary classifier per harmful class
- multi-label classifier
- multi-task classifier (we use this one)
    - learn multiple task simultaneously
    - this allows the model to learn similarities between tasks
    - shared layers + task-specific layer (classification head)
    - pros: easy to train and maintain, shared layer transforms the features in a way benef for all tasks, one training dataset

### 2. Data Preparation
2.1 Data Engineering
- Users

| User ID | Username | Age | Gender | City | Country | Email |

- Posts

| Post ID | Author ID | Author IP | Timestamp | Texual content | Image or Video | Links |

- User-post interactions

|User ID | Post ID | Interaction type | Interaction value | Timestamp |

2.2 Feature Engineering
We explore predicitive features that can be derived from a post
- textual content
    - text prep (normalization, tokenization)
    - vectorization 
        - statisticlal 
        - ML methods: BERT, DistilmBERT
- image or video
    - preprocess: decode, resize, normalize
    - feature extraction: CLIP's visual encoder, SimCLR, VideoMoCo
- user interaction to the post
    - comments can be v helpful (embed each comment then aggregate)
    - number of reactions: likes, shares, comments, reports
- author features
    - Author's violation history: number of violations, number of user reports, profane words rate
    - Author's demographics: age (bucket), gender (one-hot), city and country (embedding layer not one-hot)
    - account information: number of follower and followee, acount age
- contextual information:
    - time of day (bucket and one-hot)
    - device (one-hot)

### 3. Model Development
3.1 Model selection
Neural networks, but when choose a NN, what factors to consider? Hyperparameters: num of layers, activation functions, lr, etc => grid search.

3.2 Model training
- Construct the dataset

We construct inputs (features) offline in batches and compute fused features, and stored in feature store. To create labels:
    - hand labeling (more accurate, for eval data)
    - natural labeling (noisy, quick, large, for train)

- Choose loss function

Cross entropy (-log(pt)) for each task. Then sum up as overall loss.

A common challenge in training multimodel systems is overfitting. When learning speed varies across diff modalities, one modality such as image may dominate the learning process.
- Gradient blending
- Focal loss (-(1-p_t)^lambda * log(p_t)): setting $\lambda$ > 0 reduces the relative loss on well-classified examples (p_t>0.5), putting more focuse on hard, misclassified examples.


### 4. Evaluation
4.1 Offline metric
- Precision, recall, F1
- ROC-AUC, PR-AUC

4.2 Online metric
- Prevalence
```math
Prevalence = \frac{\text{num of harmful posts we did not prevent}}{\text{total num of posts on the platform}}
```
Shortcoming is that it treats all harm posts equally, for example, harm post with 100K views and 10 views equally.
- Harmful impressions
- Valid appeals
```math
Appeals = \frac{\text{num of reversed appeals}}{\text{num of harm posts detected by the system}}
```
- Proactive rate: perc of harm posts found and deleted by the system before user report it
```math
Appeals = \frac{\text{num of harm posts detected by the system}}{\text{num of harm posts detected by the system + reported by users}}
```
- User reports per harmful class

### 5. Serving
- harm content detection service
- violation enforcement service (take down posts with high confidence)
- demoting service (temp demote posts with low confidence)

### 6. Other talking points

### Reference
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
