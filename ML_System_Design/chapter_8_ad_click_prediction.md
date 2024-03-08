## Chapter 8: Ad Click Prediction on Social Platforms

### 0. Clarification
- business goal to maximize revenue
- every click generates same revenue
- ads can be displayed to the same user multiple times without fatigue period
- user can hide ads
- user-ad interaction for dataset
    - postive training data via user clicks
    - negative trainig data
        - assume any impression as negative until a click is observed
        - add not clicked after the ad been visible on user's screen for a certain duration
        - negtive feedback
- continual learning is necessary

### 1. Frame as ML task
1.1 Define the ML objective: maximize revenue by predicting if an ad will be clicked
1.2 Specify input and output: user => a ranked list of ads based on click prob
1.3 Choose the right ML category: pointwise LTR with <user, ad> pairs.

### 2. Data Preparation
2.1 Data Engineering
- Users

- Ads

| Ad ID | Advertiser ID | Ad group ID| Campaign ID | Category | Subcategory | Image or videos |

- User-ad interactions

| User ID | Ad ID | Interaction type | Dwell time | Location | Timestamp |

2.2 Feature Engineering

- Ad features
    - ID (ad id, advertiser id, campaign id, group id) => embedding layer
    - Image/video => SimCLR
    - category/subcategory => normalization, tokenization, token to ID (chapter 4) or LM
    - impression and num of clicks
        - total impression/click on the add
        - total impression/click on ads supplied by an advertiser
        - total impression/click of the campaign


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
