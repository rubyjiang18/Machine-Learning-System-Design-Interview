## Chapter 11: People You May Know

### 0. Clarification
- help users discover potential connections and grow their network
- location, edu bkg, work experience, exisiting connections, prev interactions
- friendship is symmetrical
- social graph of most users is not very dynamic, meaning their connections do not change significantly over a short time

### 1. Frame as ML task
1.1 Define the ML objective: maximize the num of formed connections between users

1.2 Specify input and output: user => a list of connections renked by relevance to the user

1.3 Choose the right ML category:
- Pointwise Learning to Rank (LTR) to rank connections by relevance
    - cons: inputs are two distinct users, does not consider avialble **social context**, less accurate.
- GNN edge-level prediction
    - consider one hop or two-hop neighborhood
    - We use a model that takes the entire social graph as input, and predicts the prob of an edge existing between two specific nodes
    - rank by prob

### 2. Data Preparation
2.1 Data Engineering
- Users

| User ID | School | Degree | Major | Start Date | End data |

Challenge is that a specific attribute can be represented in different forms. So it is important to standardize the raw data.
    - force user to choose from a predefined list
    - use heuristics to group diff representation of an attribute
    - use ML-based methods such as clustering or LM to group similar attributes

- Connections

| User ID 1 | User ID 2 | Timestamp when the connection was formed |

- Interactions
    - send a connection request
    - accept a request
    - follow another user
    - view a profile
    - like or react to a post
    
| User ID | Interaction type | Interaction value | Timestamp |

2.2 Feature Engineering

- User features:
    - demographic: age, gender, city, country
    - num of connections, followers, followings, and pending requests
    - account's age
    - number of received reactions over a certain period, like one week (signal of how active this user is)

- **User-user affinities**
    - edu and work affinity:
        - schools in common
        - overlapping years at school
        - same major
        - number of companies in common
        - same industry

    - social affinity:
        - num of profile visits
        - num of connections in common
        - **time discounted mutual connections**

### 3. Model Development
3.1 Model selection: GNNs
- nodes store user information
- edge store user-user characteristics/affinities
- GNNs produces node embedding for each node
- similarity is calculated using 2 user/node embeddings

3.2 Model training
- construct the dataset
    - a snap shot of social graph at time t and time t+1
    - initialize node features and edge features of the graph
    - create label based on social graph on t+1, any not formed connections are treated as negative
    - loss function: out of this book's scope

### 4. Evaluation
4.1 Offline metrics:
- binary clssification (AUC-ROC)
- mAP  ????

4.2 Online metrics:
- total num of connection requests sent in the last X days
- total num of requests accepted in the last X days

### 5. Serving
5.1 Efficiency
- 1B users on the platform, to decrease candidate, use **FoF** (1k * 1k) and **precompute** PYMK.
- online prediction: only generate on-the-fly for active users, and not inactive users
- batch prediction for all users and store in a DB + extra computation (better)

### Reference
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