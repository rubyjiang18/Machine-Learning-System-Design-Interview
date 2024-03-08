## Chapter 3: Google Stree View Blurring System
### 0. Clarifying requirements
- A street view blurring system that blurs license plates and human faces. 
- We have a training dataset of 1M images with annotated faces and plates
- Latency is not an issue
- Business goal is to protect user privacy
add 
### 1. Frame as ML task
- Define the ML objective: Protect user privacy by blurring plates and faces. So ML goal is to identify these "objects" accurately. 
- Specify system's input (image) and output (location and class) 
- Choose the right ML category
    
    An object detection has 2 responsibilities:
    - predict the location of each object => (x,y,w,h) regression problem
    - predict the class of each bounding box
    
    Two common object detection architectures
    - Two stage networks (RCNN etc)
        - Regional proposal network (RPN)
        - Classifier      
    - One stage network (YOLO, SSD)
        make prediction wrt anchors or grid of possible obj centers

Two-stage network is slower but more accurate, we only have 1 M data not huge by modern standards, so we use two-stage.

### 2. Data Preparation
2.1 Data engineering
- Annotated dataset
- Street View images

2.2 Feature engineering

Standard preprocessing (resize + normalization) + data augmentation (random crop, random saturation, flip, rotation, translation, affine transformation, changing brightness, saturation, and contrast)

Offline augmentation: requires additional data storage, but fast

Online augmentation: slow but no extra storage

### 3. Model Development
3.1 Model selection - Two stage network

- Convolution layers => feature map
- RPN: takes in feature map, generate candidate regions.
- Classifier: takes feature map + proposed candidate regions.

3.2 Model training

Forward propagation, loss calculation, backward propagation.

- Regression loss for generating bounding boxes => MSE
```math
L_{reg} = \frac{1}{M}\sum_{i=1}^{M}[(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (w_i - \hat{w}_i)^2 + (h_i - \hat{h}_i)^2] 
```

- Classification loss => log loss = cross entropy loss
```math
L_{cls} = \frac{1}{M}\sum_{i=1}^{M}\sum_{c=1}^{C}y_c log (\hat{y}_c)
```
Eventually 
```math
L = L_{cls} + \lambda L_{reg} 
```

### 4. Evaluation
When is the predicted bounding box considered correct? An IOU threshold is usually used.
```math
IOU = \frac{\text{Area of overlap}}{\text{Area of union}}
```

4.1 Offline metric
- Precision
```math
Precision = \frac{\text{Num of correct predictions}}{\text{Total num of detection}}
```
Primary disadvantage is that precision varies  with different IOU thresholds.
- Average precision (AP)
```math
AP = \int_0^1 P(r)dr
```
where P(r) is the precision at IOU threshold r.

This can be approximated as 
```math
AP = \frac{1}{11}\sum_{n=0}^{n=10}P(n)
```
- Mean average precision (mAP)
Average AP over all different class
```math
mAP = \frac{1}{C}\sum_{C=1}^{C}AP_c
```

4.1 Online metric
- number of user reports and complaints
- human annotatos to spot-check the percentage of incorrectly blurred images
- measure bias and fairness is also important (faces of diff ages and races)


### 5. Serving
Common problems: overlapping bounding boxes. Solution: Non-maximum suppression (NMS), which keeps highly confident ones.

ML system design
- batch prediction pipeline
- Data pipeline + hard negative mining


### Reference
- [x] [Google Street View](https://www.google.com/streetview/)
- [x] [DETR](https://github.com/facebookresearch/detr)
- [x] [RCNN family](https://lilianweng.github.io/posts/2017-12-31-object-recognition-part-3/)
- [x] [Fast RCNN](https://arxiv.org/abs/1504.08083)
- [x] [Faster RCNN](https://arxiv.org/abs/1506.01497)
- [ ] [YOLO family](https://pyimagesearch.com/2022/04/04/introduction-to-the-yolo-family/)
- [ ] [SSD](https://jonathan-hui.medium.com/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06)
- [x] [Data augmentation techiniques](https://www.kaggle.com/discussions/getting-started/190280)
- [x] [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)
- [ ] [Object detection details](https://dudeperf3ct.github.io/object/detection/2019/01/07/Mystery-of-Object-Detection/)
- [ ] [Forward pass and backward pass - Andrew Ng Youtube Video](https://www.youtube.com/watch?v=qzPQ8cEsVK8)
- [x] [MSE](https://en.wikipedia.org/wiki/Mean_squared_error)
- [x] [Log loss/cross entropy loss](https://en.wikipedia.org/wiki/Cross-entropy)
- [x] [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/index.html)
- [x] [Coco dataset evaluation](https://cocodataset.org/#detection-eval)
- [ ] [Object detection evaluation](https://github.com/rafaelpadilla/Object-Detection-Metrics)
- [ ] [Non-maximum suppression (NMS)](https://en.wikipedia.org/wiki/NMS)
- [x] [Python implementation of NMS](https://learnopencv.com/non-maximum-suppression-theory-and-implementation-in-pytorch/)
- [x] [Recent object detection models](https://viso.ai/deep-learning/object-detection/)
- [x] [Distributed training in TensorFlow](https://www.tensorflow.org/guide/distributed_training)
- [x] [Distributed training in Pytorch](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [x] [GDPR and ML](https://www.oreilly.com/radar/how-will-the-gdpr-impact-machine-learning/)
- [ ] [Bias and fairness in face detectin](http://sibgrapi.sid.inpe.br/col/sid.inpe.br/sibgrapi/2021/09.04.19.00/doc/103.pdf)
- [x] [AI fairness](https://www.kaggle.com/code/alexisbcook/ai-fairness)
- [x] [Continue learing](https://towardsdatascience.com/how-to-apply-continual-learning-to-your-machine-learning-models-4754adcd7f7f)
- [x] [Active learning](https://en.wikipedia.org/wiki/Active_learning_(machine_learning))
- [x] [Human-in-the-loop ML](https://arxiv.org/pdf/2108.00941.pdf)
