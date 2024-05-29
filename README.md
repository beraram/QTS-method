# Code for paper "Unsupervised Drift Detection Using Quadtree Spatial Mapping"

This code allows researchers to replicate the experiments

## Abstract
Learning from a data stream is a challenging task due to the uncertainty about the data probability distribution, which may change over time. This phenomenon is called concept drift in the data mining literature. Some drift detection methods often assume that the labels become available immediately after the data arrives. However, in real-world applications, this is may be unrealistic. In this paper, we propose an unsupervised and model-independent drift detector based on quadtree spatial analysis (QTS). Essentially, we mapped the feature space using a quadtree and and monitored variables that mimic the data stream spatial behavior. Drifts are detected when the current spatial mapping significantly changes. The proposed method underwent evaluation for both synthetic and real-world datasets and was compared to other state-of-the-art unsupervised drift detectors. 
