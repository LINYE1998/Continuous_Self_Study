# Scene Graph Benchmark in Pytorch

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/LINYE1998/Continuous_Self_Study/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

Our paper Continuous Self-Study: Scene Graph Generation
with Self-Knowledge Distillation and Spatial
Augmentation has been accepted by ACCV 2022.

## Contents

1. [Overview](#Overview)
2. [Install the Requirements](INSTALL.md)
3. [Training SGG with CSS](#perform-training-on-scene-graph-generation)


## Overview
As an extension of visual detection tasks, scene graph generation (SGG) has drawn increasing attention with the achievement of
complex image understanding. However, it still faces two challenges: one
is the distinguishing of objects with high visual similarity, the other is
the discriminating of relationships with long-tailed bias. In this paper, we
propose a Continuous Self-Study model (CSS) with self-knowledge distillation and spatial augmentation to refine the detection of hard samples.
We design a long-term memory structure for CSS to learn its own behavior with the context feature, which can perceive the hard sample of
itself and focus more on similar targets in different scenes. Meanwhile,
a fine-grained relative position encoding method is adopted to augment
spatial features and supplement relationship information. On the Visual
Genome benchmark, experiments show that the proposed CSS achieves
obvious improvements over the previous state-of-the-art methods.

## Installation
Check [README_TDE.md](README_TDE.md) for installation instructions.

## Training SGG with CSS
We abstract the ```spatial augmentation``` module in the file ```roi_heads/relation_head/saptial_augmentation.py```, which are independent of the Faster R-CNN backbone and relation-head feature extractor.
 
If you want to train SGG model with ```spatial augmentation```, you can set the ```self.spatial_on``` in ```maskrcnn_benchmark.modeling.roi_heads.relation_head.roi_relation_predictors.CausalAnalysisPredictor``` into ```True```. And if you want to train SGG model with ```self-study```, you can set the ```self_study_on``` in ```maskrcnn_benchmark.modeling.roi_heads.relation_head.roi_relation_predictors.CausalAnalysisPredictor``` into ```True```

