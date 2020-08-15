# EdgeTpu SSDLite Mobiledet Transfer Learning Turotial
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![made-with-coral](https://img.shields.io/badge/Made%20with-Coral-orange)](https://coral.ai/)


This repo contains the Colab notebook for my ssdlite mobiledet model retrain tutorial as well as the minimal example code to deploys the model on the [EdgeTpu](https://coral.ai).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Namburger/edgetpu-ssdlite-mobiledet-retrain/blob/master/ssdlite_mobiledet_transfer_learning_cat_vs_dog.ipynb)

[<img dth="777" src="https://github.com/Namburger/edgetpu-ssdlite-mobiledet-retrain/blob/master/assets/eval.png">]()
Actual eval after about 3k steps

## Quick run:
```
$ python3 run_model.py models/ssdlite_mobiledet_dog_vs_cat_edgetpu.tflite test_images
Evaluating: test_images/image1.jpg
Evaluating: test_images/image2.jpg
Evaluating: test_images/image3.jpg
Evaluating: test_images/image4.jpg
Evaluating: test_images/image5.jpg
Evaluating: test_images/image6.jpg
Evaluating: test_images/image7.jpg
Evaluating: test_images/image8.jpg
Inference time:  0.06407481100177392
```

[<img dth="777" src="https://github.com/Namburger/edgetpu-ssdlite-mobiledet-retrain/blob/master/assets/inference.png">]()
Inference Results


## References:
* Yunyang Xiong, Hanxiao Liu, Suyog Gupta, Berkin Akin, Gabriel Bender, Pieter-Jan Kindermans, Mingxing Tan, Vikas Singh, Bo Chen:
[MobileDets: Searching for Object Detection Architectures for Mobile Accelerators.](https://arxiv.org/abs/2004.14525) CoRR abs/2004.14525 (2020)
