# ECE661_Final

## Important Files Including:
- `ResNetPytorch.py`, `DataManagerPytorch.py`, `AttackWrappersWhiteBoxP.py`: sourced from the paper's Githubt repository: https://github.com/MetaMain/ViTRobust/tree/main
- `ResNet56_WhiteBox.ipynb`: notebook for training ResNet56 model, generating adversarial attacks using ResNet56, predicting adversarial attacks using ResNet56.
  
## Description
With the growing sophistication of neural network models, particularly attention-based networks like Vision Transformers (ViTs), the need to evaluate and improve their robustness against adversarial attacks has become essential. While Vision Transformers have demonstrated state-of-the-art performance in image classification, surpassing or matching traditional Convolutional Neural Networks (CNNs), their vulnerability to adversarial examples remains underexplored compared to CNNs, which have been extensively studied in this regard. This project aims to systematically investigate the robustness of Vision Transformers, CNNs, and hybrid model architectures against a variety of adversarial attack methods, providing a comparative analysis across different models and datasets.

## Hypothesis 
Null Hypothesis: There is no significant difference between the effect of generated adversarial examples across models (ViTs vs CNNs). 
Alternative Hypothesis: There is significant difference between the effect of generated adversarial examples across models (ViTs vs CNNs). 

## Dataset
CIFAR-10: In this project, we will use the CIFAR-10 dataset to evaluate and compare the robustness of various models, including Vision Transformers, CNNs, and ensemble architectures, against a range of adversarial attacks. This dataset, consisting of 60,000 32 x 32 color images across 10 classes, provides a well-established benchmark for testing model performance under adversarial conditions. 
