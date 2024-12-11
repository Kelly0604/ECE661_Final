# ECE661_Final

## Important Files Including:

- `Experiments` Folder: this folder contains the functions for building the 6 white-box attacks, black-box attacks (RayS) and transferability test. Functions are organized in their relative file to ensure more straightforward usage.
- `Experiments_inNotebook` Folder: this folder contains the same experiments on generating adversarial attacks, testing model's robust accuracy and attacks transferability. These experiments are done in 3 seperate notebooks mainly for each type of model. We started our experiments in these notebooks on Google Colab to utilize better computing resources. However, for your convenience in viewing the functions, please refer to the Experiments folder which contain similar information. 
  - `ResNetPytorch.py`, `DataManagerPytorch.py`, `AttackWrappersWhiteBoxP.py` in this folder are sourced from the paper  "On the Robustness of Vision Transformers to Adversarial Examples" Githubt repository: https://github.com/MetaMain/ViTRobust/tree/main
 
## How to use this repository
1. Clone the repository to your local machine. 
2. Choose which methods you would prefer: a). downloading the jupyter notebooks and running them on your google Colab b). Running `GenerateAttacks.py` and `TransferabilityTest.py` file which include the model evaluation and transferability tests. 
  
## Description
With the growing sophistication of neural network models, particularly attention-based networks like Vision Transformers (ViTs), the need to evaluate and improve their robustness against adversarial attacks has become essential. While Vision Transformers have demonstrated state-of-the-art performance in image classification, surpassing or matching traditional Convolutional Neural Networks (CNNs), their vulnerability to adversarial examples remains underexplored compared to CNNs, which have been extensively studied in this regard. This project aims to systematically investigate the robustness of Vision Transformers, CNNs, and hybrid model architectures against a variety of adversarial attack methods, providing a comparative analysis across different models and datasets.

## Hypothesis 
Null Hypothesis: There is no significant difference between the effect of generated adversarial examples across models (ViTs vs CNNs). 
Alternative Hypothesis: There is significant difference between the effect of generated adversarial examples across models (ViTs vs CNNs). 

## Dataset
CIFAR-10: In this project, we will use the CIFAR-10 dataset to evaluate and compare the robustness of various models, including Vision Transformers, CNNs, and ensemble architectures, against a range of adversarial attacks. This dataset, consisting of 60,000 32 x 32 color images across 10 classes, provides a well-established benchmark for testing model performance under adversarial conditions. 
