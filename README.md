# Generative Adversarial Network for Images

This is an implementation of the CycleGAN model that was defined in <a href = https://arxiv.org/abs/1703.10593> Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks </a> and described by "Hands On Generative AI with Python and TensorFlow2" Joseph Babcock and Raghav Bali. 

The original code is available on <a href="https://github.com/PacktPublishing/Hands-On-Generative-AI-with-Python-and-TensorFlow-2.git"> Github </a>. The code found in this repo is adapted to use with the <a href="https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset?resource=download"> Stanford Cars Dataset </a>.

Implementation of a CycleGAN, where 2 generators and 2 discriminators are created. Two images are fed into two generators whose outputs are used to train a discriminator. 

CycleGANs take in unpaired inputs and are setup using 2 generators and 2 discriminators.
Generators G and F generate X -> Y and Y -> X 
Discriminators F and D are used to regenerate the original input pred_x and pred_y.
Adversarial loss needs to be adapted for 2 generators and 2 discriminators: the L1 loss function is adaped for 2 generators
    -consist of cycle loss and identity loss
    -identity loss adjusts the introduction of tints, an unwanted behavior
    - overall loss is a weighted sum of the different losses
