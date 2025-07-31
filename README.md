# gen_images_CycleGAN
Implementation of a CycleGAN, where 2 generators and 2 discriminators are created. Two images are fed into two generators whose outputs are used to train a discriminator. 

CycleGANs take in unpaired inputs and are setup using 2 generators and 2 discriminators.
Generators G and F generate X -> Y and Y -> X 
Discriminators F and D are used to regenerate the original input pred_x and pred_y.
Adversarial loss needs to be adapted for 2 generators and 2 discriminators: the L1 loss function is adaped for 2 generators
    -consist of cycle loss and identity loss
    -identity loss adjusts the introduction of tints, an unwanted behavior
    - overall loss is a weighted sum of the different losses
