
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

from gan_utils_copy import downsample_block, upsample_block, discriminator_block
from data_utils import plot_sample_images, batch_generator, get_samples, imread

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8,8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

plt.rcParams.update(params)

IMG_WIDTH = 128
IMG_HEIGHT = 128

#U-Net Generator-------------------
def build_generator(img_shape, channels=3, num_filters=32):
    # Image input
    input_layer = Input(shape=img_shape)
    down_sample_1 = downsample_block(input_layer, num_filters)
    down_sample_2 = downsample_block(down_sample_1, num_filters*2)
    down_sample_3 = downsample_block(down_sample_2,num_filters*4)
    down_sample_4 = downsample_block(down_sample_3,num_filters*8)
    upsample_1 = upsample_block(down_sample_4, down_sample_3, num_filters*4)
    upsample_2 = upsample_block(upsample_1, down_sample_2, num_filters*2)
    upsample_3 = upsample_block(upsample_2, down_sample_1, num_filters)
    upsample_4 = UpSampling2D(size=2)(upsample_3)
    output_img = Conv2D(channels, 
                        kernel_size=4,
                        strides=1, 
                        padding='same', 
                        activation='tanh')(upsample_4)
    return Model(input_layer, output_img)

#Discriminator------------
def build_discriminator(img_shape,num_filters=64):
    input_layer = Input(shape=img_shape)
    disc_block_1 = discriminator_block(input_layer, 
                                       num_filters, 
                                       instance_normalization=False)
    disc_block_2 = discriminator_block(disc_block_1, num_filters*2)
    disc_block_3 = discriminator_block(disc_block_2, num_filters*4)
    disc_block_4 = discriminator_block(disc_block_3, num_filters*8)
    output = Conv2D(1, kernel_size=4, strides=1, padding='same')(disc_block_4)
    return Model(input_layer, output)


#Training Loop
def train(gen_AB, 
          gen_BA, 
          disc_A, 
          disc_B, 
          gan, 
          patch_gan_shape, 
          epochs, 
          path,
          batch_size=1, 
          sample_interval=50):
    # Adversarial loss ground truths
    real_y = np.ones((batch_size,) + patch_gan_shape)
    fake_y = np.zeros((batch_size,) + patch_gan_shape)
    
    for epoch in range(epochs):
        print("Epoch={}".format(epoch))
        for idx, (imgs_A, imgs_B) in enumerate(batch_generator(path,
                                                               batch_size,
                                                               image_res=[IMG_HEIGHT, IMG_WIDTH])):
            # train discriminators
            # # generate fake samples from both generators
            fake_B = gen_AB.predict([imgs_A])
            fake_A = gen_BA.predict([imgs_B])
            # Train the discriminators (original images = real / translated = Fake)
            disc_A_loss_real = disc_A.train_on_batch([imgs_A], real_y)
            disc_A_loss_fake = disc_A.train_on_batch([fake_A], fake_y)
            disc_A_loss = 0.5 * np.add(disc_A_loss_real, disc_A_loss_fake)
            disc_B_loss_real = disc_B.train_on_batch([imgs_B], real_y)
            disc_B_loss_fake = disc_B.train_on_batch([fake_B], fake_y)
            disc_B_loss = 0.5 * np.add(disc_B_loss_real, disc_B_loss_fake)
            # Total disciminator loss
            discriminator_loss = 0.5 * np.add(disc_A_loss, disc_B_loss)
            # train generator
            gen_loss = gan.train_on_batch([imgs_A, imgs_B],
                                          [real_y, real_y,
                                           imgs_A, imgs_B,
                                           imgs_A, imgs_B])
            # training updates every 50 iterations
            if idx % 50 == 0:
                print ("[Epoch {}/{}] [Discriminator loss: {}, accuracy: {}][Generator loss: {}, Adversarial Loss: {}, Reconstruction Loss: {}, Identity Loss: {}]"
                       .format(idx,
                               epoch,
                               discriminator_loss[0],
                               100*discriminator_loss[1],
                               gen_loss[0],
                               np.mean(gen_loss[1:3]),
                               np.mean(gen_loss[3:5]),
                               np.mean(gen_loss[5:6])))
# GAN config---------
generator_filters = 32
discriminator_filters = 64

# input shape
channels = 3
input_shape = (IMG_HEIGHT, IMG_WIDTH, channels)

# Loss weights
lambda_cycle = 10.0            
lambda_identity = 0.1 * lambda_cycle

optimizer = Adam(0.0002, 0.5)

# prepare patch size for our setup
patch = int(IMG_HEIGHT / 2**4)
patch_gan_shape = (patch, patch, 1)
print("Patch Shape={}".format(patch_gan_shape))


#build Discriminators----------
disc_A = build_discriminator(input_shape,discriminator_filters)
disc_A.compile(loss='mse',
    optimizer=optimizer,
    metrics=['accuracy'])

disc_B = build_discriminator(input_shape,discriminator_filters)
disc_B.compile(loss='mse',
    optimizer=optimizer,
    metrics=['accuracy'])

#Generators and GAN Model Objects----------
gen_AB = build_generator(input_shape,channels, generator_filters)
gen_BA = build_generator(input_shape, channels, generator_filters)

img_A = Input(shape=input_shape)
img_B = Input(shape=input_shape)

# generate fake samples from both generators
fake_B = gen_AB(img_A)
fake_A = gen_BA(img_B)

# reconstruct orginal samples from both generators
reconstruct_A = gen_BA(fake_B)
reconstruct_B = gen_AB(fake_A)

# generate identity samples
identity_A = gen_BA(img_A)
identity_B = gen_AB(img_B)

# disable discriminator training
disc_A.trainable = False
disc_B.trainable = False

# use discriminator to classify real vs fake
output_A = disc_A(fake_A)
output_B = disc_B(fake_B)

# Combined model trains generators to fool discriminators
gan = Model(inputs=[img_A, img_B],
            outputs=[output_A, output_B,
                     reconstruct_A, reconstruct_B,
                     identity_A, identity_B ])
gan.compile(loss=['mse', 'mse','mae', 'mae','mae', 'mae'],
            loss_weights=[1, 1,
                          lambda_cycle, lambda_cycle,
                          lambda_identity, lambda_identity ],
            optimizer=optimizer)

#train
train(gen_AB,
      gen_BA,
      disc_A,
      disc_B,
      gan,
      patch_gan_shape,
      epochs=20,
      batch_size=1,
      sample_interval=500,
      path="/Users/Documents/LaptopFiles/Projects/image_library/Car_Images")


for idx, (imgs_A, imgs_B) in enumerate(batch_generator(path, batch_size,image_res=[IMG_HEIGHT, IMG_WIDTH])):
            if idx <= 1:
                fake_B = gen_AB.predict([imgs_A])
                fake_A = gen_BA.predict([imgs_B])
                reconstruct_A = gen_BA.predict(fake_B)
                reconstruct_B = gen_AB.predict(fake_A)
                gen_imgs = np.concatenate([imgs_A, fake_B,
                           reconstruct_A,
                           imgs_B, fake_A,
                           reconstruct_B])
                # scale images 0 - 1
                gen_imgs = 0.5 * gen_imgs + 0.5
                titles = ['Original', 'Translated', 'Reconstructed']
                r, c = 2, 3
                fig, axs = plt.subplots(r, c)
                cnt = 0
                for i in range(r):
                     for j in range(c):
                          axs[i,j].imshow(gen_imgs[cnt])
                          axs[i, j].set_title(titles[j])
                          axs[i,j].axis('off')
                          cnt += 1
                plt.show()
                plt.close()




# save model -----------------
# Save model weights-----------------------------
gan.save_weights('/Users/Documents/LaptopFiles/Projects/GenImages/cycleGANmodel.weights.h5')
# to load the model weights, compile the model first, then load the weights
# Rebuild the model with the same architecture

# Save model
gan.save('/Users/Documents/LaptopFiles/Projects/GenImages/cycleGANmodel.keras')

# Load saved weights and models
gan = tf.keras.models.load_model('/Users/Documents/LaptopFiles/Projects/GenImages/cycleGANmodel.keras')
gan.summary()
