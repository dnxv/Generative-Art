# How to use

```
Run the python notebook (.ipynb) on Google Colab 
- Connect to accelerator
- Select "Runtime" > "Run all" 
```

# Tests and their output

```
Details:Type: WGAN-GP
2nd attempt
Epochs: 300
Latent Dimension: 100
First Layer of Generator: 1024
Augmentation: None
Original Dataset: 300
```
![](./GIF-outputs/WGAN-GP/generated-art.png)

# Architecture/Design
![](./GIF-outputs/WGAN-GP-Training-Overview.png)

```
  Generator(
  (generator): Sequential(
    (0): Sequential(
      (0): ConvTranspose2d(100, 1024, kernel_size=(4, 4), stride=(1, 1), bias=False)
      (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (1): Sequential(
      (0): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (2): Sequential(
      (0): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (3): Sequential(
      (0): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
    (4): ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (5): Tanh()
  )
)
```

```
Discriminator(
  (discriminator): Sequential(
    (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2)
    (2): Sequential(
      (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      (1): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (2): LeakyReLU(negative_slope=0.2)
    )
    (3): Sequential(
      (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      (1): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (2): LeakyReLU(negative_slope=0.2)
    )
    (4): Sequential(
      (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
      (1): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
      (2): LeakyReLU(negative_slope=0.2)
    )
    (5): Conv2d(512, 1, kernel_size=(4, 4), stride=(2, 2))
  )
)
```

# Error Detection
```
###############################
###      Mode collapse      ###
###############################
```
![](./GIF-outputs/DCGAN/image_at_epoch_180.png)

```
####################################
###  How to avoid Mode collapse  ###
####################################

#dont make the model more complex
#instead try to make it less complex

#WGAN uses different loss function
#WGAN-GP introduces a gradient penalty
  - this is to prevent discriminator to update weights 
    not too quickly and not too slowly
    (vanishing/exploding gradients)
```
```
##############################
###    Overly-Augmented    ###
##############################
```
![](./GIF-outputs/augemented-image.png)
```
#image transformations
    #avoid: 
    # rotations, random erase, random crops, five-crop

    #do the following instead:
    # Normalization: Scaling pixel values to a range (e.g., [0, 1]).
    # Resizing: Adjusting the image size if needed. (e.g., to 64x64)
```
# Gradient Penalty
![](./GIF-outputs/gradient-penalty.png)
![](./GIF-outputs/visualized-gradients.png)