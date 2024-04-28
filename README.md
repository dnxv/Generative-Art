# How to use

```
Run the python notebook (.ipynb) on Google Colab 
- Connect to accelerator
- Select "Runtime" > "Run all" 
```

# Tests and their output

```python client.py```
```
Example: x.x.x.x:12000
```

# Error Detection
```
###############################
###      Mode collapse      ###
###############################

#(done, saved mixed results) train for 100 epochs

#didnt work
#make model less complex
#make model more complex

# (didnt work) kernel size to 3,3 and removed 512 layer from discriminator
# (kinda better) change only kernel size, keep 512 layer

#(nope) kern 2,2

#adding gaussian noise
#(discriminator has it, generator shouldnt) applying dropout

#image transformations
    #augment for more diversity
    # Normalization: Scaling pixel values to a range (e.g., [0, 1]).
    # Resizing: Adjusting the image size if needed.
    # Data Augmentation: variations of the images (e.g., rotations, flips) to improve model robustness.
    # """
    # from PIL import Image
    # import torchvision.transforms as transforms

    # transforms_ = [
    #     transforms.Resize(int(img_height*1.12), Image.BICUBIC),
    #     transforms.RandomCrop((img_height, img_width)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ]
    # """

#vanishing gradient
    #Lipschitz constraint
#lack of convergence
#loss function
  # discriminator: binary_crossentropy

#minibatch discrimination

#WGAN
#transfer learning

# """
# https://spotintelligence.com/2023/10/11/mode-collapse-in-gans-explained-how-to-detect-it-practical-solutions/#1_Insufficient_Model_Capacity
# """
```

# Architecture/Design
```
def method(arg1, arg2, arg3):
  """
  description
  
  Returns:
    None
  Example:
    >>> mesg, addr, port, seqRange = 'Howdy', localhost, 12000, 10
    >>> send(mesg, addr, port, seqRange)
  """