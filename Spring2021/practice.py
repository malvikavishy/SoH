from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageEnhance
import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np

image = Image.open("hand.jpg")
other_image = Image.open("dog.jpg")
newsize = (400, 400)
size = (900, 900)
angle1 = 40

#different changes to orientation of an image
image11 = image.transpose(Image.FLIP_TOP_BOTTOM) 
image12 = image11.resize(newsize, Image.BILINEAR)
image13 = image12.rotate(angle1)
image14 = image13.convert('L')

#putting text on an image
image2 = image.copy()
draw = ImageDraw.Draw(image2)
font = ImageFont.truetype("arial.ttf", 100)
draw.text((0, 0), "Testing", (255, 0, 0), font = font)

#putting watermark on an image
image3 = image.copy()
image3.thumbnail(size)
image31 = image.copy()
image31.paste(image3, (0, 0))

#changing properties of the image and combining it whith another picture
image4 = image.copy()
image41 = ImageEnhance.Contrast(image4).enhance(2.5)
image42 = ImageEnhance.Sharpness(image41).enhance(2.5)
image43 = other_image.resize(image42.size)
image44 = Image.blend(image42, image43, 0.5)

#transforming an image
image5 = image.copy()
image5 = image5.transform(image5.size, Image.AFFINE, (1, -0.5, 0.5 * image5.size[0], 0, 1, 0))

#splitting and switching channels of an image
image6 = image.copy()
r, g, b = image.split()
image61 = Image.merge("RGB", (b, g, r))

#picture outputs 
plt.figure(figsize = (10, 10))
plt.subplot(3, 3, 1)
plt.imshow(image14, cmap = 'gray')
plt.subplot(3, 3, 2)
plt.imshow(image2)
plt.subplot(3, 3, 3)
plt.imshow(image31)
plt.subplot(3, 3, 4)
plt.imshow(image44)
plt.subplot(3, 3, 5)
plt.imshow(image5)
plt.subplot(3, 3, 6)
plt.imshow(image61)
plt.show()