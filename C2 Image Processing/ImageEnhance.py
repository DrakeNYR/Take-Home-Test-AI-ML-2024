import cv2 as cv
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# sharpen image using kernel
def sharpen_image(image_path, output_path):
    with Image.open(image_path) as img:
        sharpened_img = img.filter(ImageFilter.Kernel(
            size=(3, 3),
            kernel=[-1, -1, -1,
                    -1, 16, -1,
                    -1, -1, -1],
            scale=None,
            offset=0
        ))
        
        # Save the sharpened image
        sharpened_img.save(output_path)

# adjust color channels seperately
def adjust_rgb_channels(image_path, output_path, r_factor=1.0, g_factor=1.0, b_factor=1.0):
    # Open an image file
    with Image.open(image_path) as img:
        # Split the image into RGB channels
        r, g, b = img.split()
        
        # Enhance each channel
        r = r.point(lambda p: min(255, int(p * r_factor)))
        g = g.point(lambda p: min(255, int(p * g_factor)))
        b = b.point(lambda p: min(255, int(p * b_factor)))
        
        # Merge channels back into an image
        img_adjusted = Image.merge("RGB", (r, g, b))
        
        # Save the modified image
        img_adjusted.save(output_path)

# add weight to image
def add_weight_to_image(image_path, output_path, weight_factor=0.5, blend_factor=0.3):
    # Open an image file
    with Image.open(image_path) as img:
        # Convert the image to grayscale to create a weight mask
        weight_mask = ImageOps.grayscale(img)
        
        # Enhance the weight mask with the weight factor
        enhancer = ImageEnhance.Brightness(weight_mask)
        weight_mask = enhancer.enhance(weight_factor)
        
        # Convert weight mask to RGB
        weight_mask = weight_mask.convert('RGB')
        
        # Convert the original image to RGB
        img = img.convert('RGB')
        
        # Blend the original image with the weight mask
        blended_image = Image.blend(img, weight_mask, alpha=blend_factor)
        
        # Enhance brightness
        brightness_enhancer = ImageEnhance.Brightness(blended_image)
        enhanced_image = brightness_enhancer.enhance(1.2)
        
        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = contrast_enhancer.enhance(1.05)

        # Save the modified image
        enhanced_image.save(output_path)


add_weight_to_image('ImageEnhancement/16.jpg', 'ImageEnhancement/output_image.jpg', weight_factor=0.5, blend_factor=-0.1)
adjust_rgb_channels('ImageEnhancement/output_image.jpg', 'ImageEnhancement/output_color_image.jpg', r_factor=1.1, g_factor=1.0, b_factor=1.0)
sharpen_image('ImageEnhancement/output_color_image.jpg', 'ImageEnhancement/final_image.jpg')

old = cv.imread("ImageEnhancement/16.jpg")
new = cv.imread("ImageEnhancement/final_image.jpg")
concat = np.concatenate((old, new), axis=1)

cv.imwrite("ImageEnhancement/compare.jpg", concat)
# cv.imshow("concat", concat)
