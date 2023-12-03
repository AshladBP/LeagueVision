import os
import sys
from PIL import Image, ImageEnhance, ImageOps
import random
import calendar
import time

def process_image(file_path, output_path):
    with Image.open(file_path) as img:
        # Resize image
        size = random.randint(100, 300)
        img = img.resize((size, size), Image.Resampling.LANCZOS)

        # Rotate image
        img = img.rotate(random.randint(-30, 30))

        # Translation
        translate_x = random.randint(-10, 10)
        translate_y = random.randint(-10, 10)
        new_img = Image.new('RGB', img.size, (255, 255, 255))
        new_img.paste(img, (translate_x, translate_y))

        # Rescaling (Zooming)
        scale = random.uniform(0.8, 1.2)
        new_img = new_img.resize((int(size * scale), int(size * scale)), Image.Resampling.LANCZOS)

        # Horizontal and/or Vertical Flipping
        if random.choice([True, False]):
            new_img = ImageOps.mirror(new_img)
        if random.choice([True, False]):
            new_img = ImageOps.flip(new_img)

        # Shearing
        shear_factor = random.uniform(-0.3, 0.3)
        new_img = new_img.transform(new_img.size, Image.AFFINE, (1, shear_factor, 0, 0, 1, 0))

        # Adjust Brightness
        enhancer = ImageEnhance.Brightness(new_img)
        new_img = enhancer.enhance(random.uniform(0.2, 1.5))

        # Adjust Contrast
        enhancer = ImageEnhance.Contrast(new_img)
        new_img = enhancer.enhance(random.uniform(0.5, 1.5))

        # Save image with altered quality
        quality = random.randint(20, 95)
        if random.choice([True, False]):
            width, height = new_img.size

            # Ensure the crop dimensions are valid
            min_crop_size = min(width, height) // 4  # At least 1/4th of the image size
            max_left_top = min_crop_size  # Maximum value for left and top to ensure valid crop

            left = random.randint(0, max_left_top)
            top = random.randint(0, max_left_top)
            right = random.randint(left + min_crop_size, width)
            bottom = random.randint(top + min_crop_size, height)

            new_img = new_img.crop((left, top, right, bottom))

        new_img.save(output_path, quality=quality)



if len(sys.argv) != 4 or sys.argv[1] == "help":
    print('Usage: python3 imageModifier.py image_directory output_directory amount_of_run')
    print("ARG 1 : image_directory")
    print("ARG 2 : output_directory")
    print("ARG 3 : amount_of_run")
    exit()

# Directory containing images
image_directory = sys.argv[1]
output_directory = sys.argv[2]
# Make sure output directory exists
os.makedirs(output_directory, exist_ok=True)

for i in range(0, int(sys.argv[3])):
    print(i)
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(image_directory, filename)
            ts = calendar.timegm(time.gmtime())
            output_path = os.path.join(output_directory, str(ts) + str(i) + filename)
            print(str(ts) + filename)
            process_image(file_path, output_path)
