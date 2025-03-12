from PIL import Image, ImageDraw

# Create a new image with a white background
size = (256, 256)
image = Image.new('RGBA', size, (255, 255, 255, 0))
draw = ImageDraw.Draw(image)

# Draw a simple "D" letter
draw.ellipse([(20, 20), (236, 236)], fill=(0, 120, 212, 255))
draw.ellipse([(60, 60), (196, 196)], fill=(255, 255, 255, 0))

# Save as ICO
image.save('icon.ico', format='ICO') 