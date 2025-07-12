# first pip install rembg   onnxruntime
from rembg import remove
from PIL import Image
import io

# Paths
input_path = "C:/Users/abhin/Downloads/sticker_happy.png"
output_path = 'output_image2.png'

# Read and remove background
with open(input_path, 'rb') as input_file:
    input_data = input_file.read()

output_data = remove(input_data)

# Save the output image
output_image = Image.open(io.BytesIO(output_data))
output_image.save(output_path)

print("✅ Background removed and saved successfully!")