import os
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image
from loadimg import load_img  # Ensure this file exists and works

# Set PyTorch precision
torch.set_float32_matmul_precision("high")

# Load model
birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
birefnet.to(device)

# Transformations
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Hardcoded folders
INPUT_FOLDER = "C:/Users/abhin/OneDrive/Documents"
OUTPUT_FOLDER = "C:/Users/abhin/OneDrive/Documents"

# Create output folder if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def remove_background(image_path, output_path):
    im = load_img(image_path, output_type="pil").convert("RGB")
    image_size = im.size
    image = load_img(im)
    input_images = transform_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    im.putalpha(mask)

    im.save(output_path, "PNG")

# Process all images
if __name__ == "__main__":
    supported_formats = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(supported_formats):
            input_path = os.path.join(INPUT_FOLDER, filename)
            output_filename = os.path.splitext(filename)[0] + "_transparent.png"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            try:
                remove_background(input_path, output_path)
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# requirements.txt
# torch
# accelerate
# opencv-python
# spaces
# pillow
# numpy
# timm
# kornia
# prettytable
# typing
# scikit-image
# huggingface_hub
# transformers>=4.39.1
# gradio
# gradio_imageslider
# loadimg>=0.1.1
# einops