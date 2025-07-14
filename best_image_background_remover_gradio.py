import gradio as gr
from gradio_imageslider import ImageSlider
from loadimg import load_img
# import spaces
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms

torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
birefnet.to(device)
transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# @spaces.GPU
def fn(image):
    im = load_img(image, output_type="pil")
    im = im.convert("RGB")
    image_size = im.size
    origin = im.copy()
    image = load_img(im)
    input_images = transform_image(image).unsqueeze(0).to(device)
    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    image.putalpha(mask)
    # return (image, origin)
    image.save("img.png","PNG")
    return (image , "img.png")


img1 = gr.Image(type= "pil", image_mode="RGBA")
image = gr.Image(label="Upload an image")
file = gr.File()


chameleon = load_img("chameleon.jpg", output_type="pil")

url = "https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg"
demo = gr.Interface(
    fn, inputs=image, outputs=[img1,file], examples=[chameleon], api_name="image"
)

# tab2 = gr.Interface(fn, inputs=text, outputs=slider2, examples=[url], api_name="text")


# demo = gr.TabbedInterface(
#     [tab1, tab2], ["image", "text"], title="birefnet for background removal (WIP 🛠️, works for linux)"
# )

if __name__ == "__main__":
    demo.launch()

# requirments.txt
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




