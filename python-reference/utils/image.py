import torch
import torchvision.transforms as transforms
from PIL import Image

imsize = 128

# Reconvert tensor into PIL image
unloader = transforms.ToPILImage()

"""Takes a filepath, returns the image as a torch tensor."""
def image_loader(image_name, device, imsize=(128, 128)):
    image = Image.open(image_name)
    
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])
    # batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

"""Displays the image on the specified plt subplot ax."""
def imshow(tensor, ax, bw=False, title=None, fontsize=None, vmin=0, vmax=1.0):
    if type(tensor) is torch.Tensor:
        image = tensor.cpu().clone()
    else:
        image = tensor
    # remove the batch dimension
    if image.shape[0] == 1:
        image = image.squeeze(0)
        
    if not bw: 
        image = unloader(image)
        
    if bw:
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    else:
        ax.imshow(image)
    if title is not None:
        ax.set_title(title, fontsize=fontsize or 12)
