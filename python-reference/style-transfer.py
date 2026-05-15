print("Initializing...")
from tqdm.auto import tqdm
import os
with tqdm(total=6, desc="Importing other modules") as pbar:
    from utils.image import image_loader, imshow; pbar.update(1)
    from utils.vgg_layers import VggUtil; pbar.update(1)
    import torch; pbar.update(1)
    import torch.optim as optim; pbar.update(1)
    from datetime import datetime; pbar.update(1)
    import matplotlib.pyplot as plt; pbar.update(1)

# Set device
device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device_str)
print(f"Using {device_str} as torch device.")
# Init Utils
vgg_util = VggUtil(device)

# ====================
# === Run Settings ===
# ====================
style_img_name = 'starry_night_768x970'
# style_img_name = 'femme-nue-assise_1251x960'
# style_img_name = 'composition-vii_850x1280'

content_img_name = 'madeira_128x128'

expt_name = 'style-resolutions'
n_steps_arr = [300]
content_weights = [1]

style_weights = [500000] # Good for composition / starry night + madeira
# style_weights = [2000000] # Good for nue + madeira

resolutions = [(128, 192)]
# resolutions = [(192, 128)] # good for nue + madeira
# resolutions = [(256, 384)] # good for composition + madeira

# show_plots = False
show_plots = True
# =======================
# === End of Settings ===
# =======================

def run_image_style_transfer(content_img, style_img,
                             n_steps, style_weight, content_weight,
                             show_plots, output_title, 
                             output_filename_all, output_filename_one):
    # === Model ===
    vgg_util.init_style_transfer_model(content_img=content_img, 
                                       style_img=style_img)
    
    # === Target Image ===
    output_img = content_img.clone()
    # output_img = torch.randn(content_img.data.size())
    output_img.requires_grad_(True)
    
    # === Optimizer ===
    optimizer = optim.LBFGS([output_img])
    
    # === Run Optimization ===
    vgg_util.run_style_transfer(input_img=output_img, optimizer=optimizer,
                                n_steps=n_steps, 
                                style_weight=style_weight, 
                                content_weight=content_weight)
    # === Preview images ===
    fig, axs = plt.subplots(1, 3, figsize=[15, 10])
    imshow(style_img, axs[0], title='Style Image')
    imshow(content_img, axs[1], title='Content Image')
    imshow(output_img, axs[2], title='Output Image')
    fig.savefig(output_filename_all)
    if not show_plots: plt.close()
    
    # === Preview output image bigger ===
    fig, axs = plt.subplots(1, 1, figsize=[5, 5])
    imshow(output_img, axs, title=output_title)
    fig.savefig(output_filename_one)
    if not show_plots: plt.close()

# ===============
# Actual run loop
# ===============
os.makedirs("expt", exist_ok=True)
folder_name = f"expt/{datetime.today().strftime('%y%m%d')}_{style_img_name}_{content_img_name}_{expt_name}"
os.makedirs(folder_name, exist_ok=True)
for n_steps in n_steps_arr:
    for style_weight in style_weights:
        for content_weight in content_weights:
            for resolution in resolutions:
                content_img = image_loader(f"./assets/{content_img_name}.jpg", device)
                style_img = image_loader(f"./assets/{style_img_name}.jpg", device, resolution)
            
                run_image_style_transfer(content_img, style_img, 
                                         n_steps, style_weight, content_weight,
                                         show_plots,
                                         f'Style {resolution[0]}x{resolution[1]}, Content 128x128',
                                         f'./{folder_name}/compare_{n_steps}_{style_weight}_{content_weight}_{resolution[0]}x{resolution[1]}.png',
                                         f'./{folder_name}/{n_steps}_{style_weight}_{content_weight}_{resolution[0]}x{resolution[1]}.png') 
