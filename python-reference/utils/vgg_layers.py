# TODO: Excessive imports, clean it up
import torch
import torch.nn as nn

from torchvision.models import vgg19, VGG19_Weights

from layers.normalization import cnn_normalization_mean, cnn_normalization_std, Normalization
from layers.content_loss import ContentLoss
from layers.style_loss import StyleLoss

# Heavily based on this tutorial:
#   https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html

class VggUtil():
    """Includes settings and utilities for using VGG."""
    def __init__(
        self,
        device: torch.device,
        content_layer_names = ['relu4_2'],
        style_layer_names = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'],
    ):
        print("[VggUtil] Initializing...")
        self.device = device
        self.content_layer_names = content_layer_names
        self.style_layer_names = style_layer_names
        
        print("[VggUtil] Loading VGG weights...")
        self.cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        self.vgg_mean = cnn_normalization_mean.detach().clone().to(device)
        self.vgg_std  = cnn_normalization_std.detach().clone().to(device)

        print("[VggUtil] Setup complete!")
        

    """Set up the model used to perform style transfer.
    content_image and style_image should have shape [1, 3 (RGB), H, W].
    """
    def init_style_transfer_model(self, content_img = None, style_img = None):
        print(f"[VggUtil] Setting up style transfer model with content image {content_img.shape if content_img is not None else '(None)'} and style image {style_img.shape if style_img is not None else '(None)'}...")
        
        self.style_model = nn.Sequential()
        # List form so it's easy to access individual layers directly.
        # E.g. preview conv_1 filter 30 with vgg_util.style_layers[1].weight[30]
        self.style_layers = []
        # Style and content loss layers
        self.style_s_loss_layers = []
        self.style_c_loss_layers = []
        # Store Conv2d weights for manual convolution
        # list of [n_filters, n_channels (RGB, 64, 128 etc), 3 (Y), 3 (X)]
        self.style_conv_weights = []
        # list of [n_filters]
        self.style_conv_biases = []
        
        # === Create/upload the model ===
        # Create an input normalization layer as first layer
        input_normalization_layer = Normalization(self.vgg_mean, self.vgg_std)
        self.style_model.add_module('norm0', input_normalization_layer)
        self.style_layers.append(input_normalization_layer)
        depth = 1 # increment every time we see a pooling layer
        i = 0  # increment every time we see a conv
        for idx, layer in enumerate(self.cnn.children()):
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv{depth}_{i}'
                self.style_conv_weights.append(layer.weight.detach().to(self.device))
                self.style_conv_biases.append(layer.bias.detach().to(self.device))
            elif isinstance(layer, nn.ReLU):
                name = f'relu{depth}_{i}'
                # The in-place version doesn't play very nicely with the ``ContentLoss``
                # and ``StyleLoss`` we insert below. So we replace with out-of-place
                # ones here.
                # layer = nn.ReLU(inplace=False)
                layer = nn.ReLU(inplace=True)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool{depth}'
                depth += 1
                i = 0
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn{depth}_{i}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            self.style_model.add_module(name, layer)
            self.style_layers.append(layer) 

            if name in self.content_layer_names or name in self.style_layer_names:
                self.style_model.to(self.device)
                
            if name in self.content_layer_names:
                # add content loss:
                target = self.style_model(content_img).detach() \
                         if content_img is not None else \
                         torch.zeros([1, 3, 16, 16], dtype=torch.float32).to(self.device)
                content_loss = ContentLoss(target)
                self.style_model.add_module(f"content_loss{depth}_{i}", content_loss)
                self.style_c_loss_layers.append(content_loss)
                
            if name in self.style_layer_names:
                # add style loss:
                target_feature = self.style_model(style_img).detach() \
                                 if style_img is not None else \
                                 torch.zeros([1, 3, 16, 16], dtype=torch.float32).to(self.device)
                style_loss = StyleLoss(target_feature)
                self.style_model.add_module(f"style_loss{depth}_{i}", style_loss)
                self.style_s_loss_layers.append(style_loss)
        print("[VggUtil] Layers processed...")
        
        # now we trim off the layers after the last content and style losses
        for i in range(len(self.style_model) - 1, -1, -1):
            if isinstance(self.style_model[i], ContentLoss) or isinstance(self.style_model[i], StyleLoss):
                break
        self.style_model = self.style_model[:(i + 1)]
        print("[VggUtil] Layers trimmed. Transferring to GPU...")
    
        self.style_model.eval()
        self.style_model.requires_grad_(False)
        self.style_model.to(self.device)
        
        print("[VggUtil] style_model:", self.style_model)

    """This image runs style transfer."""
    def run_style_transfer(self,
                           input_img,
                           n_steps, 
                           optimizer,
                           style_weight=1000, 
                           content_weight=1,
                           style_layer_weights=None):
        print('Optimizing...')
        run = [0]
        while run[0] <= n_steps:
            def closure():
                # correct the values of updated input image
                with torch.no_grad():
                    input_img.clamp_(0, 1)
        
                optimizer.zero_grad()
                self.style_model(input_img)
                style_score = 0
                content_score = 0
        
                for idx, sl in enumerate(self.style_s_loss_layers):
                    style_score += sl.loss if style_layer_weights is None else sl.loss * style_layer_weights[idx]
                for cl in self.style_c_loss_layers:
                    content_score += cl.loss
        
                style_score *= style_weight
                content_score *= content_weight
        
                loss = style_score + content_score
                loss.backward()
        
                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
        
                return style_score + content_score
        
            optimizer.step(closure)
        
        # a last correction...
        with torch.no_grad():
            input_img.clamp_(0, 1)
        