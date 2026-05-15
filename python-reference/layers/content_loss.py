# From https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        # Only compute content loss if the shape matches. We do this because
        # when we're initializing the model, we need to pass the style image
        # through the network. During the actual inference, this does not trigger.
        if input.shape != self.target.shape:
            return input
            
        self.loss = F.mse_loss(input, self.target)
        return input
