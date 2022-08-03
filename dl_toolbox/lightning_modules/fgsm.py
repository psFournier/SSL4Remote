import torch


class FGSM(torch.nn.Module):
    """A class to implement FGSM attack."""


    def __init__(self, model, loss_fn, epsilon):

        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.loss_fn = loss_fn


    def forward(self, x, target):
        
        # Set requires_grad to True
        x = x.detach().clone()
        x_adv = x.requires_grad_(True)
        target = target.detach().clone()

        output = self.model(x_adv)

        loss = self.loss_fn(output, target) 
        loss.backward()

        # Calculate adversarial sample
        grad_sign = x_adv.grad.detach().sign()
        x_adv = x + self.epsilon * grad_sign

        # Set the gradients to zero so that they do not accumulate for the
        # pl.module backward
        self.model.zero_grad()

        return x_adv.detach()

