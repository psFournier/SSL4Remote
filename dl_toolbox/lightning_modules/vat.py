import torch
import torch.nn as nn


class VAT(torch.nn.Module):
    """A class to implement Virtual Adversarial Training."""


    def __init__(self, model, xi, eps, num_power_iter):

        super().__init__()

        self.model = model
        self.xi = xi
        self.eps = eps
        self.num_power_iter = num_power_iter
        self.kl_div = nn.KLDivLoss(reduction='batchmean')


    def forward(self, x, y):
        
        x = x.detach().clone()
        y = y.detach().clone()

        # r_max is estimated by computing dKL(f(x)|f(x+r))/dr (see VAT paper)
        d = torch.normal(torch.zeros(x.shape), torch.ones(x.shape)).to(x.device)
        d = d / torch.norm(d, dim=(2, 3), keepdim=True)

        for _ in range(self.num_power_iter):

            d.requires_grad = True
            y_hat = self.model(x + self.xi * d)
            log_p_y_hat = y_hat.log_softmax(dim=1)

            adv_dist = self.kl_div(log_p_y_hat, y)
            adv_dist.backward()

            d = d.grad.detach() 
            d = d / torch.norm(d, dim=(2, 3), keepdim=True)

            self.model.zero_grad()


        # Calculate the Local Distributional Smoothness
        r_adv = d * self.eps
        x_adv = x + r_adv.detach()

        return x_adv

