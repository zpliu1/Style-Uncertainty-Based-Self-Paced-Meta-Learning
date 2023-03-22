import random
from contextlib import contextmanager
import torch
import torch.nn as nn


def deactivate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(False)


def activate_mixstyle(m):
    if type(m) == MixStyle:
        m.set_activation_status(True)


def random_mixstyle(m):
    if type(m) == MixStyle:
        m.update_mix_method('random')


def crossdomain_mixstyle(m):
    if type(m) == MixStyle:
        m.update_mix_method('crossdomain')


@contextmanager
def run_without_mixstyle(model):
    # Assume MixStyle was initially activated
    try:
        model.apply(deactivate_mixstyle)
        yield
    finally:
        model.apply(activate_mixstyle)


@contextmanager
def run_with_mixstyle(model, mix=None):
    # Assume MixStyle was initially deactivated
    if mix == 'random':
        model.apply(random_mixstyle)

    elif mix == 'crossdomain':
        model.apply(crossdomain_mixstyle)

    try:
        model.apply(activate_mixstyle)
        yield
    finally:
        model.apply(deactivate_mixstyle)


class MixStyle(nn.Module):
    """MixStyle.

    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x, indd = 0):
        if not self.training or not self._activated:
            return x

        '''
        if random.random() > self.p:
            return x
        '''

        B = x.size(0)

        '''
        soft_max = soft_value.max(dim=1)[0]
        print('*****')
        print('soft:', soft_max)
        '''

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        #mu3 = torch.randn(mu.size()).cuda()
        #sig3 = torch.randn(sig.size()).cuda()


        mu2, sig2 = mu[perm], sig[perm]
        #mu_mix = mu*lmda + mu3 * (1-lmda)
        #sig_mix = sig*lmda + sig3 * (1-lmda)
        var_rand = random.uniform(0, 0.1)
        #print('a:', var_rand)

        #var_rand = round(var_rand, 3)
        #print(var_rand)

        while var_rand==0.0:
            var_rand = random.uniform(0, 0.1)




        if indd==1:
            mu3 = torch.normal(0, 0.1, mu.size(), requires_grad=False).cuda()  # ([22, 256, 1, 1])
            sig3 = torch.normal(0, 0.1, sig.size(), requires_grad=False).cuda()
        elif indd==2:
            mu3 = torch.normal(0, 0.15, mu.size(), requires_grad=False).cuda()  # ([22, 256, 1, 1])
            sig3 = torch.normal(0, 0.15, sig.size(), requires_grad=False).cuda()
        #elif indd==3:
        #    mu3 = torch.normal(0, var_rand, mu.size(), requires_grad=False).cuda()  # ([22, 256, 1, 1])
        #    sig3 = torch.normal(0, var_rand, sig.size(), requires_grad=False).cuda()
        else:
            print('*****error*****')


        #mu3 = torch.normal(0, 0.1, mu.size(), requires_grad=False).cuda()       #([22, 256, 1, 1])
        #sig3 = torch.normal(0, 0.1, sig.size(), requires_grad=False).cuda()

        #mu4 = torch.normal(0, 0.2, mu.size(), requires_grad=False).cuda()
        #sig4 = torch.normal(0, 0.2, sig.size(), requires_grad=False).cuda()



        '''
        soft_max = soft_value.max(dim=1)[0].unsqueeze(dim=1).unsqueeze(dim=1).unsqueeze(dim=1)
        thre_value = torch.zeros(soft_max.size()) + 0.9
        thre_value = thre_value.cuda()
        
        #cod_a = torch.zeros(soft_max.size()).cuda()
        #cod_b = torch.ones(soft_max.size()).cuda()

        mu_se = torch.where(soft_max > thre_value, mu4, mu3)
        sig_se = torch.where(soft_max > thre_value, sig4, sig3)
        '''
        '''
        soft_max = soft_value.max(dim=1)[0].mean()
        if soft_max>0.9:
            mu_se = mu4
            sig_se = sig4
        else:
            mu_se = mu3
            sig_se = sig3
        '''

        '''
        if random.random() > self.p:
            mu_mix = mu + mu3
            sig_mix = sig + sig3
        else:
            mu_mix = mu + mu4
            sig_mix = sig + sig4
        '''
        mu_mix = mu + mu3
        sig_mix = sig + sig3


        return x_normed*sig_mix + mu_mix
