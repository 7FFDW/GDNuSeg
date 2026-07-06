
import torch.nn as nn
import torch.nn.functional as F

import torch



class DUU(nn.Module):
    def __init__(self, in_channels, cat='cat'):
        super(DUU, self).__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )


        self.inverse_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.way = cat



    def forward(self, feature1, glcm):

        if self.way == 'cat':


            feature1_weight = self.compute_uncertainty_weight(self.compute_dirichlet_alpha(feature1))
            feature2_weight = self.compute_uncertainty_weight(self.compute_dirichlet_alpha(glcm))
            x = torch.cat([feature1_weight * feature1, feature2_weight * glcm], dim=1)
            x = self.first_conv(x)

        else:
            feature1_weight = self.compute_uncertainty_weight(self.compute_dirichlet_alpha(feature1))
            feature2_weight = self.compute_uncertainty_weight(self.compute_dirichlet_alpha(glcm))
            x = feature1_weight * feature1 + feature2_weight * glcm


        residual = x



        forward_attention = self.se(x)
        inverse_attention = 1 - self.inverse_se(x)
        x = x * forward_attention + x * inverse_attention




        x = self.final_conv(x)

        x = x + residual

        return x

    def compute_dirichlet_alpha(self,output):

        alpha = F.softplus(output) + 1e-6
        return alpha

    def compute_uncertainty_weight(self, alpha):
        alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = 1.0 / (alpha_sum + 1e-6)
        weights = 1.0 - uncertainty
        weights = torch.clamp(weights, min=0.0, max=1.0)
        return weights











