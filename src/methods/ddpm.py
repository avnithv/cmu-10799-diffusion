"""
Denoising Diffusion Probabilistic Models (DDPM)
"""

import math
from typing import Dict, Tuple, Optional, Literal, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseMethod


class DDPM(BaseMethod):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_timesteps: int,
        beta_start: float,
        beta_end: float,
        # TODO: Add your own arguments here
    ):
        super().__init__(model, device)

        self.num_timesteps = int(num_timesteps)
        self. beta_start = beta_start
        self. beta_end = beta_end

        betas = torch.linspace(beta_start, beta_end, num_timesteps)      # (T,)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)                    # ᾱ_t, (T,)

        # register_buffer: not trainable, but moves with .to(device) and is saved in state_dict
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas", alphas.sqrt())
        self.register_buffer("sqrt_betas", betas.sqrt())
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt())
        # TODO: Implement your own init

    def _extract(self, coeffs, t, x):
        """
        coeffs: (T,) schedule tensor
        t:      (B,) integer timesteps
        x:      any tensor of shape (B, ...) — used only for its shape
        returns coeffs[t] reshaped to (B, 1, 1, ..., 1) so it broadcasts against x
        """
        out = coeffs[t]                                  # (B,)
        return out.view(-1, *([1] * (x.ndim - 1)))       # (B,1) for 1-D, (B,1,1,1) for images

    # =========================================================================
    # You can add, delete or modify as many functions as you would like
    # =========================================================================
    
    # Pro tips: If you have a lot of pseudo parameters that you will specify for each
    # model run but will be fixed once you specified them (say in your config),
    # then you can use super().register_buffer(...) for these parameters

    # Pro tips 2: If you need a specific broadcasting for your tensors,
    # it's a good idea to write a general helper function for that
    
    # =========================================================================
    # Forward process
    # =========================================================================


    def forward_process(self, x_0, t): # TODO: Add your own arguments here
        # TODO: Implement the forward (noise adding) process of DDPM
        noise = torch.randn_like(x_0) 
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_0)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0)
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    # =========================================================================
    # Training loss
    # =========================================================================

    def compute_loss(self, x_0: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        TODO: Implement your DDPM loss function here

        Args:
            x_0: Clean data samples of shape (batch_size, channels, height, width)
            **kwargs: Additional method-specific arguments
        
        Returns:
            loss: Scalar loss tensor for backpropagation
            metrics: Dictionary of metrics for logging (e.g., {'mse': 0.1})
        """

        B = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (B,), device=x_0.device)
        x_t, noise = self.forward_process(x_0, t)
        pred = self.model(x_t, t)
        loss = F.mse_loss(pred, noise)
        return loss, {'loss': loss.item()}
    
    # =========================================================================
    # Reverse process (sampling)
    # =========================================================================
    
    @torch.no_grad()
    def reverse_process(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        TODO: Implement one step of the DDPM reverse process

        Args:
            x_t: Noisy samples at time t (batch_size, channels, height, width)
            t: the time
            **kwargs: Additional method-specific arguments
        
        Returns:
            x_prev: Noisy samples at time t-1 (batch_size, channels, height, width)
        """
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t)
        sqrt_alphas = self._extract(self.sqrt_alphas, t, x_t)
        beta_t = self._extract(self.betas, t, x_t)
        sqrt_beta_t = self._extract(self.sqrt_betas, t, x_t)
        # per-sample mask: 1 where t > 0 (add noise), 0 where t == 0 (final step is deterministic)
        # reshape (B,) -> (B, 1, ..., 1) so it broadcasts against x_t for any data shape
        nonzero_mask = (t > 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        z = torch.randn_like(x_t) * nonzero_mask
        x_prev = 1.0 / sqrt_alphas * (x_t - (beta_t / sqrt_one_minus_alpha_bar) * self.model(x_t, t)) + sqrt_beta_t * z
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: Tuple[int, int, int],
        return_trajectory: bool = False,
        num_steps: int = None,
        # TODO: add your arguments here
        **kwargs
    ) -> torch.Tensor:
        """
        TODO: Implement DDPM sampling loop: start from pure noise, iterate through all the time steps using reverse_process()

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of each image (channels, height, width)
            **kwargs: Additional method-specific arguments (e.g., num_steps)
        
        Returns:
            samples: Generated samples of shape (batch_size, *image_shape)
        """
        self.eval_mode()
        x_t = torch.randn(batch_size, *image_shape, device=self.device)
        trajectory = [x_t] if return_trajectory else None

        if num_steps is None or num_steps == self.num_timesteps:
            for t in reversed(range(self.num_timesteps)):
                x_prev = self.reverse_process(x_t, torch.tensor([t] * batch_size, device=self.device))
                x_t = x_prev
                if return_trajectory:
                    trajectory.append(x_t)

        else:
            steps = torch.linspace(self.num_timesteps-1, 0, num_steps).round().long()

            for i, t in enumerate(steps):
                alpha_eff = self.alphas_cumprod[t]
                if i < len(steps) - 1:
                    alpha_eff = alpha_eff / self.alphas_cumprod[steps[i+1]]

                t_vec = torch.tensor([t] * batch_size, device=self.device)

                beta_eff = 1 - alpha_eff
                sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alphas_cumprod, t_vec, x_t)
                nonzero_mask = (t_vec > 0).float().view(-1, *([1] * (x_t.ndim - 1)))
                z = torch.randn_like(x_t) * nonzero_mask
                x_prev = 1.0 / torch.sqrt(alpha_eff) * (x_t - (beta_eff / sqrt_one_minus_alpha_bar) * self.model(x_t, t_vec)) + torch.sqrt(beta_eff) * z

                x_t = x_prev
                if return_trajectory:
                    trajectory.append(x_t)

        if return_trajectory:
            return x_t, torch.stack(trajectory)
        return x_t


    # =========================================================================
    # Device / state
    # =========================================================================

    def to(self, device: torch.device) -> "DDPM":
        super().to(device)
        self.device = device
        return self

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["num_timesteps"] = self.num_timesteps
        # TODO: add other things you want to save
        return state

    @classmethod
    def from_config(cls, model: nn.Module, config: dict, device: torch.device) -> "DDPM":
        ddpm_config = config.get("ddpm", config)
        return cls(
            model=model,
            device=device,
            num_timesteps=ddpm_config["num_timesteps"],
            beta_start=ddpm_config["beta_start"],
            beta_end=ddpm_config["beta_end"],
            # TODO: add your parameters here
        )
