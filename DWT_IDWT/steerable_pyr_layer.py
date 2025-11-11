# steerable_pyr_layer.py
# Drop-in adapter that mimics the DWTForward/DWTInverse interface
# using a 1-level steerable pyramid with 3 orientations.
#
# Output packing matches WaveDiff's [LL, LH, HL, HH] per RGB channel:
#   LL := residual lowpass
#   (LH, HL, HH) := 3 oriented bandpass subbands at level 0
#
# Shapes returned to the training loop:
#   forward(x): (xll [B,3,H/2,W/2], [xh [B,3,3,H/2,W/2]])
#   inverse((xll, [xh])) -> x [B,3,H,W]
#
# Implementation uses pyrtools’ SteerablePyramidSpace (spatial) or
# SteerablePyramidFreq (Fourier). Both reconstruct via recon_pyr().
# Docs: https://pyrtools.readthedocs.io/ (SteerablePyramidSpace/Freq + recon_pyr)  # noqa

import torch
import torch.nn.functional as F
from typing import Tuple, List

# We prefer frequency version for exact recon; space is also OK.
from pyrtools.pyramids import SteerablePyramidFreq as SPyrF  # exact recon
# from pyrtools.pyramids import SteerablePyramidSpace as SPyrS  # alt option

class SPyrForward(torch.nn.Module):
    """
    Build a steerable pyramid for each image/channel, gather:
      lowpass residual (LL) + 3 oriented bands at level 0 (LH/HL/HH),
    then downsample by 2 to match WaveDiff’s (H/2, W/2).
    """
    def __init__(self, order: int = 2, height: int = 1):
        """
        order=2 -> 3 orientations (order+1); height=1 -> one scale.
        """
        super().__init__()
        self.order = order
        self.height = height

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x: [B,3,H,W] in [-1,1]
        Returns:
          xll: [B,3,H/2,W/2]
          xh:  list with one tensor [B,3,3,H/2,W/2]
               (same structure as pytorch_wavelets' DWTForward xh[0])
        """
        B, C, H, W = x.shape
        assert C == 3, "This adapter assumes RGB inputs."

        # storage
        ll_list = []
        hi_list = []  # will stack into [B,3,3,H/2,W/2]

        # iterate over batch and channel (pyrtools works per-image 2D)
        # We keep everything float32 on CPU; move back to x.device at the end.
        x_cpu = x.detach().to('cpu')

        for b in range(B):
            # Per-channel LL + 3 oriented bands at level 0
            ll_c = []
            hi_c = []
            for c in range(C):
                img = x_cpu[b, c].numpy()  # 2D numpy array

                # Build steerable pyramid (Fourier version gives exact recon). 
                # NB: SteerablePyramidFreq expects 2D array (H,W).
                pyr = SPyrF(img, height=self.height, order=self.order, is_complex=False)

                # Get coefficients as dict keyed by (level, band).
                # Residuals keys are 'residual_lowpass' (LP) and 'residual_highpass' (HP).
                coeffs = pyr.pyr_coeffs

                # Lowpass residual as our LL (shape ~ H/2^height, W/2^height)
                #ll = torch.from_numpy(coeffs[('residual_lowpass', 0)]).unsqueeze(0)  # [1,hL,wL]
                
                # coeffs is a dict from SteerablePyramidFreq.pyr_coeffs
                # Try multiple possible keys for the lowpass / residual lowpass
                if ('residual_lowpass', 0) in coeffs:
                    lp = coeffs[('residual_lowpass', 0)]
                elif ('residual_lowpass',) in coeffs:
                    lp = coeffs[('residual_lowpass',)]
                else:
                    # Fallback: use the lowest-scale bands as an approximate lowpass
                    # (this keeps things running even if pyrtools changed conventions)
                    low_keys = [k for k in coeffs.keys() if isinstance(k[0], int)]
                    if not low_keys:
                        raise KeyError(
                            "Could not find residual_lowpass in steerable pyramid coeffs; "
                            "available keys: {}".format(list(coeffs.keys())))
                     # take the max scale index as "coarsest"
                    max_scale = max(k[0] for k in low_keys)
                    # average over orientations at that coarsest scale
                    ori_keys = [k for k in low_keys if k[0] == max_scale]
                    lp = sum(coeffs[k] for k in ori_keys) / len(ori_keys)

                ll = torch.from_numpy(lp).unsqueeze(0).float()  # [1, H_lp, W_lp]

                # Level-0 oriented bands: indices 0..(num_orient-1); here 3 bands
                # Keep order: map to (LH, HL, HH) "slots"
                bands = []
                for k in range(self.order + 1):  # 3 orientations if order=2
                    bands.append(torch.from_numpy(coeffs[(0, k)]).unsqueeze(0).float())  # [1,hB,wB]
                hi = torch.stack(bands, dim=0)  # [3,1,hB,wB]

                # Downsample all to H/2, W/2 to match WaveDiff packing
                ll_ds = F.avg_pool2d(ll, kernel_size=2, stride=2).squeeze(0)     # [1,H/2,W/2]
                hi_ds = F.avg_pool2d(hi, kernel_size=2, stride=2)     # [3,1,H/2,W/2]

                ll_c.append(ll_ds)                    # list of [1,H/2,W/2]
                hi_c.append(hi_ds.squeeze(1))         # list of [3,H/2,W/2]

            # Stack channels → [3,H/2,W/2] and [3,3,H/2,W/2]
            ll_c = torch.stack(ll_c, dim=0)                            # [3,H/2,W/2]
            hi_c = torch.stack(hi_c, dim=0)                            # [3,3,H/2,W/2]

            ll_list.append(ll_c)
            hi_list.append(hi_c)

        xll = torch.stack(ll_list, dim=0).to(x.device, dtype=torch.float32)                # [B,3,H/2,W/2]
        xh  = torch.stack(hi_list, dim=0).to(x.device, dtype=torch.float32)                # [B,3,3,H/2,W/2]

        # Match pytorch_wavelets' API: xh in a list with one level
        return xll, [xh]


class SPyrInverse(torch.nn.Module):
    """
    Approximate inverse for SPyrForward used in WaveDiff.

    Input:
      (xll, [xh]) with:
        xll:   [B,3,H/2,W/2]
        xh[0]: [B,3,3,H/2,W/2]   (3 orientations)

    Output:
      x: [B,3,H,W]  (here H=W=32 for CIFAR)

    This is NOT an exact steerable pyramid inverse.
    It's a consistent deterministic reconstruction that
    treats xll as low-frequency content and the summed
    upsampled bands as high-frequency residuals.
    """

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, tup) -> torch.Tensor:
        xll, xh_list = tup
        xh = xh_list[0]                  # [B,3,3,h2,w2]
        B, C, O, h2, w2 = xh.shape
        H, W = h2 * self.scale_factor, w2 * self.scale_factor

        # Upsample lowpass to full resolution
        # [B,3,h2,w2] -> [B,3,H,W]
        xll_up = torch.nn.functional.interpolate(
            xll,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        # Merge orientations:
        # reshape [B,3,3,h2,w2] -> [B,3*3,h2,w2]
        xh_flat = xh.view(B, C * O, h2, w2)

        # Upsample high-freq maps to full resolution
        # [B,9,h2,w2] -> [B,9,H,W]
        xh_up = torch.nn.functional.interpolate(
            xh_flat,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        # Reshape back to [B,3,3,H,W] and sum across orientations
        xh_up = xh_up.view(B, C, O, H, W)
        xh_sum = xh_up.sum(dim=2)        # [B,3,H,W]

        # Combine low and high frequency components
        x_rec = xll_up + xh_sum

        # Optionally clamp to valid range (WaveDiff uses [-1,1] internally)
        x_rec = x_rec.clamp(-1.0, 1.0)

        return x_rec
