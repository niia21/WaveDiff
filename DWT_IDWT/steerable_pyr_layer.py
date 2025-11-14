# Adapter that mimics the DWTForward/DWTInverse 
# 1-level SP with 3 orientations
#
# Output matches WaveDiff's [LL, LH, HL, HH] in our case that would be:
#   LL := residual lowpass
#   (LH, HL, HH) := 3 oriented bandpass subbands at level 0
#
# Returned shapes:
#   forward(x): (xll [B,3,H/2,W/2], [xh [B,3,3,H/2,W/2]])
#   inverse((xll, [xh])) -> x [B,3,H,W]
#
#
# https://pyrtools.readthedocs.io/en/latest/tutorials/03_steerable_pyramids.html


import torch
import torch.nn.functional as F
from typing import Tuple, List
from pyrtools.pyramids import SteerablePyramidFreq as SPyrF  


class SPyrForward(torch.nn.Module):
    """
    Replacement for DWT_2D/pytorch_wavelets.DWTForward
    They return tuple (yl, yh), so we would return the same one
    
    
    We need: low-pass residual and 3 oriented bands at level 0
    then downsample by 2 -> (H/2, W/2)
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
        x: [B,3,H,W] 
        
        Returns:
          xll: [B,3,H/2,W/2]
          xh:  list with one tensor [B,3,3,H/2,W/2]
        """
        B, C, H, W = x.shape # batch, num of color channels, height, width
        assert C == 3, "Only RGB inputs allowed"

        
        ll_list = []
        hi_list = []  # will stack into [B,3,3,H/2,W/2]

        # pyrtools is numpy based so calculation will be made on CPU and then transfered to GPU
        x_cpu = x.detach().to('cpu')
        
        # for each batch
        for b in range(B):
            ll_c = [] # for low pass residual
            hi_c = [] # for high pass ones
            for c in range(C): # for each color channel
                """
                pyrtools only works on 2D numpy arrays :( (no tensors 😭)
                Meaning that we need to transfer our processes temporary on CPU
                and process each channel separately, as otherwise RGB is a 3D array
                """
                img = x_cpu[b, c].numpy()  # take image

                # build SP
                pyr = SPyrF(img, height=self.height, order=self.order, is_complex=False)

                # get coefficients: pyr_coeffs['residual_lowpass']/pyr_coeffs['residual_highpass']
                coeffs = pyr.pyr_coeffs

                # CLEAN IT UP!!!!!!!😡
                # I tried multiple ways to access low pass residuals
                if ('residual_lowpass', 0) in coeffs:
                    lp = coeffs[('residual_lowpass', 0)]
                elif ('residual_lowpass',) in coeffs:
                    lp = coeffs[('residual_lowpass',)]
                else:
                    # Fallback: use the lowest-scale bands as an approximate lowpass
                    low_keys = [k for k in coeffs.keys() if isinstance(k[0], int)]
                    if not low_keys:
                        raise KeyError(
                            "Could not find residual_lowpass in steerable pyramid coeffs; "
                            "available keys: {}".format(list(coeffs.keys())))
                     # take the max scale index 
                    max_scale = max(k[0] for k in low_keys)
                    # average over orientations at that coarsest scale to get approximation of lowpass
                    ori_keys = [k for k in low_keys if k[0] == max_scale]
                    lp = sum(coeffs[k] for k in ori_keys) / len(ori_keys)

                ll = torch.from_numpy(lp).unsqueeze(0).float()  # [1, H_lp, W_lp]

                
                # Take orientation bands that steerable pyramid already computes
                bands = []
                for k in range(self.order + 1):  # 3 orientations if order=2 (0, 60, 120)
                    bands.append(torch.from_numpy(coeffs[(0, k)]).unsqueeze(0).float())  # [1,hB,wB]
                hi = torch.stack(bands, dim=0)  # [3,1,hB,wB]

                # Downsample all to H/2, W/2 (needed cause Wavediff DWT does that)
                ll_ds = F.avg_pool2d(ll, kernel_size=2, stride=2).squeeze(0)     # [1,H/2,W/2]
                hi_ds = F.avg_pool2d(hi, kernel_size=2, stride=2)                # [3,1,H/2,W/2]

                ll_c.append(ll_ds)                    # list of [1,H/2,W/2]
                hi_c.append(hi_ds.squeeze(1))         # list of [3,H/2,W/2]

            # Stack channels -> [3,H/2,W/2] and [3,3,H/2,W/2]
            ll_c = torch.stack(ll_c, dim=0)                            # [3,H/2,W/2]
            hi_c = torch.stack(hi_c, dim=0)                            # [3,3,H/2,W/2]

            ll_list.append(ll_c)
            hi_list.append(hi_c)

        
        # finally stuck batches and return ti GPU
        # num on imgs(in a batch), num of channels(RGB), Height/2, Width/2
        xll = torch.stack(ll_list, dim=0).to(x.device, dtype=torch.float32)                # [B,3,H/2,W/2]
        xh  = torch.stack(hi_list, dim=0).to(x.device, dtype=torch.float32)                # [B,3,3,H/2,W/2]

        return xll, [xh]


class SPyrInverse(torch.nn.Module):
    """
    replacement for IDWT_2D/pytorch_wavelets.DWTInverse
    They return: Reconstructed input of shape (𝑁,𝐶_𝑖𝑛,𝐻_𝑖𝑛,𝑊_𝑖𝑛)
    So we need to return the same

    
    Input:
      (xll, [xh]) with:
        xll:   [B,3,H/2,W/2]
        xh[0]: [B,3,3,H/2,W/2]   

    Output:
      x: [B,3,H,W]  

    This is NOT an exact mathematical SP inverse.
    It's a deterministic reconstruction, where we take
    xll and summed over xh and upsample and sum them over.
    We cant use exact inverse recon_pyr that pyrtools provide as
    recon_pyr() expects a full pyramid. While our defined forward doesnt compute it
    as we sticked to producing structure line in wavelet [LL, LH, HL, HH], thus only level 0 orientations
    thus not enough info for creating a proper pyramid
    """

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        #as we need to only upsample the components based on that factor
        # H/2 -> H     W/2 -> W
        self.scale_factor = scale_factor

    def forward(self, tup) -> torch.Tensor:
        xll, xh_list = tup
        xh = xh_list[0]                  #  [B,3,3,h2,w2] aka [B,3,3,h/2,w/2]
        B, C, O, h2, w2 = xh.shape       # batch size, num of channels, num of orientations, h2, w2
        H, W = h2 * self.scale_factor, w2 * self.scale_factor # get normal heoght and width

        # Upsample lowpas  [B,3,h2,w2] -> [B,3,H,W]
        xll_up = torch.nn.functional.interpolate(
            xll,
            size=(H, W),
            mode="bilinear", # like linear but for 2D
            align_corners=False,
        )


        
        
        # reshape [B,3,3,h2,w2] -> [B,3*3,h2,w2] as interpolate expects [B, C, H, W]
        xh_flat = xh.view(B, C * O, h2, w2)
        # upsample highpass [B,9,h2,w2] -> [B,9,H,W]
        xh_up = torch.nn.functional.interpolate(
            xh_flat,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

        # Reshape back to [B,3,3,H,W] 
        xh_up = xh_up.view(B, C, O, H, W)
        # sum over orientations
        xh_sum = xh_up.sum(dim=2)        # [B,3,H,W]

        # Combine low and high frequency components
        x_rec = xll_up + xh_sum
        # This part is optional so MORE TESTING NEEDS TO BE DONE 😡
        x_rec = x_rec.clamp(-1.0, 1.0)

        return x_rec
