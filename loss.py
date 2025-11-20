# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # try optional deps
# try:
#     import torchvision.models as tvmodels  # for VGG perceptual
#     _HAS_TORCHVISION = True
# except Exception:
#     _HAS_TORCHVISION = False

# try:
#     import pyiqa  # for URanker / MUSIQ (optional)
#     _HAS_PYIQA = True
# except Exception:
#     _HAS_PYIQA = False

# from option import get_option


# # -------------------- Basic losses --------------------

# class CharbonnierLoss(nn.Module):
#     def __init__(self, eps: float = 1e-6):
#         super().__init__()
#         self.eps2 = float(eps) ** 2

#     def forward(self, inp, target):
#         return ((F.mse_loss(inp, target, reduction='none') + self.eps2) ** 0.5).mean()


# class OutlierAwareLoss(nn.Module):
#     """
#     Robust L1 with spatial reweighting from residual statistics.
#     """
#     def __init__(self, eps: float = 1e-6):
#         super().__init__()
#         self.eps = eps

#     def forward(self, pred, gt):
#         delta = pred - gt
#         # spatial statistics
#         avg = delta.mean(dim=(2, 3), keepdim=True)
#         var = delta.std(dim=(2, 3), keepdim=True) / (2 ** 0.5)
#         weight = torch.tanh((delta - avg).abs() / (var + self.eps)).detach()
#         return (delta.abs() * weight).mean()


# # -------------------- LVW: Local Variance-Weighted (paper Eq.7–10) --------------------

# class LVWLoss(nn.Module):
#     """
#     Paper-accurate LVW:
#       Δ = mean(|pred - gt|, dim=channel)  -> (B,1,H,W)
#       μ = box_mean(Δ), σ = sqrt(box_mean(Δ^2) - μ^2 + eps)
#       w = tanh(|Δ - μ| / (σ + eps))
#       LVW = mean(w * Δ)
#     """
#     def __init__(self, win_size: int = 11, eps: float = 1e-3, channel_reduce: str = "l1"):
#         super().__init__()
#         assert win_size % 2 == 1 and win_size >= 3
#         self.win = int(win_size)
#         self.eps = float(eps)
#         self.channel_reduce = channel_reduce
#         self.pad = self.win // 2
#         # use normalized box filter via avg_pool2d with replicate padding
#         # (no registered kernel needed)

#     def _box_mean(self, x: torch.Tensor) -> torch.Tensor:
#         x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='replicate')
#         return F.avg_pool2d(x, self.win, stride=1)

#     def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
#         diff = pred - gt
#         if self.channel_reduce == "l2":
#             delta = torch.sqrt((diff * diff).mean(dim=1, keepdim=True) + 1e-12)     # B,1,H,W
#         else:
#             delta = diff.abs().mean(dim=1, keepdim=True)                             # B,1,H,W

#         mu   = self._box_mean(delta)
#         ex2  = self._box_mean(delta * delta)
#         var  = (ex2 - mu * mu).clamp_min(0.0)
#         sigma = torch.sqrt(var + self.eps)

#         w = torch.tanh((delta - mu).abs() / (sigma + self.eps))
#         return (w * delta).mean()


# # -------------------- MSSSIM (borrowed/trimmed from UIEDP) --------------------

# def _gaussian(window_size, sigma):
#     gauss = torch.tensor([torch.exp(torch.tensor(-(x - window_size//2)**2 / float(2*sigma**2))) for x in range(window_size)])
#     return (gauss / gauss.sum()).float()

# def _create_window(window_size, channel):
#     _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = (_1D_window @ _1D_window.t()).unsqueeze(0).unsqueeze(0)
#     window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
#     return window

# def _ssim(img1, img2, window, window_size, channel, val_range=None):
#     if val_range is None:
#         max_val = 1
#         min_val = 0
#         L = max_val - min_val
#     else:
#         L = val_range
#     padd = 0
#     mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
#     sigma12   = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
#     C1 = (0.01 * L) ** 2
#     C2 = (0.03 * L) ** 2
#     v1 = 2.0 * sigma12 + C2
#     v2 = sigma1_sq + sigma2_sq + C2
#     cs = (v1 / v2).mean(dim=(1,2,3))
#     ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
#     ret = ssim_map.mean(dim=(1,2,3))
#     return ret, cs

# def _msssim(img1, img2, window_size=7, size_average=True, val_range=None, normalize=True):
#     device = img1.device
#     channel = img1.size(1)
#     window = _create_window(window_size, channel).to(device)
#     weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=device)
#     levels = weights.size(0)
#     mcs = []
#     for _ in range(levels):
#         ssim_val, cs = _ssim(img1, img2, window, window_size, channel, val_range)
#         mcs.append(cs)
#         img1 = F.avg_pool2d(img1, kernel_size=2, stride=2, padding=0)
#         img2 = F.avg_pool2d(img2, kernel_size=2, stride=2, padding=0)
#     mssim = torch.prod(torch.stack([v.clamp(min=1e-6) ** w for v, w in zip([ssim_val] + mcs[:-1], [weights[0]] + list(weights[1:]))], dim=0), dim=0)
#     if normalize:
#         return (1 - mssim.mean())  # as a loss
#     else:
#         return (1 - mssim.mean())

# class MSSSIMLoss(nn.Module):
#     def __init__(self, window_size=7):
#         super().__init__()
#         self.window_size = window_size

#     def forward(self, x, y):
#         return _msssim(x, y, window_size=self.window_size, size_average=True, val_range=None, normalize=True)


# # -------------------- VGG perceptual --------------------

# class VGGPerceptualLoss(nn.Module):
#     """
#     Content loss using VGG16 feature maps (relu3_3 and relu4_3).
#     Falls back to 0 if torchvision or weights are unavailable.
#     """
#     def __init__(self, layers=('relu3_3', 'relu4_3'), weight_per_layer=(1.0, 1.0)):
#         super().__init__()
#         self.enabled = _HAS_TORCHVISION
#         self.layers = layers
#         self.weight_per_layer = weight_per_layer
#         if not _HAS_TORCHVISION:
#             return
#         try:
#             # load features
#             vgg = tvmodels.vgg16(weights=getattr(tvmodels, 'VGG16_Weights', None).DEFAULT if hasattr(tvmodels, 'VGG16_Weights') else None)
#         except Exception:
#             try:
#                 vgg = tvmodels.vgg16(pretrained=True)
#             except Exception:
#                 self.enabled = False
#                 return
#         features = vgg.features
#         # indices for relu3_3 and relu4_3 in VGG16
#         self.idx = {'relu1_2':3, 'relu2_2':8, 'relu3_3':16, 'relu4_3':23}
#         self.blocks = nn.ModuleDict({
#             'relu3_3': nn.Sequential(*[features[i] for i in range(self.idx['relu3_3']+1)]).eval(),
#             'relu4_3': nn.Sequential(*[features[i] for i in range(self.idx['relu4_3']+1)]).eval(),
#         })
#         for p in self.blocks.parameters():
#             p.requires_grad = False
#         # imagenet normalization
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
#         std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
#         self.register_buffer('mean', mean)
#         self.register_buffer('std', std)

#     def forward(self, x, y):
#         if not self.enabled:
#             return x.new_zeros(())
#         # normalize to imagenet stats
#         xn = (x - self.mean) / self.std
#         yn = (y - self.mean) / self.std
#         loss = 0.0
#         for li, w in zip(self.layers, self.weight_per_layer):
#             fx = self.blocks[li](xn)
#             fy = self.blocks[li](yn)
#             loss = loss + w * F.l1_loss(fx, fy)
#         return loss


# # -------------------- PSNR (unchanged utility) --------------------

# class PSNRLoss(nn.Module):
#     def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
#         super().__init__()
#         assert reduction == 'mean'
#         self.loss_weight = loss_weight
#         self.toY = toY
#         self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
#         self.first = True

#     def forward(self, pred, target):
#         assert len(pred.size()) == 4
#         if self.toY:
#             if self.first:
#                 self.coef = self.coef.to(pred.device)
#                 self.first = False
#             pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
#             target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
#             pred, target = pred / 255., target / 255.

#         imdff = pred - target
#         rmse = ((imdff ** 2).mean(dim=(1, 2, 3)) + 1e-8).sqrt()
#         loss = 20 * torch.log10(1 / rmse).mean()
#         loss = (50.0 - loss) / 100.0
#         return loss


# # -------------------- Warmup / ISP losses (kept as before) --------------------

# class LossWarmup(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.loss_cb = CharbonnierLoss(1e-8)
#         self.loss_cs = nn.CosineSimilarity()

#     def forward(self, inp, gt, warmup1, warmup2):
#         loss = self.loss_cb(warmup2, inp) + (self.loss_cb(warmup1, gt) + (1 - self.loss_cs(warmup1.clamp(0, 1), gt)).mean())
#         return loss


# class LossISP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.loss_cs = nn.CosineSimilarity()
#         self.loss_oa = OutlierAwareLoss()
#         self.psnr = PSNRLoss()

#     def forward(self, out, gt):
#         return (self.loss_oa(out, gt) + (1 - self.loss_cs(out.clamp(0, 1), gt)).mean()) + 2 * self.psnr(out, gt)


# # -------------------- LLE (MAIN): Aggregated composite loss --------------------

# class LossLLE(nn.Module):
#     """
#     L = L_charb + L_outlier + L_LVW
#         + w_msssim * L_MS-SSIM
#         + w_vgg   * L_VGG(content 3/4)
#         + w_ur    * (-URanker) [low-frequency]
#         + w_mus   * (-MUSIQ)   [low-frequency]
#     Color consistency is handled outside (main2.py) as before.
#     """
#     def __init__(self):
#         super().__init__()
#         # defaults
#         opt = get_option()
#         cfg = getattr(opt, 'config', {}) or {}
#         lcfg = (cfg.get('loss', {}) if isinstance(cfg, dict) else {})
#         wcfg = lcfg.get('weights', {}) if isinstance(lcfg, dict) else {}

#         self.w_charb   = float(wcfg.get('charb', 1.0))
#         self.w_outlier = float(wcfg.get('outlier', 1.0))
#         self.w_lvw     = float(wcfg.get('lvw', 1.0))
#         self.w_msssim  = float(wcfg.get('msssim', 0.2))
#         self.w_vgg     = float(wcfg.get('vgg', 0.01))
#         self.w_ur      = float(wcfg.get('uranker', 1e-3))
#         self.w_mus     = float(wcfg.get('musiq', 1e-5))

#         lvw_win = int(lcfg.get('lvw_win', 11))
#         lvw_eps = float(lcfg.get('lvw_eps', 1e-3))
#         msssim_win = int(lcfg.get('msssim_win', 7))

#         # ---- LVW enable switch (backward compatible) ----
#         lvw_section = lcfg.get('lvw', {}) if isinstance(lcfg, dict) else {}
#         self.use_lvw = bool(lvw_section.get('enable', True))  # default True if not provided

#         icfg = lcfg.get('iqa', {}) if isinstance(lcfg, dict) else {}
#         self.iqa_freq = int(icfg.get('freq', 4))         # compute every N steps
#         self.iqa_size = int(icfg.get('down', 224))       # downsample to NxN
#         # step counter
#         self.register_buffer('_step', torch.zeros((), dtype=torch.long))

#         # components
#         self.charb = CharbonnierLoss()
#         self.outlier = OutlierAwareLoss()
#         self.lvw = LVWLoss(win_size=lvw_win, eps=lvw_eps) if (self.use_lvw and self.w_lvw > 0) else None
#         self.msssim = MSSSIMLoss(window_size=msssim_win)
#         self.vgg = VGGPerceptualLoss(layers=('relu3_3','relu4_3'), weight_per_layer=(1.0,1.0))

#         # Optional IQA models
#         if _HAS_PYIQA and (self.w_ur > 0 or self.w_mus > 0):
#             try:
#                 self.iqa_ur = pyiqa.create_metric('uranker', device=opt.device if hasattr(opt, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
#             except Exception:
#                 self.iqa_ur = None
#             try:
#                 self.iqa_mus = pyiqa.create_metric('musiq', device=opt.device if hasattr(opt, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
#             except Exception:
#                 self.iqa_mus = None
#         else:
#             self.iqa_ur = None
#             self.iqa_mus = None

#     def _iqa_term(self, x):
#         # x expected in [0,1]
#         terms = []
#         if (self._step.item() % max(1,self.iqa_freq)) != 0:
#             return 0.0
#         ds = F.interpolate(x, size=(self.iqa_size, self.iqa_size), mode='bilinear', align_corners=False)
#         if self.iqa_ur is not None and self.w_ur > 0:
#             try:
#                 q = self.iqa_ur(ds)  # higher is better
#                 terms.append(self.w_ur * (-q.mean()))
#             except Exception:
#                 pass
#         if self.iqa_mus is not None and self.w_mus > 0:
#             try:
#                 q = self.iqa_mus(ds)
#                 terms.append(self.w_mus * (-q.mean()))
#             except Exception:
#                 pass
#         if not terms:
#             return 0.0
#         return torch.stack([t if torch.is_tensor(t) else x.new_tensor(float(t)) for t in terms]).sum()

#     def forward(self, pred, gt):
#         pred = pred.clamp(0, 1)
#         gt   = gt.clamp(0, 1)

#         # increment step
#         self._step += 1

#         loss = 0.0
#         if self.w_charb:
#             loss = loss + self.w_charb   * self.charb(pred, gt)
#         if self.w_outlier:
#             loss = loss + self.w_outlier * self.outlier(pred, gt)
#         if self.use_lvw and self.w_lvw and (self.lvw is not None):
#             loss = loss + self.w_lvw     * self.lvw(pred, gt)
#         if self.w_msssim:
#             loss = loss + self.w_msssim  * self.msssim(pred, gt)
#         if self.w_vgg:
#             loss = loss + self.w_vgg     * self.vgg(pred, gt)

#         # optional IQA (low-frequency)
#         iqa = self._iqa_term(pred)
#         if torch.is_tensor(iqa):
#             loss = loss + iqa

#         return loss


# # -------------------- factory --------------------

# def import_loss(training_task: str):
#     if training_task == 'isp':
#         return LossISP()
#     elif training_task == 'lle':
#         return LossLLE()
#     elif training_task == 'warmup':
#         return LossWarmup()
#     else:
#         raise ValueError('unknown training task, please choose from [isp, lle, warmup].')
# 读yaml里的uidep损失
import torch
import torch.nn as nn
import torch.nn.functional as F

# try optional deps
try:
    import torchvision.models as tvmodels  # for VGG perceptual
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

try:
    import pyiqa  # for URanker / MUSIQ (optional)
    _HAS_PYIQA = True
except Exception:
    _HAS_PYIQA = False

from option import get_option


# -------------------- Basic losses --------------------

class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps2 = float(eps) ** 2

    def forward(self, inp, target):
        return ((F.mse_loss(inp, target, reduction='none') + self.eps2) ** 0.5).mean()


class OutlierAwareLoss(nn.Module):
    """
    Robust L1 with spatial reweighting from residual statistics.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, gt):
        delta = pred - gt
        # spatial statistics
        avg = delta.mean(dim=(2, 3), keepdim=True)
        var = delta.std(dim=(2, 3), keepdim=True) / (2 ** 0.5)
        weight = torch.tanh((delta - avg).abs() / (var + self.eps)).detach()
        return (delta.abs() * weight).mean()


# -------------------- LVW: Local Variance-Weighted (paper Eq.7–10) --------------------

class LVWLoss(nn.Module):
    """
    Paper-accurate LVW:
      Δ = mean(|pred - gt|, dim=channel)  -> (B,1,H,W)
      μ = box_mean(Δ), σ = sqrt(box_mean(Δ^2) - μ^2 + eps)
      w = tanh(|Δ - μ| / (σ + eps))
      LVW = mean(w * Δ)
    """
    def __init__(self, win_size: int = 11, eps: float = 1e-3, channel_reduce: str = "l1"):
        super().__init__()
        assert win_size % 2 == 1 and win_size >= 3
        self.win = int(win_size)
        self.eps = float(eps)
        self.channel_reduce = channel_reduce
        self.pad = self.win // 2
        # use normalized box filter via avg_pool2d with replicate padding
        # (no registered kernel needed)

    def _box_mean(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='replicate')
        return F.avg_pool2d(x, self.win, stride=1)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        diff = pred - gt
        if self.channel_reduce == "l2":
            delta = torch.sqrt((diff * diff).mean(dim=1, keepdim=True) + 1e-12)     # B,1,H,W
        else:
            delta = diff.abs().mean(dim=1, keepdim=True)                             # B,1,H,W

        mu   = self._box_mean(delta)
        ex2  = self._box_mean(delta * delta)
        var  = (ex2 - mu * mu).clamp_min(0.0)
        sigma = torch.sqrt(var + self.eps)

        w = torch.tanh((delta - mu).abs() / (sigma + self.eps))
        return (w * delta).mean()


# -------------------- MSSSIM (borrowed/trimmed from UIEDP) --------------------

def _gaussian(window_size, sigma):
    gauss = torch.tensor([torch.exp(torch.tensor(-(x - window_size//2)**2 / float(2*sigma**2))) for x in range(window_size)])
    return (gauss / gauss.sum()).float()

def _create_window(window_size, channel):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = (_1D_window @ _1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, val_range=None):
    if val_range is None:
        max_val = 1
        min_val = 0
        L = max_val - min_val
    else:
        L = val_range
    padd = 0
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = (v1 / v2).mean(dim=(1,2,3))
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map.mean(dim=(1,2,3))
    return ret, cs

def _msssim(img1, img2, window_size=7, size_average=True, val_range=None, normalize=True):
    device = img1.device
    channel = img1.size(1)
    window = _create_window(window_size, channel).to(device)
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=device)
    levels = weights.size(0)
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(img1, img2, window, window_size, channel, val_range)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, kernel_size=2, stride=2, padding=0)
        img2 = F.avg_pool2d(img2, kernel_size=2, stride=2, padding=0)
    mssim = torch.prod(torch.stack([v.clamp(min=1e-6) ** w for v, w in zip([ssim_val] + mcs[:-1], [weights[0]] + list(weights[1:]))], dim=0), dim=0)
    if normalize:
        return (1 - mssim.mean())  # as a loss
    else:
        return (1 - mssim.mean())

class MSSSIMLoss(nn.Module):
    def __init__(self, window_size=7):
        super().__init__()
        self.window_size = window_size

    def forward(self, x, y):
        return _msssim(x, y, window_size=self.window_size, size_average=True, val_range=None, normalize=True)


# -------------------- VGG perceptual --------------------

class VGGPerceptualLoss(nn.Module):
    """
    Content loss using VGG16 feature maps (relu3_3 and relu4_3).
    Falls back to 0 if torchvision or weights are unavailable.
    """
    def __init__(self, layers=('relu3_3', 'relu4_3'), weight_per_layer=(1.0, 1.0)):
        super().__init__()
        self.enabled = _HAS_TORCHVISION
        self.layers = layers
        self.weight_per_layer = weight_per_layer
        if not _HAS_TORCHVISION:
            return
        try:
            # load features
            vgg = tvmodels.vgg16(weights=getattr(tvmodels, 'VGG16_Weights', None).DEFAULT if hasattr(tvmodels, 'VGG16_Weights') else None)
        except Exception:
            try:
                vgg = tvmodels.vgg16(pretrained=True)
            except Exception:
                self.enabled = False
                return
        features = vgg.features
        # indices for relu3_3 and relu4_3 in VGG16
        self.idx = {'relu1_2':3, 'relu2_2':8, 'relu3_3':16, 'relu4_3':23}
        self.blocks = nn.ModuleDict({
            'relu3_3': nn.Sequential(*[features[i] for i in range(self.idx['relu3_3']+1)]).eval(),
            'relu4_3': nn.Sequential(*[features[i] for i in range(self.idx['relu4_3']+1)]).eval(),
        })
        for p in self.blocks.parameters():
            p.requires_grad = False
        # imagenet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x, y):
        if not self.enabled:
            return x.new_zeros(())
        # normalize to imagenet stats
        xn = (x - self.mean) / self.std
        yn = (y - self.mean) / self.std
        loss = 0.0
        for li, w in zip(self.layers, self.weight_per_layer):
            fx = self.blocks[li](xn)
            fy = self.blocks[li](yn)
            loss = loss + w * F.l1_loss(fx, fy)
        return loss


# -------------------- PSNR (unchanged utility) --------------------

class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super().__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False
            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            pred, target = pred / 255., target / 255.

        imdff = pred - target
        rmse = ((imdff ** 2).mean(dim=(1, 2, 3)) + 1e-8).sqrt()
        loss = 20 * torch.log10(1 / rmse).mean()
        loss = (50.0 - loss) / 100.0
        return loss


# -------------------- Warmup / ISP losses (kept as before) --------------------

class LossWarmup(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_cb = CharbonnierLoss(1e-8)
        self.loss_cs = nn.CosineSimilarity()

    def forward(self, inp, gt, warmup1, warmup2):
        loss = self.loss_cb(warmup2, inp) + (self.loss_cb(warmup1, gt) + (1 - self.loss_cs(warmup1.clamp(0, 1), gt)).mean())
        return loss


class LossISP(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_cs = nn.CosineSimilarity()
        self.loss_oa = OutlierAwareLoss()
        self.psnr = PSNRLoss()

    def forward(self, out, gt):
        return (self.loss_oa(out, gt) + (1 - self.loss_cs(out.clamp(0, 1), gt)).mean()) + 2 * self.psnr(out, gt)


# -------------------- LLE (MAIN): Aggregated composite loss --------------------

class LossLLE(nn.Module):
    """
    L = L_charb + L_outlier + L_LVW
        + w_msssim * L_MS-SSIM
        + w_vgg   * L_VGG(content 3/4)
        + w_ur    * (-URanker) [low-frequency]
        + w_mus   * (-MUSIQ)   [low-frequency]
    Color consistency is handled outside (main2.py) as before.
    """
    def __init__(self):
        super().__init__()
        # defaults
        opt = get_option()
        cfg = getattr(opt, 'config', {}) or {}
        lcfg = (cfg.get('loss', {}) if isinstance(cfg, dict) else {})
        wcfg = lcfg.get('weights', {}) if isinstance(lcfg, dict) else {}

        # ---- 主损失权重（Charb / Outlier / LVW）从 weights 读取 ----
        self.w_charb   = float(wcfg.get('charb', 1.0))
        self.w_outlier = float(wcfg.get('outlier', 1.0))
        self.w_lvw     = float(wcfg.get('lvw', 1.0))

        # ---- UIEDP（MS-SSIM / VGG / UR / MUSIQ）权重：优先从 loss.uiedp 读取 ----
        ucfg = lcfg.get('uiedp', {}) if isinstance(lcfg, dict) else {}
        self.uiedp_enable = bool(ucfg.get('enable', True))

        # YAML 中 uiedp 中的命名：w_ms / w_perc / w_musiq / w_uranker
        # 若未提供，则回退到 weights 下的配置，再回退到默认值
        self.w_msssim = float(ucfg.get('w_ms',   wcfg.get('msssim', 0.2)))
        self.w_vgg    = float(ucfg.get('w_perc', wcfg.get('vgg',    0.01)))
        self.w_ur     = float(ucfg.get('w_uranker', wcfg.get('uranker', 1e-3)))
        self.w_mus    = float(ucfg.get('w_musiq',   wcfg.get('musiq',   1e-5)))

        lvw_win = int(lcfg.get('lvw_win', 11))
        lvw_eps = float(lcfg.get('lvw_eps', 1e-3))
        msssim_win = int(lcfg.get('msssim_win', 7))

        # ---- LVW enable switch (backward compatible) ----
        lvw_section = lcfg.get('lvw', {}) if isinstance(lcfg, dict) else {}
        self.use_lvw = bool(lvw_section.get('enable', True))  # default True if not provided

        # IQA/UIEDP 调度：
        # - 首选 loss.uiedp 下的 iqa_start / iqa_every / iqa_down
        # - 若缺失，则回退到旧的 loss.iqa 配置
        icfg = lcfg.get('iqa', {}) if isinstance(lcfg, dict) else {}
        self.iqa_freq = int(ucfg.get('iqa_every', icfg.get('freq', 4)))   # compute every N steps
        self.iqa_size = int(ucfg.get('iqa_down',  icfg.get('down', 224))) # downsample to NxN

        # 从第多少步开始启用 IQA（UR/MUSIQ），避免一开始就引导太强
        self.iqa_start_step = int(ucfg.get('iqa_start', 0))

        # MS-SSIM / VGG 的启用起始 epoch（粗粒度控制），默认立刻启用
        self.ms_perc_start_epoch = int(ucfg.get('ms_perc_start', 0))
        # step counter
        self.register_buffer('_step', torch.zeros((), dtype=torch.long))

        # components
        self.charb = CharbonnierLoss()
        self.outlier = OutlierAwareLoss()
        self.lvw = LVWLoss(win_size=lvw_win, eps=lvw_eps) if (self.use_lvw and self.w_lvw > 0) else None
        self.msssim = MSSSIMLoss(window_size=msssim_win)
        self.vgg = VGGPerceptualLoss(layers=('relu3_3','relu4_3'), weight_per_layer=(1.0,1.0))

        # Optional IQA models
        if _HAS_PYIQA and (self.w_ur > 0 or self.w_mus > 0):
            try:
                self.iqa_ur = pyiqa.create_metric('uranker', device=opt.device if hasattr(opt, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
            except Exception:
                self.iqa_ur = None
            try:
                self.iqa_mus = pyiqa.create_metric('musiq', device=opt.device if hasattr(opt, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu')
            except Exception:
                self.iqa_mus = None
        else:
            self.iqa_ur = None
            self.iqa_mus = None

    def _iqa_term(self, x):
        # x expected in [0,1]
        if not self.uiedp_enable:
            return 0.0

        step = self._step.item()
        terms = []

        # 频率控制 + 起始 step 控制
        if step < self.iqa_start_step:
            return 0.0
        if (step % max(1, self.iqa_freq)) != 0:
            return 0.0
        ds = F.interpolate(x, size=(self.iqa_size, self.iqa_size), mode='bilinear', align_corners=False)
        if self.iqa_ur is not None and self.w_ur > 0:
            try:
                q = self.iqa_ur(ds)  # higher is better
                terms.append(self.w_ur * (-q.mean()))
            except Exception:
                pass
        if self.iqa_mus is not None and self.w_mus > 0:
            try:
                q = self.iqa_mus(ds)
                terms.append(self.w_mus * (-q.mean()))
            except Exception:
                pass
        if not terms:
            return 0.0
        return torch.stack([t if torch.is_tensor(t) else x.new_tensor(float(t)) for t in terms]).sum()

    def forward(self, pred, gt):
        pred = pred.clamp(0, 1)
        gt   = gt.clamp(0, 1)

        # increment step
        self._step += 1

        loss = 0.0
        if self.w_charb:
            loss = loss + self.w_charb   * self.charb(pred, gt)
        if self.w_outlier:
            loss = loss + self.w_outlier * self.outlier(pred, gt)
        if self.use_lvw and self.w_lvw and (self.lvw is not None):
            loss = loss + self.w_lvw     * self.lvw(pred, gt)
        # MS-SSIM / VGG 按 uiedp 开关与启动 epoch 控制
        if self.uiedp_enable:
            cur_epoch = 0  # 若需要按 epoch 控制，可在外部传入或通过其他方式设置
            _allow_ms_perc = (cur_epoch >= self.ms_perc_start_epoch)
        else:
            _allow_ms_perc = True

        if self.w_msssim and _allow_ms_perc:
            loss = loss + self.w_msssim  * self.msssim(pred, gt)
        if self.w_vgg and _allow_ms_perc:
            loss = loss + self.w_vgg     * self.vgg(pred, gt)

        # optional IQA (low-frequency)
        iqa = self._iqa_term(pred)
        if torch.is_tensor(iqa):
            loss = loss + iqa

        return loss


# -------------------- factory --------------------

def import_loss(training_task: str):
    if training_task == 'isp':
        return LossISP()
    elif training_task == 'lle':
        return LossLLE()
    elif training_task == 'lle_psnr':  # ★ Stage2 PSNR微调也用 LossLLE
        return LossLLE()
    elif training_task == 'warmup':
        return LossWarmup()
    else:
        raise ValueError('unknown training task, please choose from [isp, lle, lle_psnr, warmup].')
