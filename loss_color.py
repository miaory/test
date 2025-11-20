import torch
import torch.nn as nn

# ---- RGB[0,1] -> Lab（CIELAB, D65）----
# 简化实现（线性化 + XYZ + Lab），保持数值安全
def rgb_to_lab(img):
    """
    img: [B,C,H,W], RGB, 0..1
    return: L*, a*, b*  （范围近似：L∈[0,100]，a/b≈[-128,127]）
    """
    x = img.clamp(0, 1).to(torch.float32)

    # sRGB -> linear RGB
    thr = 0.04045
    linear = torch.where(x <= thr, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))

    # linear RGB -> XYZ (D65)
    # Matrix from sRGB to XYZ (IEC 61966-2-1:1999)
    r, g, b = linear[:, 0:1], linear[:, 1:2], linear[:, 2:3]
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

    # Normalize by D65 white point
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = (X / Xn).clamp(min=1e-8)
    y = (Y / Yn).clamp(min=1e-8)
    z = (Z / Zn).clamp(min=1e-8)

    def _f(t):
        eps = 216/24389  # 0.008856...
        kappa = 24389/27 # 903.3...
        return torch.where(t > eps, t.pow(1/3), (kappa * t + 16) / 116)

    fx, fy, fz = _f(x), _f(y), _f(z)
    L = (116 * fy - 16)            # [0,100]
    a = 500 * (fx - fy)            # ~[-128,127]
    b = 200 * (fy - fz)            # ~[-128,127]

    # 数值安全
    L = torch.nan_to_num(L, nan=50.0, posinf=100.0, neginf=0.0)
    a = torch.nan_to_num(a, nan=0.0,  posinf=127.0, neginf=-128.0)
    b = torch.nan_to_num(b, nan=0.0,  posinf=127.0, neginf=-128.0)
    return L, a, b

def _safe_mean(x, dims):
    m = x.mean(dim=dims, keepdim=False)
    return torch.nan_to_num(m, nan=0.0)

def _safe_var(x, dims, eps=1e-6):
    v = x.var(dim=dims, unbiased=False, keepdim=False)
    v = torch.nan_to_num(v, nan=0.0)
    return v + eps

def _pair_corr(x, y, dims, eps=1e-6, clip_delta=1e-6):
    xm = x - x.mean(dim=dims, keepdim=True)
    ym = y - y.mean(dim=dims, keepdim=True)
    cov = (xm * ym).mean(dim=dims, keepdim=False)
    varx = x.var(dim=dims, unbiased=False, keepdim=False) + eps
    vary = y.var(dim=dims, unbiased=False, keepdim=False) + eps
    den = torch.sqrt(varx * vary) + eps
    rho = cov / den
    rho = torch.nan_to_num(rho, nan=0.0)
    # clamp 到 (-1,1) 内，留一点余量避免后续 sqrt/log 之类操作
    rho = torch.clamp(rho, min=-1.0 + clip_delta, max=1.0 - clip_delta)
    return rho

class ColorConsistencyLoss(nn.Module):
    """
    由三个部分组成：
    1) L_Lab_mean_window：ā/b̄ 均值落在给定窗口；窗口外按距离线性惩罚。
    2) L_Lab_var：a/b 方差落在自然带宽内；带宽外惩罚。
    3) L_RGB_stat：RGB 通道统计稳定（均值/方差约束 + 低相关性）。
    """
    def __init__(self,
                 a_window=(126.0, 134.0),
                 b_window=(128.0, 140.0),
                 var_band=(40.0, 1200.0),
                 w_lab_mean=0.08,
                 w_lab_var=0.02,
                 w_rgb_stat=0.03,
                 mu_gap=0.08,
                 rho_min=0.10):
        super().__init__()
        self.a_min, self.a_max = a_window
        self.b_min, self.b_max = b_window
        self.var_lo, self.var_hi = var_band
        self.w_lab_mean = float(w_lab_mean)
        self.w_lab_var  = float(w_lab_var)
        self.w_rgb_stat = float(w_rgb_stat)
        self.mu_gap = float(mu_gap)      # RGB 均值偏移惩罚
        self.rho_min = float(rho_min)    # 通道相关性的最小“安全距离”

    def forward(self, pred_rgb):
        """
        pred_rgb: [B,3,H,W], 0..1
        """
        x = pred_rgb.clamp(0, 1).to(torch.float32)

        # ---- Lab 统计 ----
        L, a, b = rgb_to_lab(x)   # L∈[0,100], a/b≈[-128,127]

        dims_spatial = (2, 3)
        a_mean = _safe_mean(a, dims_spatial) + 128.0  # 变换到 OpenCV 常用 0..255 视觉域
        b_mean = _safe_mean(b, dims_spatial) + 128.0
        a_var  = _safe_var(a, dims_spatial)           # 方差加 eps
        b_var  = _safe_var(b, dims_spatial)

        # 均值窗口惩罚（超出窗口的距离，L1 惩罚）
        def _window_penalty(m, lo, hi):
            under = torch.relu(lo - m)
            over  = torch.relu(m - hi)
            return (under + over).mean()

        l_mean = _window_penalty(a_mean, self.a_min, self.a_max) + \
                 _window_penalty(b_mean, self.b_min, self.b_max)

        # 方差带宽惩罚（落在 [var_lo, var_hi] 外的距离）
        def _band_penalty(v, lo, hi):
            low  = torch.relu(lo - v)
            high = torch.relu(v - hi)
            return (low + high).mean()

        l_var  = _band_penalty(a_var, self.var_lo, self.var_hi) + \
                 _band_penalty(b_var, self.var_lo, self.var_hi)

        # ---- RGB 统计（均值、方差、相关性）----
        # 均值尽量居中（0.5）且不过分偏移；方差不要太小（避免退化）
        mu = _safe_mean(x, dims_spatial)              # [B,3]
        vr = _safe_var(x, dims_spatial)               # [B,3]

        # 均值偏移：|mu - 0.5| - mu_gap
        l_mu = torch.relu((mu - 0.5).abs() - self.mu_gap).mean()
        # 方差下界：太小容易产生条带/退化
        l_var_rgb = torch.relu(0.01 - vr).mean()      # 经验下界，可按需调节

        # 通道相关性：希望不过分线性耦合
        r_rg = _pair_corr(x[:,0:1], x[:,1:2], dims_spatial)
        r_rb = _pair_corr(x[:,0:1], x[:,2:2+1], dims_spatial)
        r_gb = _pair_corr(x[:,1:2], x[:,2:2+1], dims_spatial)
        # 若 |rho| > (1 - rho_min) 则惩罚靠近 ±1 的强相关
        def _corr_penalty(rho, rho_min):
            return torch.relu(rho.abs() - (1.0 - rho_min)).mean()
        l_corr = _corr_penalty(r_rg, self.rho_min) + \
                 _corr_penalty(r_rb, self.rho_min) + \
                 _corr_penalty(r_gb, self.rho_min)

        l_rgb = l_mu + l_var_rgb + l_corr

        # ---- 汇总 ----
        L_color = self.w_lab_mean * l_mean + self.w_lab_var * l_var + self.w_rgb_stat * l_rgb

        # 训练日志用
        logs = {
            "L_color_total": float(L_color.detach().mean().item()),
            "L_Lab_mean_window": float(l_mean.detach().mean().item()),
            "L_Lab_var": float(l_var.detach().mean().item()),
            "L_RGB_stat": float(l_rgb.detach().mean().item()),
            "a_mean_cv": float(a_mean.detach().mean().item()),
            "b_mean_cv": float(b_mean.detach().mean().item()),
        }
        return L_color, logs
