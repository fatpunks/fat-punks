"""BN folding + QAT with EXACT on-chain quantization grids.

Chain semantics (Artificial.sol, verified):
    A_k = ReLU(W_k_int8 . A_{k-1} + b_k_int8 * SI_k),  A_0 = int8 latent,
    activations are wide ints, never rescaled; final meanAbs norm is
    scale-free.

In our normalized training convention (inputs /128 inside the model) this
is exactly equivalent to a float forward where, for layer k:
    weight grid step = s_k            (s_k = max|W_k|/127, per-tensor)
    bias: per-layer int8 with its own integer scale SI_k; the chain adds
    b_int8 * SI_k (pure integer). Float step = SI_k * prod_{j<=k}(s_j) / 128.
Historical note: the original prod-grid bias scheme range-clipped every
vanishing representable range on chain (b*128 is negligible against wide
accumulators) -- QAT therefore sees them collapse toward 0 and adapts the
weights, instead of us exporting math the chain silently ignores.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Generator, G_CHANNELS, LATENT_DIM, COND_DIM


# ---------------------------------------------------------------- BN folding

def fold_bn(gen_bn: Generator) -> Generator:
    """Generator(bn=True) -> Generator(bn=False) with identical eval output."""
    gen_bn = gen_bn.eval()
    out = Generator(bn=False).eval()
    out.fc.load_state_dict(gen_bn.fc.state_dict())
    out.output.load_state_dict(gen_bn.output.state_dict())

    src = list(gen_bn.conv_blocks)   # [convT,bn,relu] x3
    dst = list(out.conv_blocks)      # [convT,relu] x3
    for i in range(3):
        conv, bn = src[3 * i], src[3 * i + 1]
        tgt = dst[2 * i]
        gamma, beta = bn.weight.data, bn.bias.data
        mu, var, eps = bn.running_mean, bn.running_var, bn.eps
        scale = gamma / torch.sqrt(var + eps)            # per out-channel
        # ConvTranspose2d weight: [in_ch, out_ch, kH, kW] -> scale dim 1
        tgt.weight.data = conv.weight.data * scale.view(1, -1, 1, 1)
        b = conv.bias.data if conv.bias is not None else torch.zeros_like(mu)
        tgt.bias.data = (b - mu) * scale + beta
    return out


def assert_fold_exact(gen_bn, gen_folded, n=64, tol=1e-9):
    z = torch.randint(-128, 128, (n, LATENT_DIM)).float()
    c = torch.randint(0, 21, (n,))
    from model import cond_vector
    cv = cond_vector(c)
    with torch.no_grad():
        a = gen_bn.eval()(z, cv)
        b = gen_folded.eval()(z, cv)
    mse = F.mse_loss(a, b).item()
    rel = mse / (a.pow(2).mean().item() + 1e-12)
    assert rel < tol, f"BN fold mismatch: rel mse {rel:.3e}"
    return rel


# ----------------------------------------------------------------- fake-quant

def _fq_weight(w):
    s = w.detach().abs().max() / 127.0
    s = torch.clamp(s, min=1e-12)
    w_q = torch.clamp(torch.round(w / s), -127, 127) * s
    return w + (w_q - w).detach(), float(s)            # STE


def bias_scale_int(b, prod):
    """Per-layer bias scale, as the integer the chain multiplies b_int8 by.
    Chain bias = b_int8 * SI; float-equivalent step = SI * prod / 128.
    Computed in float64 so training/export agree to the bit."""
    bs_f = float(b.detach().abs().max()) / 127.0
    if bs_f <= 0.0:
        return 1
    return max(1, round(bs_f * 128.0 / prod))


def _fq_bias(b, prod):
    """Fake-quant bias on its own per-layer int8 grid (chain: b8 * SI)."""
    si = bias_scale_int(b, float(prod))
    step = b.new_tensor(si * (float(prod) / 128.0))
    b_q = torch.clamp(torch.round(b / step), -127, 127) * step
    return b + (b_q - b).detach()                       # STE


class QATGenerator(nn.Module):
    """Same architecture as folded Generator; forward applies exact-grid
    fake-quant to every weight and bias. Init from a folded Generator."""

    def __init__(self, folded: Generator):
        super().__init__()
        g = copy.deepcopy(folded)
        self.fc = g.fc
        convs = [m for m in g.conv_blocks if isinstance(m, nn.ConvTranspose2d)]
        assert len(convs) == 3
        self.convs = nn.ModuleList(convs)
        self.out = g.output

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1) / 128.0
        w, s = _fq_weight(self.fc.weight)
        prod = s
        x = F.relu(F.linear(x, w, _fq_bias(self.fc.bias, prod)))
        x = x.view(-1, G_CHANNELS[0], 4, 4)
        for conv in self.convs:
            w, s = _fq_weight(conv.weight)
            prod *= s
            x = F.relu(F.conv_transpose2d(
                x, w, _fq_bias(conv.bias, prod), stride=2, padding=1))
        w, s = _fq_weight(self.out.weight)
        prod *= s
        return F.conv2d(x, w, _fq_bias(self.out.bias, prod))

    # ------------------------------------------------------------ export

    @torch.no_grad()
    def export_int(self):
        """-> list of layer dicts with int8 weights/biases (chain semantics).
        Layer order: fc, convT1, convT2, convT3, conv1x1."""
        layers, prod = [], 1.0
        def grab(name, w, b):
            nonlocal prod
            s = float(w.abs().max() / 127.0)
            prod *= s
            w_int = torch.clamp(torch.round(w / s), -127, 127).to(torch.int8)
            si = bias_scale_int(b, prod)
            step = si * (prod / 128.0)
            b_int = torch.clamp(torch.round(b / step), -127, 127).to(torch.int8)
            layers.append(dict(name=name, w=w_int.cpu(), b=b_int.cpu(),
                               w_scale=s, b_scale_int=si))
        grab("fc", self.fc.weight, self.fc.bias)
        for i, conv in enumerate(self.convs):
            grab(f"convt{i+1}", conv.weight, conv.bias)
        grab("conv1x1", self.out.weight, self.out.bias)
        return layers
