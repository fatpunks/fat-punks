"""Fat Punks conditional DCGAN - mirrors the on-chain inference engine.

Generator layer semantics MUST match contracts/src/FatPunksRenderer.sol:
  dense(40 -> 256) + ReLU            <- engine fuses ReLU into the dense op
  convT(16->16,4,2,1) [+BN] + ReLU   4x4   -> 8x8
  convT(16-> 8,4,2,1) [+BN] + ReLU   8x8   -> 16x16
  convT( 8-> 8,4,2,1) [+BN] + ReLU   16x16 -> 32x32
  conv1x1(8->1)                      (no activation)

On-chain integer semantics (verified against Artificial.sol):
  A_k = ReLU(W_k_int8 . A_{k-1} + b_k_int8 * 128), A_0 = keccak-latent int8s.
  Activations are wide ints, never rescaled between layers; the only
  normalization is the final scale-free  pixel = out*256/(3*meanAbs)+128.

Training convention (exactly equivalent up to a positive global scale,
which meanAbs cancels): inputs are divided by 128 inside the model, so
callers always pass RAW integer-valued tensors (latent int8s, cond int8).
Output is raw logits, thresholded directly against fixed absolute cuts
(Y_THRESHOLDS); QAT export turns those cuts into int256 chain constants.
"""
import torch
import torch.nn as nn

LATENT_DIM = 32
COND_DIM = 8
G_CHANNELS = [16, 16, 8, 8]
IMG = 32
MAX_LEVEL = 20

# Gray anchors of the training data, mapped to logit space via
# y = (gray-128)/128, so anchors sit at {-1, -0.53, 0.02, 0.48, 0.95}.
GRAY_ANCHORS = [0, 60, 130, 190, 250]
# ABSOLUTE-THRESHOLD DESIGN (v2): the model's raw logits ARE the rendered
# quantity. No meanAbs normalization anywhere — Han's per-image auto-contrast
# pins mean|y| to 1/3, which makes mostly-background slim bodies infeasible
# under fixed thresholds (observed as uniform bucket-1 collapse). Instead,
# exact-grid QAT gives chain_logit = float_logit * C with C = 128/prod(s_j)
# known at export, so fixed float cuts become fixed int256 chain constants.
# Cuts are midpoints between anchors in y units: (pixel_cut - 128)/128.
PIXEL_THRESHOLDS = [30, 95, 160, 220]            # documentation / pixel space
Y_THRESHOLDS = [(t - 128) / 128.0 for t in PIXEL_THRESHOLDS]
#             = [-0.765625, -0.2578125, 0.25, 0.71875]


def level_to_cond_int(level):
    """fat level 0..20 -> int8 scalar in [-120, 120], step 12."""
    return level * 12 - 120


def sample_latent(n, device="cpu", generator=None):
    # Critical: uniform int8, matching keccak-derived bytes on chain. NOT randn.
    return torch.randint(-128, 128, (n, LATENT_DIM), device=device,
                         generator=generator).float()


def cond_vector(levels, device="cpu"):
    """levels: LongTensor (n,) -> (n, COND_DIM) float of replicated int8s."""
    c = (levels.float() * 12 - 120).to(device)
    return c.unsqueeze(1).repeat(1, COND_DIM)


def to_critic_image(y):
    """Raw logits -> [-1,1] image for the critic."""
    return y.clamp(-1.0, 1.0)


def real_to_critic_image(gray_u8):
    """uint8 grayscale tensor -> critic image: y = (gray-128)/128."""
    return (gray_u8.float() - 128.0) / 128.0


def bucketize_y(y):
    """Raw logits -> level index 0..4 using the frozen absolute cuts."""
    lev = torch.zeros_like(y, dtype=torch.long)
    for t in Y_THRESHOLDS:
        lev = lev + (y >= t).long()
    return lev


class Generator(nn.Module):
    def __init__(self, bn=True):
        super().__init__()
        ch = G_CHANNELS
        self.fc = nn.Linear(LATENT_DIM + COND_DIM, ch[0] * 4 * 4)

        def block(i, o):
            layers = [nn.ConvTranspose2d(i, o, 4, 2, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(o))
            layers.append(nn.ReLU(True))
            return layers
        self.conv_blocks = nn.Sequential(
            *block(ch[0], ch[1]), *block(ch[1], ch[2]), *block(ch[2], ch[3]))
        self.output = nn.Conv2d(ch[3], 1, 1, 1, 0)

    def forward(self, z, c):
        """z, c: RAW integer-valued tensors (int8 range). Returns raw logits."""
        x = torch.cat([z, c], dim=1) / 128.0
        x = torch.relu(self.fc(x))
        x = x.view(-1, G_CHANNELS[0], 4, 4)
        x = self.conv_blocks(x)
        return self.output(x)


class Critic(nn.Module):
    """WGAN-GP critic; cond enters as a constant extra channel. No BN (GP)."""

    def __init__(self):
        super().__init__()
        def gn(c):
            return nn.GroupNorm(1, c)  # LayerNorm-style, GP-safe
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1), nn.LeakyReLU(0.2, True),            # 16
            nn.Conv2d(32, 64, 4, 2, 1), gn(64), nn.LeakyReLU(0.2, True),   # 8
            nn.Conv2d(64, 128, 4, 2, 1), gn(128), nn.LeakyReLU(0.2, True), # 4
        )
        self.head = nn.Linear(128 * 4 * 4, 1)

    def forward(self, img, c):
        cc = (c[:, :1] / 128.0).view(-1, 1, 1, 1).expand(-1, 1, IMG, IMG)
        h = self.net(torch.cat([img, cc], dim=1))
        return self.head(h.flatten(1))
