"""Fat Punks conditional WGAN-GP trainer (float phase + exact-grid QAT phase).

Usage:
  python3 train.py --phase float --epochs 400
  python3 train.py --phase float --resume          # continue from checkpoint
  python3 train.py --phase qat   --epochs 200      # starts from EMA float ckpt
  python3 train.py --bench                          # 20-iter timing estimate

Epoch = one pass of the critic over the dataset (~45 critic batches at
batch 128 -> 9 generator steps with n_critic=5). Checkpoints + sample grids
land in ml/out/.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import model as M
from model import (Generator, Critic, sample_latent, cond_vector,
                   to_critic_image, real_to_critic_image, bucketize_y,
                   GRAY_ANCHORS, MAX_LEVEL, LATENT_DIM)
from quant import fold_bn, assert_fold_exact, QATGenerator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DATA = os.path.join(HERE, "data")
TIERS = {**{l: "slim" for l in range(0, 6)},
         **{l: "chubby" for l in range(6, 11)},
         **{l: "fat" for l in range(11, 16)},
         **{l: "huge" for l in range(16, 21)}}


# ------------------------------------------------------------------- data

def load_dataset():
    labels = json.load(open(os.path.join(DATA, "labels.json")))
    n = len(labels)
    imgs = torch.empty(n, 1, M.IMG, M.IMG)
    levels = torch.empty(n, dtype=torch.long)
    for i, (fn, lev) in enumerate(labels.items()):
        p = os.path.join(DATA, "bodies", TIERS[lev], fn)
        a = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        imgs[i, 0] = real_to_critic_image(torch.from_numpy(a))
        levels[i] = lev
    return imgs, levels


# ------------------------------------------------------------------- helpers

def ema_update(ema_sd, model, decay):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point and k in ema_sd:
                ema_sd[k].mul_(decay).add_(v, alpha=1 - decay)


def materialize_ema(model, ema_sd):
    m = copy.deepcopy(model)
    sd = m.state_dict()
    for k in ema_sd:
        sd[k] = ema_sd[k].clone()
    m.load_state_dict(sd)
    return m.eval()


def gradient_penalty(critic, real, fake, c):
    a = torch.rand(real.size(0), 1, 1, 1)
    x = (a * real + (1 - a) * fake).requires_grad_(True)
    out = critic(x, c)
    g, = torch.autograd.grad(out.sum(), x, create_graph=True)
    return ((g.flatten(1).norm(2, dim=1) - 1) ** 2).mean()


@torch.no_grad()
def save_sample_grid(gen, path, ids=6, levels=(0, 5, 10, 15, 20), seed=123):
    gen.eval()
    rng = torch.Generator().manual_seed(seed)
    zs = sample_latent(ids, generator=rng)               # one z per identity
    rows = []
    for i in range(ids):
        row = []
        for lev in levels:
            y = gen(zs[i:i + 1], cond_vector(torch.tensor([lev])))
            lvl = bucketize_y(y)[0, 0]                    # 32x32 ints 0..4
            g = torch.tensor(GRAY_ANCHORS, dtype=torch.uint8)[lvl]
            row.append(g.numpy())
        rows.append(np.concatenate(row, axis=1))
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid, "L").resize(
        (grid.shape[1] * 4, grid.shape[0] * 4), Image.NEAREST).save(path)


# ------------------------------------------------------------------- train

def build_models(args):
    """-> G, D, ema_sd, start_epoch (handles phase + resume)."""
    ck = os.path.join(OUT, f"ckpt_{args.phase}.pt")
    if args.resume and os.path.exists(ck):
        st = torch.load(ck, map_location="cpu", weights_only=False)
        if args.phase == "float":
            G = Generator(bn=True)
        else:
            G = QATGenerator(Generator(bn=False))
        D = Critic()
        G.load_state_dict(st["G"]); D.load_state_dict(st["D"])
        return G, D, st["ema"], st["epoch"], st.get("opt")

    if args.phase == "float":
        G = Generator(bn=True)
        D = Critic()
    else:                                   # qat: start from float EMA ckpt
        fck = torch.load(os.path.join(OUT, "ckpt_float.pt"),
                         map_location="cpu", weights_only=False)
        Gf = Generator(bn=True)
        Gf.load_state_dict(fck["G"])
        Gf = materialize_ema(Gf, fck["ema"])      # EMA weights, live buffers
        folded = fold_bn(Gf)
        rel = assert_fold_exact(Gf, folded)
        print(f"[qat] BN fold rel-mse {rel:.2e}")
        G = QATGenerator(folded)
        D = Critic(); D.load_state_dict(fck["D"])
    ema = {k: v.clone() for k, v in G.state_dict().items()
           if v.dtype.is_floating_point}
    return G, D, ema, 0, None


def train(args):
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(1337)
    imgs, levels = load_dataset()
    n = imgs.size(0)
    print(f"dataset {n} imgs in RAM")

    G, D, ema, start_ep, opt_state = build_models(args)
    g_updates = start_ep * 9  # 45 critic batches / n_critic — for EMA warmup
    lr = args.lr if args.lr else (2e-4 if args.phase == "float" else 5e-5)
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    if opt_state:
        optG.load_state_dict(opt_state["G"]); optD.load_state_dict(opt_state["D"])
    decay = 0.9995 if args.phase == "float" else 0.999

    bs, nc = args.batch, args.n_critic
    steps_per_ep = max(1, n // (bs * nc))
    log = open(os.path.join(OUT, f"train_{args.phase}.log"), "a", buffering=1)

    def batch():
        idx = torch.randint(0, n, (bs,))
        return imgs[idx], levels[idx]

    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        dl_sum = gl_sum = 0.0
        for _ in range(steps_per_ep):
            for _ in range(nc):
                real, lev = batch()
                c = cond_vector(lev)
                with torch.no_grad():
                    fake = to_critic_image(G(sample_latent(bs), c))
                d_loss = (D(fake, c).mean() - D(real, c).mean()
                          + args.gp * gradient_penalty(D, real, fake, c))
                optD.zero_grad(set_to_none=True)
                d_loss.backward()
                optD.step()
                dl_sum += d_loss.item()
            lev = torch.randint(0, MAX_LEVEL + 1, (bs,))
            c = cond_vector(lev)
            fake = to_critic_image(G(sample_latent(bs), c))
            g_loss = -D(fake, c).mean()
            optG.zero_grad(set_to_none=True)
            g_loss.backward()
            optG.step()
            g_updates += 1
            eff_decay = min(decay, (1.0 + g_updates) / (10.0 + g_updates))
            ema_update(ema, G, eff_decay)
            gl_sum += g_loss.item()

        msg = (f"ep {ep + 1}/{args.epochs} d {dl_sum / (steps_per_ep * nc):+.3f} "
               f"g {gl_sum / steps_per_ep:+.3f} "
               f"{(time.time() - t0) / (ep - start_ep + 1):.1f}s/ep")
        print(msg); log.write(msg + "\n")

        if (ep + 1) % args.sample_every == 0 or ep + 1 == args.epochs:
            Ge = materialize_ema(G, ema)
            save_sample_grid(Ge, os.path.join(
                OUT, f"samples_{args.phase}_{ep + 1:04d}.png"))
        if (ep + 1) % args.ckpt_every == 0 or ep + 1 == args.epochs:
            torch.save(dict(epoch=ep + 1, G=G.state_dict(), D=D.state_dict(),
                            ema=ema,
                            opt=dict(G=optG.state_dict(), D=optD.state_dict())),
                       os.path.join(OUT, f"ckpt_{args.phase}.pt"))
    log.close()


def bench(args):
    imgs, levels = load_dataset()
    G, D = Generator(bn=True), Critic()
    optG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
    bs = args.batch
    t0 = time.time()
    iters = 20
    for _ in range(iters):
        idx = torch.randint(0, imgs.size(0), (bs,))
        real, c = imgs[idx], cond_vector(levels[idx])
        with torch.no_grad():
            fake = to_critic_image(G(sample_latent(bs), c))
        d = (D(fake, c).mean() - D(real, c).mean()
             + 10 * gradient_penalty(D, real, fake, c))
        optD.zero_grad(); d.backward(); optD.step()
    tc = (time.time() - t0) / iters
    t0 = time.time()
    for _ in range(iters):
        c = cond_vector(torch.randint(0, 21, (bs,)))
        f = to_critic_image(G(sample_latent(bs), c))
        g = -D(f, c).mean()
        optG.zero_grad(); g.backward(); optG.step()
    tg = (time.time() - t0) / iters
    n = imgs.size(0)
    spe = (n // (bs * args.n_critic)) * (args.n_critic * tc + tg)
    print(f"critic-iter {tc * 1000:.0f}ms  G-iter {tg * 1000:.0f}ms  "
          f"-> ~{spe:.1f}s/epoch ({3600 / spe:.0f} ep/h) "
          f"at batch {bs}, n_critic {args.n_critic}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["float", "qat"], default="float")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n_critic", type=int, default=5)
    ap.add_argument("--gp", type=float, default=10.0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sample_every", type=int, default=10)
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(1)
    bench(a) if a.bench else train(a)
