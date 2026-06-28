"""Pure-Python mirror of the on-chain Fat Punks forward pass.

Exact integer arithmetic end-to-end (Python bigints == EVM int256 here:
no division anywhere, only mul/add/ReLU/compare). Verifies byte-for-byte
against real EVM execution results from contracts/evm-harness/run.js.

Usage:
  python3 verify/pure_forward.py --check          # compare vs evm_results.json
  python3 verify/pure_forward.py --token 0 --level 20   # print one body
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eth_hash.auto import keccak  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")

W = json.load(open(os.path.join(OUT, "weights.json")))
LAYERS = {L["name"]: L for L in W["layers"]}
THRESH = W["thresholds"]


def _i8(b):
    return b - 256 if b >= 128 else b


def latent_and_cond(token_id: int, fat_level: int):
    buf = []
    seed = token_id.to_bytes(32, "big")
    for batch in range(4):
        h = keccak(seed + batch.to_bytes(32, "big"))
        for j in range(8):
            buf.append(_i8(h[31 - j]))           # byte j from the LSB end
    c = fat_level * 12 - 120
    buf += [c] * 8
    return buf                                    # 40 ints


def dense(x, name):
    L = LAYERS[name]
    o_n, i_n = L["shape"]
    w, b, bs = L["w"], L["b"], L["b_scale_int"]
    out = []
    for o in range(o_n):
        acc = sum(w[o * i_n + i] * x[i] for i in range(i_n)) + b[o] * bs
        out.append(max(0, acc))
    return out


def convt(x, name, in_ch, out_ch, in_h, in_w):
    L = LAYERS[name]
    w, b, bs = L["w"], L["b"], L["b_scale_int"]   # shape [ic][oc][4][4]
    out_h, out_w = in_h * 2, in_w * 2
    out = [0] * (out_ch * out_h * out_w)
    for ic in range(in_ch):
        for ih in range(in_h):
            for iw in range(in_w):
                v = x[ic * in_h * in_w + ih * in_w + iw]
                if v == 0:
                    continue
                for kh in range(4):
                    oh = 2 * ih + kh - 1
                    if oh < 0 or oh >= out_h:
                        continue
                    for kw in range(4):
                        ow = 2 * iw + kw - 1
                        if ow < 0 or ow >= out_w:
                            continue
                        for oc in range(out_ch):
                            wv = w[((ic * out_ch + oc) * 4 + kh) * 4 + kw]
                            out[oc * out_h * out_w + oh * out_w + ow] += wv * v
    for oc in range(out_ch):
        bias = b[oc] * bs
        base = oc * out_h * out_w
        for p in range(out_h * out_w):
            out[base + p] = max(0, out[base + p] + bias)
    return out


def conv1x1(x, name, in_ch, out_ch, hw):
    L = LAYERS[name]
    w, b, bs = L["w"], L["b"], L["b_scale_int"]   # shape [oc][ic][1][1]
    out = [0] * (out_ch * hw)
    for oc in range(out_ch):
        bias = b[oc] * bs
        for p in range(hw):
            acc = sum(w[oc * in_ch + ic] * x[ic * hw + p]
                      for ic in range(in_ch))
            out[oc * hw + p] = acc + bias         # no ReLU
    return out


def body_levels(token_id: int, fat_level: int) -> bytes:
    x = latent_and_cond(token_id, fat_level)
    x = dense(x, "fc")                            # 256 = 16ch x 4x4
    x = convt(x, "convt1", 16, 16, 4, 4)
    x = convt(x, "convt2", 16, 8, 8, 8)
    x = convt(x, "convt3", 8, 8, 16, 16)
    x = conv1x1(x, "conv1x1", 8, 1, 32 * 32)
    lv = bytearray(1024)
    t0, t1, t2, t3 = THRESH
    for i, v in enumerate(x):
        lv[i] = 0 if v < t0 else 1 if v < t1 else 2 if v < t2 else 3 if v < t3 else 4
    return bytes(lv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--token", type=int, default=0)
    ap.add_argument("--level", type=int, default=20)
    a = ap.parse_args()
    if a.check:
        evm = json.load(open(os.path.join(OUT, "evm_results.json")))
        bad = 0
        for key, hx in evm.items():
            tid, lvl = (int(v) for v in key.split("_"))
            mine = body_levels(tid, lvl).hex()
            if mine != hx:
                bad += 1
                d = next(i for i in range(len(hx))
                         if hx[i] != mine[i]) // 2
                print(f"MISMATCH {key}: first diff at byte {d} "
                      f"(evm {hx[2*d:2*d+2]} vs py {mine[2*d:2*d+2]})")
        n = len(evm)
        if bad == 0:
            print(f"BYTE-EXACT: {n}/{n} bodies identical (python == EVM)")
        else:
            print(f"FAILED: {bad}/{n} mismatched")
            sys.exit(1)
    else:
        lv = body_levels(a.token, a.level)
        for y in range(32):
            print("".join(" .:+#"[lv[y * 32 + x]] for x in range(32)))


if __name__ == "__main__":
    main()
