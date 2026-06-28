"""Export QAT weights -> contracts/src/render/WeightData.sol (+ weights.json).

Packs int8 tensors into the exact byte layouts FixedPointMath.sol consumes,
canonical-Huffman-compresses them with a single shared table compatible with
the on-chain decoder (61-byte header, 6-byte rows, 14-bit window), and emits
the Solidity library. `--random` exports an untrained QATGenerator so the
whole compile/verify pipeline can be proven before training finishes.
"""
import argparse
import heapq
import json
import os
from collections import Counter

import torch

import model as M
from model import Generator, Y_THRESHOLDS
from quant import QATGenerator, fold_bn

HERE = os.path.dirname(os.path.abspath(__file__))
SOL_OUT = os.path.join(HERE, "..", "contracts", "src", "render", "WeightData.sol")
JSON_OUT = os.path.join(HERE, "out", "weights.json")


# ----------------------------------------------------------- huffman encoder

def _code_lengths(freqs, max_len):
    """Package-merge length-limited Huffman. freqs: {sym: count} -> {sym: len}."""
    syms = sorted(freqs)
    if len(syms) == 1:
        return {syms[0]: 1}
    # package-merge
    items = [(freqs[s], (s,)) for s in syms]
    # boundary package-merge: lists[1] = items; L-1 package/merge rounds
    # yield lists[L]; taking 2(n-1) entries gives lengths capped at L.
    pkgs = sorted(items)
    for _ in range(max_len - 1):
        merged = []
        for i in range(0, len(pkgs) - 1, 2):
            merged.append((pkgs[i][0] + pkgs[i + 1][0],
                           pkgs[i][1] + pkgs[i + 1][1]))
        pkgs = sorted(items + merged)
    # take first 2*(n-1) packages of the final level
    lens = {s: 0 for s in syms}
    for f, group in pkgs[:2 * (len(syms) - 1)]:
        for s in group:
            lens[s] += 1
    assert max(lens.values()) <= max_len
    return lens


def build_huffman(data: bytes, max_len=10):
    """-> (codes {sym:(len,code)}, table_bytes) matching the on-chain decoder."""
    freqs = Counter(data)
    lens = _code_lengths(freqs, max_len)
    order = sorted(lens, key=lambda s: (lens[s], s))      # canonical order
    # canonical code assignment
    codes, code, prev_len = {}, 0, lens[order[0]]
    code <<= 0
    first = True
    for s in order:
        L = lens[s]
        if first:
            code = 0
            prev_len = L
            first = False
        else:
            code = (code + 1) << (L - prev_len)
            prev_len = L
        codes[s] = (L, code)
    # header rows per length, splitting counts > 255
    rows, sym_table = [], bytearray()
    by_len = {}
    for s in order:
        by_len.setdefault(lens[s], []).append(s)
    for L in sorted(by_len):
        group = by_len[L]
        i = 0
        while i < len(group):
            chunk = group[i:i + 255]
            first_code = codes[chunk[0]][1]
            rows.append((L, first_code, len(chunk), len(sym_table)))
            sym_table.extend(chunk)
            i += 255
    assert len(rows) <= 10, f"{len(rows)} huffman rows exceed 61-byte header"
    header = bytearray()
    for (L, fc, cnt, off) in rows:
        header += bytes([L, fc >> 8, fc & 0xFF, cnt, off >> 8, off & 0xFF])
    header += b"\x00" * (61 - len(header))
    return codes, bytes(header) + bytes(sym_table)


def huff_encode(data: bytes, codes):
    bits = bitlen = 0
    out = bytearray()
    for b in data:
        L, c = codes[b]
        bits = (bits << L) | c
        bitlen += L
        while bitlen >= 8:
            out.append((bits >> (bitlen - 8)) & 0xFF)
            bitlen -= 8
    if bitlen:
        out.append((bits << (8 - bitlen)) & 0xFF)
    total_bits = sum(codes[b][0] for b in data)
    out += b"\x00" * 4                                    # window safety pad
    return bytes(out), total_bits


def huff_decode_py(comp: bytes, table: bytes, n: int) -> bytes:
    """Python mirror of the on-chain decoder, for self-test."""
    rows = []
    for i in range(0, 60, 6):
        if table[i] == 0:
            break
        rows.append((table[i], (table[i + 1] << 8) | table[i + 2],
                     table[i + 3], (table[i + 4] << 8) | table[i + 5]))
    syms = table[61:]
    out, bitpos = bytearray(), 0
    while len(out) < n:
        byte_idx, bit_off = bitpos >> 3, bitpos & 7
        raw = int.from_bytes(comp[byte_idx:byte_idx + 3].ljust(3, b"\0"), "big")
        window = (raw >> (24 - (bit_off + 14))) & 16383
        matched = False
        for (L, fc, cnt, off) in rows:
            code = window >> (14 - L)
            if fc <= code < fc + cnt:
                out.append(syms[off + code - fc])
                bitpos += L
                matched = True
                break
        if not matched:
            bitpos += 1
    return bytes(out)


# ------------------------------------------------------------- byte packing

def i8b(t):                                               # int8 tensor -> bytes
    return bytes(int(v) & 0xFF for v in t.flatten().tolist())


def pad32(b: bytes) -> bytes:
    return b + b"\x00" * ((-len(b)) % 32 or 0) + (b"\x00" * 32 if len(b) % 32 == 0 and len(b) == 0 else b"")


def pack_dense(w):                                        # (256,40)
    return i8b(w)


def pack_convt(w):                                        # (inCh,outCh,4,4)
    ic_n, oc_n = w.shape[0], w.shape[1]
    out = bytearray()
    for ic in range(ic_n):
        for t in range(oc_n // 2):
            for oc in (2 * t, 2 * t + 1):
                out += i8b(w[ic, oc].flatten())
    return bytes(out)


def pack_conv1x1(w):                                      # (outCh,inCh,1,1)
    out = bytearray()
    for oc in range(w.shape[0]):
        out += i8b(w[oc].flatten())
    out += b"\x00" * 32                                   # mload overread pad
    return bytes(out)


def bias_word(b):                                         # int8 tensor -> uint256
    bb = i8b(b)
    assert len(bb) <= 32
    return int.from_bytes(bb.ljust(32, b"\x00"), "big")


# ----------------------------------------------------------------- solidity

SOL_HEADER = """\
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import { FixedPointMath } from "./FixedPointMath.sol";

/// @notice GENERATED by ml/export.py — do not edit by hand.
///         Int8 model weights, Huffman-compressed, decoded on demand.
/// @dev    get_* functions are PUBLIC so the ~16KB weight blob deploys as
///         its own library contract (DELEGATECALLed by the renderer) —
///         keeps both deployed artifacts under the EIP-170 limit while the
///         whole network still lives in contract bytecode. Constants stay
///         internal (inlined into the renderer at compile time).
library WeightData {
"""


def emit(layers, trained: bool):
    blobs = {
        "FC_WEIGHT": pack_dense(layers[0]["w"]),
        "CONVT1_WEIGHT": pack_convt(layers[1]["w"]),
        "CONVT2_WEIGHT": pack_convt(layers[2]["w"]),
        "CONVT3_WEIGHT": pack_convt(layers[3]["w"]),
        "CONV1X1_WEIGHT": pack_conv1x1(layers[4]["w"]),
        "FC_BIAS": i8b(layers[0]["b"]),
    }
    allbytes = b"".join(blobs.values())
    codes, table = build_huffman(allbytes)
    parts = [SOL_HEADER]
    parts.append(f'    bytes internal constant HUFF_TABLE = hex"{table.hex()}";\n')
    raw_total, comp_total = 0, len(table)
    for name, blob in blobs.items():
        comp, _bits = huff_encode(blob, codes)
        assert huff_decode_py(comp, table, len(blob)) == blob, name
        raw_total += len(blob)
        comp_total += len(comp)
        parts.append(
            f'\n    bytes internal constant {name}_HUFF = hex"{comp.hex()}";\n'
            f"    function get_{name}() public pure returns (bytes memory) {{\n"
            f"        return FixedPointMath.huffDecode({name}_HUFF, HUFF_TABLE, {len(blob)});\n"
            f"    }}\n")
    for i, key in ((1, "CONVT1"), (2, "CONVT2"), (3, "CONVT3"), (4, "CONV1X1")):
        parts.append(f"    uint256 internal constant BIAS_{key} = "
                     f"0x{bias_word(layers[i]['b']):064x};\n")
    # per-layer integer bias scales: chain bias = b_int8 * BIAS_SCALE_<layer>
    for i, key in enumerate(("FC", "CONVT1", "CONVT2", "CONVT3", "CONV1X1")):
        parts.append(f"    int256 internal constant BIAS_SCALE_{key} = "
                     f"{layers[i]['b_scale_int']};\n")
    # chain logit = float logit * C, C = 128 / prod(weight scales)
    prod_s = 1.0
    for L in layers:
        prod_s *= L["w_scale"]
    C = 128.0 / prod_s
    cuts = [round(t * C) for t in Y_THRESHOLDS]
    for i, c in enumerate(cuts):
        parts.append(f"    int256 internal constant THRESH{i} = {c};\n")
    parts.append(f"    bool internal constant TRAINED = {'true' if trained else 'false'};\n")
    parts.append("}\n")
    os.makedirs(os.path.dirname(SOL_OUT), exist_ok=True)
    with open(SOL_OUT, "w") as f:
        f.write("".join(parts))
    print(f"WeightData.sol: raw {raw_total}B -> huff {comp_total}B "
          f"({100 * comp_total / raw_total:.0f}%), trained={trained}")

    side = {"thresholds": cuts, "scale_C": C, "layers": [{
        "name": L["name"], "b_scale_int": L["b_scale_int"],
        "w": L["w"].flatten().tolist(),
        "shape": list(L["w"].shape), "b": L["b"].flatten().tolist()}
        for L in layers]}
    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    json.dump(side, open(JSON_OUT, "w"))
    print(f"sidecar -> {JSON_OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="ckpt_qat.pt path")
    ap.add_argument("--random", action="store_true")
    a = ap.parse_args()
    if a.random:
        torch.manual_seed(7)
        g = Generator(bn=True).train()
        from model import sample_latent, cond_vector
        for _ in range(3):                                # settle BN stats
            g(sample_latent(64), cond_vector(torch.randint(0, 21, (64,))))
        q = QATGenerator(fold_bn(g))
        trained = False
    else:
        st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        q = QATGenerator(Generator(bn=False))
        q.load_state_dict(st["ema"] if "ema" in st else st["G"])
        trained = True
    emit(q.export_int(), trained)


if __name__ == "__main__":
    main()
