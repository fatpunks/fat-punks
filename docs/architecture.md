# Architecture

## The read path (one free eth_call)

```
tokenURI(id)                              FatPunks.sol
  └─ renderer.tokenURI(id, fatLevel[id])  FatPunksRenderer.sol
       ├─ seed = id  ──keccak──▶ 32 int8 latents
       ├─ cond  = level·12 − 120 (×8)
       ├─ forward pass (all int256, weights from bytecode):
       │    dense 40→256 → ReLU            (16ch · 4×4)
       │    convT 16→16, 4×4 s2 → ReLU     (8×8)
       │    convT 16→8,  4×4 s2 → ReLU     (16×16)
       │    convT 8→8,   4×4 s2 → ReLU     (32×32)
       │    conv1×1 8→1                    (32×32 logits)
       ├─ bucketize: 4 signed comparisons vs WeightData.THRESH0..3
       │    → 1024 body shade levels (0=bg, 1..4 = dark→light)
       ├─ skin ramp: probe punk RGBA @ offset 2244 → 4 shades
       │    (c·9/25, c·18/25, c, min(255, c·5/4))
       ├─ head: PUNKS.punkImage(id) — 24×24 RGBA, original alpha
       └─ SVG: 32×40 grid of <rect> (body at y+8, head at x+4), body behind head, bg #638596
```

`bodyLevels(id, level)` exposes the raw 1024-byte bucket map — this is the
quantity verified byte-for-byte against the pure-Python mirror.

## Why the math is exactly mirrorable

After exact-grid QAT, every on-chain activation equals the float model's
activation times `128/∏(weight scales up to that layer)` — ReLU commutes
with positive scaling, biases are stored pre-multiplied onto the same grid,
and the absolute-threshold decode adds **no division**. The whole forward
pass is integer mul/add/ReLU/compare, so Python bigints reproduce the EVM
bit-for-bit. There is no float anywhere on the verification path.

## Weight storage

`export.py` serializes int8 weights with a canonical Huffman code
(package-merge, max code length 10; 61-byte header). Layer layouts match
the engine adapted from Artificial After All: dense row-major, transposed
conv pair-packed `[ic][oc][kh][kw]`, conv1×1 with a 32-byte mload pad.
Biases are int8 with a per-layer integer scale constant (chain adds
`b8 × BIAS_SCALE_k`). The blob lives in `WeightData.sol`, whose `get_*`
functions are public — so the ~16 KB of weights + the Huffman decoder
deploy as their own library contract, DELEGATECALLed by the renderer.
Both deployed artifacts sit under EIP-170.

## Token mechanics

ERC721A with `_startTokenId = 0`, `_sequentialUpTo = 0`: OpenSea
Drops / SeaDrop mints use a deterministic affine-scrambled order,
`tokenId = (mintOrdinal * 7321 + 7900) % 10000`. `7321` is coprime with
`10000`, so every id 0..9999 appears exactly once. Token 0 is the one
sequential ERC721A mint; ids 1..9999 are spot mints. A Fat Punk's `tokenId`
is still the matching CryptoPunk index, but public users cannot choose exact
ids or snipe obvious early sequential ids. The order is public and
deterministic, not true hidden randomness. Fatness changes only through
burrito feeding:
`feed(tokenId)` is current-owner only, adds exactly one level, respects the
cooldown, and caps at 20. `feedAll(uint256[])` batch-feeds eligible owned
punks and skips ineligible ids instead of reverting the whole batch.
Transfers preserve the stored `FatState{uint8,uint64}` but never increase it.
The renderer address is swappable until `lockRenderer()` — after that the art
is frozen forever.

## Source Data Across Environments

Local and testnet environments may use mock source data for verification.
Mainnet rendering uses canonical CryptoPunksData at
`0x16F5A35647D6F03D5D3da7b35409D65ba03aF3B2`.
