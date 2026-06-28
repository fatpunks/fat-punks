# Mechanism

This document summarizes the contract, renderer, and reproducibility mechanism.

## Burrito Feeding And Transfer Behavior

Fatness changes only through explicit burrito feeding. `feed(tokenId)` adds one
fatness level for the current owner, respects the per-token cooldown, and caps
at level 20. `feedAll(uint256[])` lets the app batch known owned ids and skips
ineligible ids instead of reverting the entire batch.

Transfers preserve the stored fatness level and cooldown timestamp. Buying or
selling a Fat Punk does not change its fatness.

## Deterministic Mint Order

Public minting uses OpenSea Drops / SeaDrop. Each mint ordinal maps to a token
id with the affine permutation:

```text
tokenId = (mintOrdinal * 7321 + 7900) % 10000
```

Because `7321` is coprime with `10000`, every id from 0 through 9999 appears
exactly once. This keeps token ids aligned with source punk indexes while
preventing public minters from directly choosing exact ids through the mint
flow. The order is public and deterministic; it is not hidden randomness.

ERC721A spot mints support the permuted ids. Token `0` remains the only
sequential ERC721A mint slot, and ids `1..9999` are minted with `_mintSpot`.

## On-Chain Metadata And Renderer Locking

The token contract stores fatness state. The renderer generates SVG metadata on
demand from `(tokenId, fatnessLevel)` and the canonical CryptoPunksData source
image.

The renderer can be updated until `lockRenderer()` is called. After locking,
the renderer/art path is frozen. Renderer updates are separate from minting,
ownership, fatness state, feeding rules, and SeaDrop configuration.

## Integer Renderer And Byte-Exact Verification

The renderer is designed so the on-chain forward pass can be mirrored exactly
off-chain. Latents are uniform signed int8 values derived from `keccak256`.
Conditioning is `level * 12 - 120`, replicated across eight inputs. The model
uses integer mul/add/ReLU/compare operations, so Python big integers can
reproduce the EVM result byte-for-byte.

Quantization-aware training keeps weights, biases, activations, and threshold
decode on an exact integer grid. Biases use per-layer integer scale constants so
later layers retain useful dynamic range without introducing division or
floating-point math on-chain.

## Absolute Body Thresholds

The body decoder uses fixed absolute thresholds instead of per-image contrast
normalization. Fixed thresholds preserve calibrated shade buckets across
different body sizes and keep the on-chain decoder simple: four signed
comparisons per pixel.

This choice fits the project because fatness changes the ratio of body pixels to
background pixels. Per-image normalization would make that ratio harder to
represent consistently.

## Weight Storage And EIP-170

The trained renderer weights are stored in the deployed `WeightData` library.
Keeping the weight blob and decoder in a separate library keeps the renderer and
token contracts under the EIP-170 bytecode size limit while preserving a fully
on-chain metadata path.

`WeightData` exposes public getters for the renderer to read through the linked
library. The network still lives in contract bytecode; there are no off-chain
image or metadata servers in the token URI path.

## Body Generation And Skin Recoloring

The network generates a grayscale five-bucket body conditioned on fatness. The
renderer recolors those buckets with a four-shade ramp derived from the punk
source image, then composites the generated body behind the original head.

Generating body shape separately from skin color keeps the model compact while
preserving source-punk palette variation.

## Weight Encoding

`export.py` serializes int8 weights with a canonical Huffman code. Layer layouts
match the EVM renderer: dense weights are row-major, transposed convolution
weights are pair-packed, and 1x1 convolution weights include the padding needed
for efficient memory reads.

The Python verifier and EVM renderer share the exported constants and decode
rules so byte-exact checks cover the deployed arithmetic path.

## Verification Tooling

The repository includes both Foundry tests and an independent ethereumjs harness.
The harness compiles with solc-js and executes renderer/token behavior in an EVM
environment, which gives another check on bytecode size, token behavior, and
byte-exact renderer output.

Foundry remains the standard contract test tool where it is available.

## Public Safety Boundaries

Admin disclosures are documented separately from executable code. Local
environment files, signing credentials, and non-public RPC credentials must
remain outside tracked source.
