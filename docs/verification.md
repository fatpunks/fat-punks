# Verification

This document records the renderer characteristics and verification surface.

## Model Behavior

- Fatness conditioning is monotonic across all 21 levels. Silhouettes grow from
  slim to enormous while staying coherent.
- Per-token `keccak256` latents give each punk a stable body shape for every
  `(tokenId, fatnessLevel)` pair.
- The body style follows the project sprite language: flat mid shades, darker
  edge shading, belly-fold strokes on larger bodies, and small highlight
  details.
- Skin compositing preserves source-punk palette variation by recoloring the
  generated body from the punk source image.

## Known Visual Limitations

- Slim-level necks can appear narrow under some heads.
- Edge outlines are softer than a fully hand-drawn one-pixel outline.
- Largest bodies use suggestive shading rather than highly detailed folds.

These limits are accepted for the deliberately compact, on-chain renderer and
the crude punk aesthetic.

## Training Summary

- Float training used WGAN-GP until body growth, shade separation, and identity
  variation reached the target style.
- Quantization-aware training preserved visual quality while aligning the model
  to the exact integer grid used by the renderer.
- Exported weights, bias scales, decode thresholds, and Huffman data are checked
  by the verifier before use.

## Verification Summary

- The Python integer mirror and EVM execution agree byte-for-byte for checked
  `bodyLevels(tokenId, fatnessLevel)` outputs.
- The trained contracts are under the EIP-170 deployed bytecode size limit.
- SVG and tokenURI rendering execute end-to-end through the renderer.
- Contract tests cover SeaDrop quantity minting, deterministic token assignment,
  public exact-index claim removal, owner-only burrito feeding, cooldowns,
  transfer persistence, batch feeding skips, tokenURI state consistency, and
  renderer locking.

## Check Commands

Run:

```sh
cd contracts
forge test -vv
forge build --sizes
```

The ethereumjs harness in `contracts/evm-harness` remains an additional
independent EVM check for renderer output and token behavior.
