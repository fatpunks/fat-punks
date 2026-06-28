# Reproducibility

These commands train, export, and verify the renderer artifacts. Commands are
shown from the repository root unless a `cd` command is included.

## Float Training

```sh
cd ml
python3 train.py --phase float --epochs 500 --resume
```

The float model uses WGAN-GP with five critic steps, batch size 128, Adam
`2e-4`, and EMA tracking. Sample grids should show smooth slim-to-huge body
growth across fatness levels without bucket collapse.

## Quantization-Aware Training

```sh
cd ml
python3 train.py --phase qat --epochs 200
```

QAT loads the float EMA, folds batch normalization, and trains the int8
fake-quant model. The exported model should remain visually close to the float
model while matching the integer grid used by the renderer.

## Export

```sh
cd ml
python3 export.py --ckpt out/ckpt_qat.pt
```

Export writes the renderer weight library and a JSON sidecar used by the Python
mirror. The export step also checks the Huffman round trip.

## Contract Compile And Size Check

```sh
cd contracts/evm-harness
node compile.js
```

The trained renderer, token contract, and `WeightData` library must remain under
the EIP-170 deployed bytecode size limit.

## Byte-Exact Verification

```sh
cd contracts/evm-harness
node run.js --seeds 300
cd ../../ml
python3 verify/pure_forward.py --check
```

The verifier compares the pure-Python integer mirror against EVM execution of
`bodyLevels(tokenId, fatnessLevel)`. A passing run proves the Python and EVM
forward passes agree byte-for-byte for the checked token/level pairs.

## Visual Review

```sh
cd ml
python3 verify/render_strip.py
```

The visual strip review checks that bodies progress coherently from slim to huge
and that skin recoloring remains consistent across source-punk types.

## Build Checks

```sh
cd contracts
forge test -vv
forge build --sizes
```

These checks cover contract behavior and contract size reporting.
Administrative surfaces are listed in [admin.md](admin.md).
