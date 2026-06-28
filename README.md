# Fat Punks

Fat Punks is a 10,000-token on-chain art project where every Fat Punk can be
fed burritos until it becomes an absolute unit.

Public minting happens through OpenSea Drops / SeaDrop. The website at
[fatpunks.xyz](https://fatpunks.xyz) is the burrito station.

## What It Is

- Fully on-chain art and metadata. The token contract stores fatness state, and
  the renderer generates SVG metadata on demand.
- Burrito feeding. `feed(tokenId)` adds one fatness level, respects a per-token
  cooldown, and caps at level 20. `feedAll(uint256[])` feeds eligible owned
  tokens and skips ineligible ids.
- Fatness sticks. Transfers preserve the stored fatness level and cooldown
  timestamp; transfers never increase fatness.
- Deterministic mint order. SeaDrop quantity mints use:

  ```text
  tokenId = (mintOrdinal * 7321 + 7900) % 10000
  ```

  The affine order is public, deterministic, and covers every id 0 through
  9999 exactly once.
- Token ids map 1:1 to original punk indexes for trait lookup and art-source
  composition. Public minters cannot choose exact ids through the mint flow.

The neural body renderer uses a compact int8 model whose weights live in the
deployed `WeightData` library. It draws the body from `(tokenId, fatnessLevel)`,
recolors it from the punk source image, and composites the result into an SVG.
There are no off-chain image servers or IPFS metadata dependencies in the token
URI path.

## Mainnet Contracts

| Contract | Address |
| --- | --- |
| FatPunks | `0x40b832368613B74e6114f01c9256BcCeF401894C` |
| Active renderer | `0x237E369057b7cCC5fdb568D6bD2B3Be3e29c3c49` |
| WeightData | `0x26617126f82A9901E077F71A592e5E0A853Ca0D6` |
| Canonical CryptoPunksData | `0x16F5A35647D6F03D5D3da7b35409D65ba03aF3B2` |

Token symbol: `FATPUNK`.

The renderer lock is enforced by the contract. Until `lockRenderer()` is
called, the owner can update the renderer. After locking, the art path is
frozen. Other owner powers are documented in [admin](docs/admin.md). The
deployment record is in [docs/contracts.md](docs/contracts.md).

## Repository Layout

```text
contracts/  Foundry contracts, tests, scripts, and EVM harness
docs/       architecture, mechanism, contracts, admin, and verification docs
ml/         training, quantization, export, and renderer verification
```

## Contract Setup

```sh
cd contracts
forge build --sizes
forge test -vv
```

For contract tests and local scripts, copy `contracts/.env.example` to
`contracts/.env` and fill local values. Local `.env` files are ignored. Private
runtime configuration belongs outside tracked source. See [admin](docs/admin.md).

## Verification Gates

Reproducibility and verification commands are in
[docs/reproducibility.md](docs/reproducibility.md):

- byte-exact Python to EVM agreement
- contract size checks under EIP-170
- renderer and token behavior tests
- mainnet-fork render checks against canonical CryptoPunksData
- human review of rendered art strips

## License

MIT. See [LICENSE](LICENSE).

MIT applies only to original Fat Punks repository code, scripts, documentation,
and project-authored materials. Fat Punks is an independent on-chain art
experiment and is not affiliated with, sponsored by, endorsed by, or officially
connected to CryptoPunks, Larva Labs, Yuga Labs, Infinite Node Foundation, NODE
Foundation, or any official CryptoPunks rights holder. Nothing in this repo
grants any license to CryptoPunks, CryptoPunks trademarks, third-party NFT
artwork, third-party deployed contracts, or materials owned by their respective
rights holders; third-party names, marks, and references remain property of
their respective owners.

The renderer architecture was adapted from the MIT-licensed
[Artificial After All](https://github.com/hanrgba/artificialafterall). The
Fat Punks token mechanics, training pipeline, exported weights, compositor, and
fatness decode are original to this project.
