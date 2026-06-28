# Admin

This page summarizes the public deployment state and the administrative surfaces
that remain on mainnet.

## Mainnet Contracts

| Contract | Address |
| --- | --- |
| FatPunks | `0x40b832368613B74e6114f01c9256BcCeF401894C` |
| Active renderer | `0x237E369057b7cCC5fdb568D6bD2B3Be3e29c3c49` |
| WeightData | `0x26617126f82A9901E077F71A592e5E0A853Ca0D6` |
| Canonical CryptoPunksData | `0x16F5A35647D6F03D5D3da7b35409D65ba03aF3B2` |
| Canonical SeaDrop | `0x00005EA00Ac477B1030CE78506496e8C2dE24bf5` |

Public mint configuration is managed through OpenSea Drops / SeaDrop. The token
contract uses SeaDrop for minting and keeps token id assignment deterministic.

## Renderer Status

The renderer lock is enforced by the contract. While unlocked, `setRenderer()`
can set a replacement renderer. After locking, the renderer/art path is frozen.

Renderer changes cannot change token symbol, minting, ownership, fatness state,
feeding rules, or SeaDrop configuration.

## Admin Surfaces

The remaining owner/admin powers are:

- update the renderer until `lockRenderer()` freezes the art path
- lock the renderer permanently
- set the feeding cooldown
- withdraw ETH held by the token contract, if any
- update royalties, contract/drop metadata, allowed SeaDrop configuration,
  public mint/drop configuration, payout and fee-recipient settings, and the
  transfer validator through inherited SeaDrop surfaces
- transfer or renounce ownership
- manage Lifebuoy rescue flags until the relevant rescue locks are set

These powers are separate from token ownership. Transfers preserve stored
fatness state and cooldown timestamps.

## Configuration

Deploy keys, private environment files, and private RPC keys must not be
tracked. Local `.env` files are ignored. Private runtime configuration belongs
outside tracked source.
