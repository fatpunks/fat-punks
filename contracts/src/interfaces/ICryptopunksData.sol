// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/// @notice On-chain CryptoPunks pixel/attribute store by Larva Labs.
///         Mainnet: 0x16F5A35647D6F03D5D3da7b35409D65ba03aF3B2
interface ICryptopunksData {
    /// @return 24*24*4 RGBA bytes, row-major; alpha 0 = transparent.
    function punkImage(uint16 index) external view returns (bytes memory);

    function punkImageSvg(uint16 index) external view returns (string memory);

    function punkAttributes(uint16 index) external view returns (string memory);
}
