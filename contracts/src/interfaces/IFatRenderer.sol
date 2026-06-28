// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

interface IFatRenderer {
    /// @notice Full ERC721 metadata for a punk at a given fatness level.
    function tokenURI(uint256 tokenId, uint8 fatLevel)
        external view returns (string memory);

    /// @notice Raw SVG — also used by the dApp's fatness-preview slider.
    function svg(uint256 tokenId, uint8 fatLevel)
        external view returns (string memory);
}
