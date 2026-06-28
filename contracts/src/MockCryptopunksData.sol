// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import { ICryptopunksData } from "./interfaces/ICryptopunksData.sol";

/// @notice Test-only stand-in for the mainnet CryptopunksData contract.
///         Real punk RGBA bytes (from the official punks composite) are
///         loaded via setPunk in test setup. NEVER deployed to mainnet —
///         there the renderer points at 0x16F5A35647D6F03D5D3da7b35409D65ba03aF3B2.
contract MockCryptopunksData is ICryptopunksData {
    mapping(uint16 => bytes) private _img;
    mapping(uint16 => string) private _attributes;

    function setPunk(uint16 index, bytes calldata rgba) external {
        require(rgba.length == 2304, "bad len");
        _img[index] = rgba;
    }

    function setPunkAttributes(uint16 index, string calldata attributes) external {
        _attributes[index] = attributes;
    }

    function punkImage(uint16 index) external view returns (bytes memory) {
        bytes memory d = _img[index];
        require(d.length == 2304, "punk not loaded");
        return d;
    }

    function punkImageSvg(uint16) external pure returns (string memory) {
        revert("not in mock");
    }

    function punkAttributes(uint16 index) external view returns (string memory) {
        string memory attributes = _attributes[index];
        if (bytes(attributes).length == 0) return "Mock";
        return attributes;
    }
}
