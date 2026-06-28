// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import { Script } from "forge-std/Script.sol";
import { FatPunksRenderer } from "../src/FatPunksRenderer.sol";
import { MockCryptopunksData } from "../src/MockCryptopunksData.sol";
import { ICryptopunksData } from "../src/interfaces/ICryptopunksData.sol";
import { LibString } from "solady/src/utils/LibString.sol";

/// @notice Dumps bodyLevels for a deterministic seed sweep so the pure-Python
///         mirror can verify byte-exactness through Foundry's EVM as well.
///         (The repo also ships an ethereumjs harness that does the same.)
contract ParityDump is Script {
    using LibString for uint256;

    function run() external {
        FatPunksRenderer r =
            new FatPunksRenderer(ICryptopunksData(address(new MockCryptopunksData())));
        // bodyLevels is pure — no punk data needed for the parity sweep.
        string memory json = "{";
        uint64 rng = 12345;
        for (uint256 i = 0; i < 300; i++) {
            unchecked {
                rng = rng * 6364136223846793005 + 1442695040888963407;
            }
            uint256 id = rng % 10_000;
            uint8 lvl = uint8(rng % 21);
            bytes memory lv = r.bodyLevels(id, lvl);
            json = string(abi.encodePacked(
                json, i == 0 ? "" : ",", '"', id.toString(), "_",
                uint256(lvl).toString(), '":"',
                LibString.toHexStringNoPrefix(lv), '"'));
        }
        json = string(abi.encodePacked(json, "}"));
        vm.writeFile("parity_out/evm_results.json", json);
    }
}
