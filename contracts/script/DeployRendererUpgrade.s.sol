// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import { Script, console2 } from "forge-std/Script.sol";
import { FatPunksRenderer } from "../src/FatPunksRenderer.sol";
import { ICryptopunksData } from "../src/interfaces/ICryptopunksData.sol";

/// @notice Renderer-only mainnet deployment helper.
///         Deploys a new FatPunksRenderer linked against the existing
///         WeightData library. It does NOT call setRenderer(), does NOT touch
///         FatPunks, and does NOT lock the renderer.
///         The script is gated by an explicit confirmation inside run().
contract DeployRendererUpgrade is Script {
    address constant PUNKS_DATA = 0x16F5A35647D6F03D5D3da7b35409D65ba03aF3B2;

    function run() external returns (FatPunksRenderer renderer) {
        require(
            vm.envOr("I_UNDERSTAND_RENDERER_UPGRADE", uint256(0)) == 1,
            "set I_UNDERSTAND_RENDERER_UPGRADE=1"
        );

        vm.startBroadcast();

        renderer = new FatPunksRenderer(ICryptopunksData(PUNKS_DATA));

        vm.stopBroadcast();

        console2.log("FatPunksRenderer:", address(renderer));
        console2.log("PUNKS:           ", address(renderer.PUNKS()));
        console2.log("NOTE: renderer deployed only; setRenderer is a separate manual step.");
    }
}
