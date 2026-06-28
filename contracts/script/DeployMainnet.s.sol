// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import { Script, console2 } from "forge-std/Script.sol";
import { FatPunks } from "../src/FatPunks.sol";
import { FatPunksRenderer } from "../src/FatPunksRenderer.sol";
import { ICryptopunksData } from "../src/interfaces/ICryptopunksData.sol";
import { IFatRenderer } from "../src/interfaces/IFatRenderer.sol";

/// @notice Mainnet deployment script for the token and initial renderer.
///         The script is gated by an explicit confirmation inside run().
///         Read docs/admin.md before use.
contract DeployMainnet is Script {
    address constant PUNKS_DATA = 0x16F5A35647D6F03D5D3da7b35409D65ba03aF3B2;
    address constant SEADROP = 0x00005EA00Ac477B1030CE78506496e8C2dE24bf5;

    function run() external {
        require(
            vm.envOr("I_UNDERSTAND_THIS_IS_MAINNET", uint256(0)) == 1,
            "set I_UNDERSTAND_THIS_IS_MAINNET=1"
        );

        vm.startBroadcast();

        FatPunksRenderer renderer =
            new FatPunksRenderer(ICryptopunksData(PUNKS_DATA));

        address[] memory allowed = new address[](1);
        allowed[0] = SEADROP;
        FatPunks nft = new FatPunks(allowed, IFatRenderer(address(renderer)));

        nft.setMaxSupply(10_000);
        // Public mint config is managed on SeaDrop/OpenSea, not on this token.

        vm.stopBroadcast();

        console2.log("FatPunksRenderer:", address(renderer));
        console2.log("FatPunks:        ", address(nft));
        console2.log("Affine A:        ", nft.AFFINE_A());
        console2.log("Affine B:        ", nft.AFFINE_B());
        console2.log("");
        console2.log("== POST-DEPLOY CHECKLIST (human, in order) ==");
        console2.log("1. Sanity: cast call renderer 'svg(uint256,uint8)' 0 20");
        console2.log("2. Configure SeaDrop/OpenSea public mint + payout");
        console2.log("3. tokenURI renders in a marketplace metadata preview");
        console2.log("4. Royalties: nft.setRoyaltyInfo((address,uint96))");
        console2.log("5. After confidence period: nft.lockRenderer()");
        console2.log("6. Review Lifebuoy rescue flags before locking");
    }
}
