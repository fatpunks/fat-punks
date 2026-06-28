// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import { Script, console2 } from "forge-std/Script.sol";
import { stdJson } from "forge-std/StdJson.sol";
import { FatPunks } from "../src/FatPunks.sol";
import { FatPunksRenderer } from "../src/FatPunksRenderer.sol";
import { MockCryptopunksData } from "../src/MockCryptopunksData.sol";
import { ICryptopunksData } from "../src/interfaces/ICryptopunksData.sol";
import { IFatRenderer } from "../src/interfaces/IFatRenderer.sol";

/// @notice Testnet deployment helper.
///         Uses MockCryptopunksData because canonical CryptoPunksData is
///         available on mainnet only.
contract DeploySepolia is Script {
    using stdJson for string;

    address constant SEADROP = 0x00005EA00Ac477B1030CE78506496e8C2dE24bf5;

    function run() external {
        vm.startBroadcast();

        MockCryptopunksData mock = new MockCryptopunksData();
        string memory json = vm.readFile("test/fixtures/punks.json");
        uint16[9] memory ids = [uint16(0), 1, 2, 3, 4, 5, 117, 372, 635];
        for (uint256 i = 0; i < ids.length; i++) {
            bytes memory rgba = json.readBytes(
                string(abi.encodePacked(".", vm.toString(uint256(ids[i])))));
            mock.setPunk(ids[i], rgba);
        }

        FatPunksRenderer renderer =
            new FatPunksRenderer(ICryptopunksData(address(mock)));

        address[] memory allowed = new address[](1);
        allowed[0] = SEADROP;
        FatPunks nft = new FatPunks(allowed, IFatRenderer(address(renderer)));

        nft.setMaxSupply(10_000);

        vm.stopBroadcast();

        console2.log("MockCryptopunksData:", address(mock));
        console2.log("FatPunksRenderer:  ", address(renderer));
        console2.log("FatPunks:          ", address(nft));
        console2.log("NOTE: affine mint #0 assigns token id 7900.");
        console2.log("NOTE: token 0 appears at affine mint ordinal 100.");
        console2.log("NOTE: fixture ids 0-5/117/372/635 are mock-render fixtures.");
    }
}
