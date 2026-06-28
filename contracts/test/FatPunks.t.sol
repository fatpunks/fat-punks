// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import { Test } from "forge-std/Test.sol";
import { ERC721A__IERC721Receiver } from "ERC721A/ERC721A.sol";
import { FatPunks } from "../src/FatPunks.sol";
import { IFatRenderer } from "../src/interfaces/IFatRenderer.sol";
import { LibString } from "solady/src/utils/LibString.sol";
import { INonFungibleSeaDropToken } from
    "seadrop/interfaces/INonFungibleSeaDropToken.sol";

/// @dev Minimal renderer stub: encodes (id, level) so URI plumbing is testable.
contract StubRenderer is IFatRenderer {
    using LibString for uint256;

    function tokenURI(uint256 id, uint8 lvl)
        external pure returns (string memory)
    {
        return string(abi.encodePacked("stub:", id.toString(), ":",
            uint256(lvl).toString()));
    }

    function svg(uint256, uint8) external pure returns (string memory) {
        return "<svg/>";
    }
}

contract TestFatPunks is FatPunks {
    constructor(address[] memory allowedSeaDrop, IFatRenderer renderer_)
        FatPunks(allowedSeaDrop, renderer_)
    {}

    function unsafeSetFattenCooldown(uint64 cooldown) external {
        fattenCooldown = cooldown;
    }
}

contract SequentialFatPunksForGas is FatPunks {
    constructor(address[] memory allowedSeaDrop, IFatRenderer renderer_)
        FatPunks(allowedSeaDrop, renderer_)
    {}

    function _sequentialUpTo() internal view virtual override returns (uint256) {
        return PUNK_SUPPLY - 1;
    }

    function mintSeaDrop(address minter, uint256 quantity)
        external
        virtual
        override
        nonReentrant
    {
        _onlyAllowedSeaDrop(msg.sender);
        if (_totalMinted() + quantity > maxSupply()) {
            revert MintQuantityExceedsMaxSupply(
                _totalMinted() + quantity,
                maxSupply()
            );
        }
        _safeMint(minter, quantity);
    }
}

contract ReceiverProbe is ERC721A__IERC721Receiver {
    FatPunks public immutable nft;
    uint256 public expectedBalance;
    uint256 public observedFirstBalance;
    uint256 public calls;

    constructor(FatPunks nft_) {
        nft = nft_;
    }

    function setExpectedBalance(uint256 value) external {
        expectedBalance = value;
    }

    function onERC721Received(
        address,
        address,
        uint256 tokenId,
        bytes calldata
    ) external returns (bytes4) {
        require(msg.sender == address(nft), "wrong nft");
        if (calls == 0) {
            observedFirstBalance = nft.balanceOf(address(this));
            require(observedFirstBalance == expectedBalance, "partial batch");
        }
        require(nft.ownerOf(tokenId) == address(this), "not owner");
        ++calls;
        return this.onERC721Received.selector;
    }
}

contract FatPunksTest is Test {
    // Mirror of FatPunks.Fed — solc 0.8.17 cannot emit a qualified event
    // from another contract (that syntax landed in 0.8.21).
    event Fed(uint256 indexed tokenId, uint8 newLevel);

    FatPunks nft;
    StubRenderer stub;
    address seaDrop = address(0x5EA);
    address alice = address(0xA11CE);
    address bob = address(0xB0B);

    function setUp() public {
        stub = new StubRenderer();
        address[] memory allowed = new address[](1);
        allowed[0] = seaDrop;
        nft = new TestFatPunks(allowed, IFatRenderer(address(stub)));
        nft.setMaxSupply(10_000);
        vm.deal(alice, 100 ether);
        vm.deal(bob, 100 ether);
    }

    // ─────────────────────────────── minting ──────────────────────────────

    function test_affineConstantsAreCoprimeAndFullDomainBijection()
        public view
    {
        bool[] memory seen = new bool[](10_000);
        uint256 sum;

        assertEq(nft.AFFINE_A(), 7_321);
        assertEq(nft.AFFINE_B(), 7_900);
        assertEq(_gcd(nft.AFFINE_A(), 10_000), 1);

        for (uint256 ordinal = 0; ordinal < 10_000; ordinal++) {
            uint256 tokenId = nft.permutedTokenId(ordinal);
            assertLt(tokenId, 10_000);
            assertFalse(seen[tokenId], "duplicate token id");
            seen[tokenId] = true;
            sum += tokenId;
        }

        assertEq(sum, 49_995_000);
    }

    function test_firstAffineAssignments() public view {
        assertEq(nft.permutedTokenId(0), 7900);
        assertEq(nft.permutedTokenId(1), 5221);
        assertEq(nft.permutedTokenId(2), 2542);
        assertEq(nft.permutedTokenId(3), 9863);
        assertEq(nft.permutedTokenId(4), 7184);
    }

    function test_mintSeaDropAllowedMintsQuantity() public {
        vm.prank(seaDrop);
        nft.mintSeaDrop(alice, 3);
        assertEq(nft.ownerOf(nft.permutedTokenId(0)), alice);
        assertEq(nft.ownerOf(nft.permutedTokenId(1)), alice);
        assertEq(nft.ownerOf(nft.permutedTokenId(2)), alice);
        assertEq(nft.totalSupply(), 3);
        assertEq(nft.balanceOf(alice), 3);

        (
            uint256 minterNumMinted,
            uint256 currentTotalSupply,
            uint256 maxSupply
        ) = nft.getMintStats(alice);
        assertEq(minterNumMinted, 3);
        assertEq(currentTotalSupply, 3);
        assertEq(maxSupply, 10_000);
    }

    function test_mintSeaDropOnlyAllowedSeaDrop() public {
        vm.prank(alice);
        vm.expectRevert(INonFungibleSeaDropToken.OnlyAllowedSeaDrop.selector);
        nft.mintSeaDrop(alice, 1);
    }

    function test_mintSeaDropStartsAtAffineFirstToken() public {
        uint256 first = nft.permutedTokenId(0);
        vm.prank(seaDrop);
        nft.mintSeaDrop(alice, 1);
        assertEq(first, 7900);
        assertEq(nft.ownerOf(first), alice);
        vm.expectRevert();
        nft.ownerOf(0);
    }

    function test_mintSeaDropMaxSupply() public {
        nft.setMaxSupply(2);
        vm.prank(seaDrop);
        nft.mintSeaDrop(alice, 2);
        assertEq(nft.ownerOf(nft.permutedTokenId(0)), alice);
        assertEq(nft.ownerOf(nft.permutedTokenId(1)), alice);

        vm.prank(seaDrop);
        vm.expectRevert();
        nft.mintSeaDrop(alice, 1);
    }

    function test_publicExactIndexClaimUnavailable() public {
        uint256[] memory ids = new uint256[](2);
        ids[0] = 635;
        ids[1] = 7804;
        vm.prank(alice);
        (bool ok, ) = address(nft).call(
            abi.encodeWithSignature("claim(uint256[])", ids)
        );
        assertFalse(ok);
        vm.expectRevert();
        nft.ownerOf(635);
    }

    function test_tokenZeroMintsAtItsPermutedOrdinal() public {
        vm.prank(seaDrop);
        nft.mintSeaDrop(alice, 100);

        vm.expectRevert();
        nft.ownerOf(0);

        vm.prank(seaDrop);
        nft.mintSeaDrop(bob, 1);

        assertEq(nft.permutedTokenId(100), 0);
        assertEq(nft.ownerOf(0), bob);
        assertEq(nft.totalSupply(), 101);
    }

    function test_fullSupplyPermutationNoDuplicateMintedIdsAndMaxSupply()
        public
    {
        bool[] memory seen = new bool[](10_000);

        for (uint256 minted; minted < 10_000; minted += 100) {
            vm.prank(seaDrop);
            nft.mintSeaDrop(alice, 100);
        }

        assertEq(nft.totalSupply(), 10_000);
        assertEq(nft.balanceOf(alice), 10_000);

        for (uint256 ordinal = 0; ordinal < 10_000; ordinal++) {
            uint256 tokenId = nft.permutedTokenId(ordinal);
            assertFalse(seen[tokenId], "duplicate after mint");
            seen[tokenId] = true;
            assertEq(nft.ownerOf(tokenId), alice);
        }

        vm.prank(seaDrop);
        vm.expectRevert();
        nft.mintSeaDrop(alice, 1);
    }

    function test_contractReceiverSeesFullBatchBeforeFirstCallback() public {
        ReceiverProbe receiver = new ReceiverProbe(nft);
        receiver.setExpectedBalance(3);

        vm.prank(seaDrop);
        nft.mintSeaDrop(address(receiver), 3);

        assertEq(receiver.calls(), 3);
        assertEq(receiver.observedFirstBalance(), 3);
    }

    // ──────────────────────── feeding (v2 mechanic) ───────────────────────

    function _mintOne(address who) internal returns (uint256 id) {
        id = nft.permutedTokenId(nft.totalSupply());
        vm.prank(seaDrop);
        nft.mintSeaDrop(who, 1);
    }

    function _mintMany(address who, uint256 quantity)
        internal returns (uint256[] memory ids)
    {
        uint256 firstOrdinal = nft.totalSupply();
        ids = new uint256[](quantity);
        for (uint256 i = 0; i < quantity; i++) {
            ids[i] = nft.permutedTokenId(firstOrdinal + i);
        }
        vm.prank(seaDrop);
        nft.mintSeaDrop(who, quantity);
    }

    function _ids(uint256 a) internal pure returns (uint256[] memory r) {
        r = new uint256[](1);
        r[0] = a;
    }

    function _ids2(uint256 a, uint256 b)
        internal pure returns (uint256[] memory r)
    {
        r = new uint256[](2);
        r[0] = a;
        r[1] = b;
    }

    function _ids3(uint256 a, uint256 b, uint256 c)
        internal pure returns (uint256[] memory r)
    {
        r = new uint256[](3);
        r[0] = a;
        r[1] = b;
        r[2] = c;
    }

    function test_mintDoesNotFatten() public {
        uint256 id = _mintOne(alice);
        assertEq(nft.fatLevel(id), 0);
    }

    /// @dev The core v2 change: transfers must NOT change fatness.
    function test_transferDoesNotFatten() public {
        uint256 id = _mintOne(alice);
        // feed once so it has a non-zero level to preserve
        vm.prank(alice);
        nft.feed(id);
        assertEq(nft.fatLevel(id), 1);
        // transfer wallet→wallet: level stays put
        vm.prank(alice);
        nft.transferFrom(alice, bob, id);
        assertEq(nft.fatLevel(id), 1);
        vm.prank(bob);
        nft.transferFrom(bob, alice, id);
        assertEq(nft.fatLevel(id), 1);
    }

    function test_feedIncrements() public {
        uint256 id = _mintOne(alice);
        vm.prank(alice);
        nft.feed(id);
        assertEq(nft.fatLevel(id), 1);
    }

    function test_feedEmitsFed() public {
        uint256 id = _mintOne(alice);
        vm.expectEmit(true, false, false, true);
        emit Fed(id, 1);
        vm.prank(alice);
        nft.feed(id);
    }

    function test_feedOnlyOwner() public {
        uint256 id = _mintOne(alice);
        vm.prank(bob);
        vm.expectRevert(FatPunks.NotPunkOwner.selector);
        nft.feed(id);
    }

    function test_approvedOperatorCannotFeed() public {
        uint256 id = _mintOne(alice);
        vm.prank(alice);
        nft.setApprovalForAll(bob, true);
        vm.prank(bob);
        vm.expectRevert(FatPunks.NotPunkOwner.selector);
        nft.feed(id);
    }

    function test_feedCooldown() public {
        uint256 id = _mintOne(alice);
        vm.prank(alice);
        nft.feed(id);                 // first feed: no prior timestamp
        assertEq(nft.fatLevel(id), 1);
        // within 24h: reverts
        vm.prank(alice);
        vm.expectRevert(FatPunks.FeedOnCooldown.selector);
        nft.feed(id);
        // after 24h: succeeds
        vm.warp(block.timestamp + 1 days);
        vm.prank(alice);
        nft.feed(id);
        assertEq(nft.fatLevel(id), 2);
    }

    function test_publicFeedTimingReads() public {
        uint256 id = _mintOne(alice);
        assertEq(nft.MAX_FAT_LEVEL(), 20);
        assertEq(nft.fattenCooldown(), 1 days);
        assertEq(nft.lastFattened(id), 0);

        vm.prank(alice);
        nft.feed(id);
        assertEq(nft.lastFattened(id), block.timestamp);
    }

    function test_setFattenCooldownZeroReverts() public {
        vm.expectRevert(FatPunks.InvalidFattenCooldown.selector);
        nft.setFattenCooldown(0);
        assertEq(nft.fattenCooldown(), 1 days);
    }

    function test_transferPreservesFatnessAndNewOwnerFeedsAfterCooldown()
        public
    {
        uint256 id = _mintOne(alice);
        vm.prank(alice);
        nft.feed(id);
        assertEq(nft.fatLevel(id), 1);

        vm.prank(alice);
        nft.transferFrom(alice, bob, id);
        assertEq(nft.ownerOf(id), bob);
        assertEq(nft.fatLevel(id), 1);

        vm.prank(alice);
        vm.expectRevert(FatPunks.NotPunkOwner.selector);
        nft.feed(id);

        vm.prank(bob);
        vm.expectRevert(FatPunks.FeedOnCooldown.selector);
        nft.feed(id);

        vm.warp(block.timestamp + 1 days);
        vm.prank(bob);
        nft.feed(id);
        assertEq(nft.fatLevel(id), 2);
    }

    function test_feedCapAt20() public {
        uint256 id = _mintOne(alice);
        // accumulate time explicitly — reading block.timestamp inside a
        // warping loop can evaluate stale under via-IR.
        uint256 t = block.timestamp;
        for (uint256 i = 0; i < 20; i++) {
            vm.prank(alice);
            nft.feed(id);
            t += 1 days + 1;
            vm.warp(t);
        }
        assertEq(nft.fatLevel(id), 20);
        // 21st feed reverts at the cap
        vm.prank(alice);
        vm.expectRevert(FatPunks.AlreadyMaxFat.selector);
        nft.feed(id);
    }

    /// @dev feedAll feeds eligible ids and silently skips the rest.
    function test_feedAllPartialBatch() public {
        uint256 feedableId = _mintOne(alice);
        uint256 maxedId = _mintOne(alice);
        uint256 bobId = _mintOne(bob);
        uint256 nonexistentId = nft.totalSupply() + 99;

        // max out maxedId over 20 days (explicit time accumulator)
        uint256 t = block.timestamp;
        for (uint256 i = 0; i < 20; i++) {
            vm.prank(alice);
            nft.feed(maxedId);
            t += 1 days + 1;
            vm.warp(t);
        }
        assertEq(nft.fatLevel(maxedId), 20);

        // batch: feedable, maxed, not owned, nonexistent.
        uint256[] memory batch = new uint256[](4);
        batch[0] = feedableId;
        batch[1] = maxedId;
        batch[2] = bobId;
        batch[3] = nonexistentId;
        vm.prank(alice);
        nft.feedAll(batch);

        assertEq(nft.fatLevel(feedableId), 1); // fed
        assertEq(nft.fatLevel(maxedId), 20);   // unchanged (maxed)
        assertEq(nft.fatLevel(bobId), 0);      // unchanged (not alice's)
    }

    /// @dev feedAll respects cooldown per-token without reverting the batch.
    function test_feedAllSkipsCooldown() public {
        uint256 id = _mintOne(alice);
        vm.prank(alice);
        nft.feed(id);                    // level 1, just fed
        assertEq(nft.fatLevel(id), 1);
        // immediate feedAll: still on cooldown → skipped, no revert
        vm.prank(alice);
        nft.feedAll(_ids(id));
        assertEq(nft.fatLevel(id), 1);
        // after cooldown: feedAll feeds it
        vm.warp(block.timestamp + 1 days);
        vm.prank(alice);
        nft.feedAll(_ids(id));
        assertEq(nft.fatLevel(id), 2);
    }

    function test_feedAllMultipleOwnedEmitsFedEvents() public {
        uint256[] memory ids = _mintMany(alice, 3);
        uint256 first = ids[0];
        uint256 second = ids[1];
        uint256 third = ids[2];

        vm.expectEmit(true, false, false, true);
        emit Fed(first, 1);
        vm.expectEmit(true, false, false, true);
        emit Fed(second, 1);
        vm.expectEmit(true, false, false, true);
        emit Fed(third, 1);
        vm.prank(alice);
        nft.feedAll(_ids3(first, second, third));

        assertEq(nft.fatLevel(first), 1);
        assertEq(nft.fatLevel(second), 1);
        assertEq(nft.fatLevel(third), 1);
    }

    function test_feedAllDuplicateTokenIdsDoNotDoubleFeed() public {
        uint256 id = _mintOne(alice);

        vm.expectEmit(true, false, false, true);
        emit Fed(id, 1);
        vm.prank(alice);
        nft.feedAll(_ids2(id, id));

        assertEq(nft.fatLevel(id), 1);
    }

    function test_feedAllDuplicateTokenIdsDoNotDoubleFeedWithZeroCooldown()
        public
    {
        uint256 id = _mintOne(alice);
        TestFatPunks(address(nft)).unsafeSetFattenCooldown(0);

        vm.expectEmit(true, false, false, true);
        emit Fed(id, 1);
        vm.prank(alice);
        nft.feedAll(_ids2(id, id));

        assertEq(nft.fatLevel(id), 1);
    }

    // ──────────────────────────── tokenURI / admin ────────────────────────

    function test_tokenURIPlumbing() public {
        uint256 id = _mintOne(alice);
        assertEq(nft.tokenURI(id), "stub:7900:0");
        vm.prank(alice);
        nft.feed(id);
        assertEq(nft.tokenURI(id), "stub:7900:1");
    }

    function test_tokenURINonexistent() public {
        vm.expectRevert();
        nft.tokenURI(404);
    }

    function test_rendererLock() public {
        nft.setRenderer(IFatRenderer(address(0xDEAD)));
        nft.lockRenderer();
        vm.expectRevert(FatPunks.RendererIsLocked.selector);
        nft.setRenderer(IFatRenderer(address(stub)));
    }

    function test_adminGating() public {
        vm.startPrank(alice);
        vm.expectRevert();
        nft.setFattenCooldown(1);
        vm.expectRevert();
        nft.setRenderer(IFatRenderer(address(0)));
        vm.expectRevert();
        nft.withdraw(payable(alice));
        vm.stopPrank();
    }

    function test_gasSequentialVsPermutedMintQuantityOne() public {
        SequentialFatPunksForGas seq = _newSequentialForGas();

        uint256 g0 = gasleft();
        vm.prank(seaDrop);
        seq.mintSeaDrop(alice, 1);
        uint256 sequentialGas = g0 - gasleft();

        g0 = gasleft();
        vm.prank(seaDrop);
        nft.mintSeaDrop(alice, 1);
        uint256 permutedGas = g0 - gasleft();

        emit log_named_uint("sequential mint q1 gas", sequentialGas);
        emit log_named_uint("permuted mint q1 gas", permutedGas);
    }

    function test_gasSequentialVsPermutedMintQuantityFive() public {
        SequentialFatPunksForGas seq = _newSequentialForGas();

        uint256 g0 = gasleft();
        vm.prank(seaDrop);
        seq.mintSeaDrop(alice, 5);
        uint256 sequentialGas = g0 - gasleft();

        g0 = gasleft();
        vm.prank(seaDrop);
        nft.mintSeaDrop(alice, 5);
        uint256 permutedGas = g0 - gasleft();

        emit log_named_uint("sequential mint q5 gas", sequentialGas);
        emit log_named_uint("permuted mint q5 gas", permutedGas);
    }

    function test_gasSequentialVsPermutedTokenDeploy() public {
        address[] memory allowed = new address[](1);
        allowed[0] = seaDrop;

        uint256 g0 = gasleft();
        SequentialFatPunksForGas seq =
            new SequentialFatPunksForGas(allowed, IFatRenderer(address(stub)));
        seq.setMaxSupply(10_000);
        uint256 sequentialGas = g0 - gasleft();

        g0 = gasleft();
        FatPunks permuted =
            new FatPunks(allowed, IFatRenderer(address(stub)));
        permuted.setMaxSupply(10_000);
        uint256 permutedGas = g0 - gasleft();

        emit log_named_uint("sequential token deploy+max gas", sequentialGas);
        emit log_named_uint("permuted token deploy+max gas", permutedGas);
    }

    function _newSequentialForGas()
        internal returns (SequentialFatPunksForGas seq)
    {
        address[] memory allowed = new address[](1);
        allowed[0] = seaDrop;
        seq = new SequentialFatPunksForGas(
            allowed,
            IFatRenderer(address(stub))
        );
        seq.setMaxSupply(10_000);
    }

    function _gcd(uint256 a, uint256 b) internal pure returns (uint256) {
        while (b != 0) {
            uint256 t = b;
            b = a % b;
            a = t;
        }
        return a;
    }
}
