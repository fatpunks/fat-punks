// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import { ERC721SeaDrop } from "seadrop/ERC721SeaDrop.sol";
import { ERC721A__IERC721Receiver } from "ERC721A/ERC721A.sol";
import { Lifebuoy } from "solady/src/utils/Lifebuoy.sol";
import { IFatRenderer } from "./interfaces/IFatRenderer.sol";

/// @title  Fat Punks
/// @notice 10,000 fully on-chain derivatives of CryptoPunks. tokenId == punk
///         index. Owners make a punk FATTER by FEEDING it (`feed`), at most
///         once per cooldown, up to level 20. Fatness only ever goes up and
///         PERSISTS THROUGH TRANSFERS — a fat punk stays fat when sold or
///         sent. Art is generated at read time by a tiny neural network
///         embedded in the renderer's bytecode and composited with the real
///         punk head from on-chain CryptopunksData.
///
///         Public minting goes through OpenSea Drops / SeaDrop's standard
///         quantity mint hook. Token ids are assigned in a deterministic
///         affine-scrambled order over 0..9999, so tokenId always remains the
///         matching CryptoPunk index while public minters cannot choose exact
///         ids or snipe obvious early sequential ids.
contract FatPunks is ERC721SeaDrop, Lifebuoy {
    // ─────────────────────────────── errors ───────────────────────────────
    error RendererIsLocked();
    error WithdrawFailed();
    error NotPunkOwner();
    error AlreadyMaxFat();
    error FeedOnCooldown();
    error InvalidFattenCooldown();
    error InvalidAffineOrdinal(uint256 ordinal);

    // ─────────────────────────────── events ───────────────────────────────
    event Fed(uint256 indexed tokenId, uint8 newLevel);
    event RendererSet(address renderer);
    event RendererLockedForever();
    event FattenCooldownSet(uint64 cooldown);

    // ─────────────────────────────── state ────────────────────────────────
    struct FatState {
        uint8 level;          // 0..20
        uint64 lastFattened;  // unix ts of last fattening (0 = never)
    }

    mapping(uint256 => FatState) private _fat;

    IFatRenderer public renderer;
    bool public rendererLocked;
    uint64 public fattenCooldown;   // seconds between feeds; never zero

    uint256 public constant MAX_FAT_LEVEL = 20;
    uint256 public constant PUNK_SUPPLY = 10_000;
    uint256 public constant AFFINE_A = 7_321;
    uint256 public constant AFFINE_B = 7_900;

    constructor(address[] memory allowedSeaDrop, IFatRenderer renderer_)
        ERC721SeaDrop("Fat Punks", "FATPUNK", allowedSeaDrop)
    {
        renderer = renderer_;
        emit RendererSet(address(renderer_));
        fattenCooldown = 1 days; // minimum time between feeds (24h)
        emit FattenCooldownSet(1 days);
    }

    // ───────────────────── ERC721A id-space configuration ─────────────────
    /// @dev Punk indices start at 0 (SeaDrop default is 1).
    function _startTokenId() internal view virtual override returns (uint256) {
        return 0;
    }

    /// @dev Reserve only token 0 for sequential minting. Tokens 1..9999 are
    ///      minted as ERC721A spot mints in affine-permuted order.
    function _sequentialUpTo() internal view virtual override returns (uint256) {
        return 0;
    }

    // ─────────────────────────────── minting ──────────────────────────────

    function permutedTokenId(uint256 mintOrdinal)
        public pure returns (uint256)
    {
        if (mintOrdinal >= PUNK_SUPPLY) revert InvalidAffineOrdinal(mintOrdinal);
        return _permutedTokenIdUnchecked(mintOrdinal);
    }

    /// @notice SeaDrop mint hook. SeaDrop enforces drop config, then this
    ///         assigns the next `quantity` mint ordinals through the public,
    ///         deterministic affine permutation.
    function mintSeaDrop(address minter, uint256 quantity)
        external
        virtual
        override
        nonReentrant
    {
        _onlyAllowedSeaDrop(msg.sender);
        if (quantity == 0) revert MintZeroQuantity();

        uint256 firstOrdinal = _totalMinted();
        uint256 endOrdinal = firstOrdinal + quantity;
        if (endOrdinal > maxSupply()) {
            revert MintQuantityExceedsMaxSupply(endOrdinal, maxSupply());
        }

        for (uint256 ordinal = firstOrdinal; ordinal < endOrdinal; ) {
            uint256 tokenId = _permutedTokenIdUnchecked(ordinal);
            if (tokenId == 0) {
                _mint(minter, 1);
            } else {
                _mintSpot(minter, tokenId);
            }
            unchecked { ++ordinal; }
        }

        if (minter.code.length != 0) {
            for (uint256 ordinal = firstOrdinal; ordinal < endOrdinal; ) {
                _checkOnERC721ReceivedAfterMint(
                    minter,
                    _permutedTokenIdUnchecked(ordinal)
                );
                unchecked { ++ordinal; }
            }
        }
    }

    function _permutedTokenIdUnchecked(uint256 mintOrdinal)
        internal pure returns (uint256)
    {
        return (mintOrdinal * AFFINE_A + AFFINE_B) % PUNK_SUPPLY;
    }

    function _checkOnERC721ReceivedAfterMint(address to, uint256 tokenId)
        private
    {
        try ERC721A__IERC721Receiver(to).onERC721Received(
            _msgSenderERC721A(),
            address(0),
            tokenId,
            ""
        ) returns (bytes4 retval) {
            if (retval != ERC721A__IERC721Receiver.onERC721Received.selector) {
                revert TransferToNonERC721ReceiverImplementer();
            }
        } catch (bytes memory reason) {
            if (reason.length == 0) {
                revert TransferToNonERC721ReceiverImplementer();
            }
            assembly {
                revert(add(32, reason), mload(reason))
            }
        }
    }

    // ─────────────────────────────── feeding ──────────────────────────────
    //
    // Fatness changes ONLY through feeding. Transfers no longer affect it
    // (there is deliberately no _afterTokenTransfers override), so fatness
    // persists unchanged when a punk is sold or sent. Fatness only goes up.

    /// @notice Feed one of your punks: +1 fatness (cap 20), at most once per
    ///         cooldown (24h by default).
    function feed(uint256 tokenId) external {
        if (ownerOf(tokenId) != msg.sender) revert NotPunkOwner();
        FatState memory s = _fat[tokenId];
        if (s.level >= MAX_FAT_LEVEL) revert AlreadyMaxFat();
        if (
            s.lastFattened != 0 &&
            block.timestamp < uint256(s.lastFattened) + fattenCooldown
        ) revert FeedOnCooldown();
        uint8 newLevel = s.level + 1;
        _fat[tokenId] = FatState(newLevel, uint64(block.timestamp));
        emit Fed(tokenId, newLevel);
    }

    /// @notice Feed many punks in one transaction. The caller supplies the ids
    ///         (the dApp passes the wallet's punks — ids are NOT auto-discovered
    ///         on-chain). Duplicate, unowned, maxed, nonexistent, or
    ///         cooling-down ids are SKIPPED rather than reverting, so a partial
    ///         batch still succeeds.
    function feedAll(uint256[] calldata tokenIds) external {
        uint256 ts = block.timestamp;
        for (uint256 i = 0; i < tokenIds.length; ) {
            uint256 id = tokenIds[i];
            if (!_seenEarlier(tokenIds, i) && _isFeedable(id, msg.sender, ts)) {
                uint8 newLevel = _fat[id].level + 1;
                _fat[id] = FatState(newLevel, uint64(ts));
                emit Fed(id, newLevel);
            }
            unchecked { ++i; }
        }
    }

    function _seenEarlier(uint256[] calldata tokenIds, uint256 currentIndex)
        internal pure returns (bool)
    {
        uint256 id = tokenIds[currentIndex];
        for (uint256 i = 0; i < currentIndex; ) {
            if (tokenIds[i] == id) return true;
            unchecked { ++i; }
        }
        return false;
    }

    /// @dev True only if `who` currently owns `id` and it can be fed at time
    ///      `ts`. Lets feedAll skip ineligible ids without reverting. Short-
    ///      circuits so ownerOf() is never called on a nonexistent token.
    function _isFeedable(uint256 id, address who, uint256 ts)
        internal view returns (bool)
    {
        if (!_exists(id) || ownerOf(id) != who) return false;
        FatState memory s = _fat[id];
        if (s.level >= MAX_FAT_LEVEL) return false;
        if (s.lastFattened != 0 && ts < uint256(s.lastFattened) + fattenCooldown)
            return false;
        return true;
    }

    // ─────────────────────────────── views ────────────────────────────────

    function fatLevel(uint256 tokenId) public view returns (uint8) {
        return _fat[tokenId].level;
    }

    function lastFattened(uint256 tokenId) external view returns (uint64) {
        return _fat[tokenId].lastFattened;
    }

    function tokenURI(uint256 tokenId)
        public view virtual override returns (string memory)
    {
        if (!_exists(tokenId)) revert URIQueryForNonexistentToken();
        return renderer.tokenURI(tokenId, _fat[tokenId].level);
    }

    /// @notice Preview any punk at any fatness — powers the dApp slider.
    function previewSvg(uint256 tokenId, uint8 level)
        external view returns (string memory)
    {
        return renderer.svg(tokenId, level);
    }

    // ─────────────────────────────── admin ────────────────────────────────

    function setRenderer(IFatRenderer newRenderer) external onlyOwner {
        if (rendererLocked) revert RendererIsLocked();
        renderer = newRenderer;
        emit RendererSet(address(newRenderer));
    }

    /// @notice One-way switch: freezes the art forever.
    function lockRenderer() external onlyOwner {
        rendererLocked = true;
        emit RendererLockedForever();
    }

    function setFattenCooldown(uint64 cooldown) external onlyOwner {
        if (cooldown == 0) revert InvalidFattenCooldown();
        fattenCooldown = cooldown;
        emit FattenCooldownSet(cooldown);
    }

    function withdraw(address payable to) external onlyOwner {
        (bool ok, ) = to.call{ value: address(this).balance }("");
        if (!ok) revert WithdrawFailed();
    }
}
