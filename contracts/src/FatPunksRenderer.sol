// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import { Base64 } from "solady/src/utils/Base64.sol";
import { LibString } from "solady/src/utils/LibString.sol";
import { DynamicBufferLib } from "solady/src/utils/DynamicBufferLib.sol";
import { FixedPointMath } from "./render/FixedPointMath.sol";
import { WeightData } from "./render/WeightData.sol";
import { ICryptopunksData } from "./interfaces/ICryptopunksData.sol";
import { IFatRenderer } from "./interfaces/IFatRenderer.sol";

/// @title  FatPunksRenderer
/// @notice Fully on-chain renderer: a tiny int8 conditional DCGAN embedded in
///         bytecode generates the fat body (grayscale, 32x32), which is
///         bucketed into 5 shades, recolored by the punk's own skin tone
///         (sampled from CryptopunksData), and composited under the real
///         24x24 punk head. Everything runs in a free `view` eth_call.
///         Neural engine adapted from "Artificial After All" by Han (MIT).
contract FatPunksRenderer is IFatRenderer {
    using DynamicBufferLib for DynamicBufferLib.DynamicBuffer;
    using LibString for uint256;

    ICryptopunksData public immutable PUNKS;

    // ───────────────────────── geometry / palette constants ──────────────
    uint256 private constant CANVAS_W = 32;
    uint256 private constant CANVAS_H = 40;
    uint256 private constant PUNK_X = 4;   // 24x24 punk pasted at (4, 0)
    uint256 private constant BODY_Y = 8;   // 32x32 body pasted at (0, 8)
    // Skin probe: punk pixel (x=9, y=23) is lit neck skin/fur on ALL 10,000
    // punks (verified exhaustively offline). RGBA offset = (23*24+9)*4.
    uint256 private constant SKIN_OFFSET = 2244;
    // Gray-bucket cuts live in WeightData (THRESH0..3): they are the fixed
    // float-space cuts y=(pixel-128)/128 for pixel in {30,95,160,220},
    // scaled by C = 128/prod(weight scales) into raw chain-logit units.

    string private constant BG = "#638596";

    constructor(ICryptopunksData punks_) {
        PUNKS = punks_;
    }

    // ───────────────────────────── neural forward ─────────────────────────

    /// @dev keccak latent (32 int8) + fatness conditioning (8 replicated int8)
    function _latentAndCond(uint256 seed, uint256 fatLevel, int256[] memory buf)
        internal pure
    {
        assembly {
            let ptr := add(buf, 32)
            mstore(0x00, seed)
            for { let batch := 0 } lt(batch, 4) { batch := add(batch, 1) } {
                mstore(0x20, batch)
                let hash := keccak256(0x00, 64)
                let base := mul(batch, 8)
                for { let j := 0 } lt(j, 8) { j := add(j, 1) } {
                    let val := signextend(0, and(shr(mul(j, 8), hash), 0xFF))
                    mstore(add(ptr, mul(add(base, j), 32)), val)
                }
            }
            // cond int8 = level*12 - 120, replicated in slots 32..39
            let c := sub(mul(fatLevel, 12), 120)
            for { let j := 32 } lt(j, 40) { j := add(j, 1) } {
                mstore(add(ptr, mul(j, 32)), c)
            }
        }
    }

    /// @dev Threshold raw chain logits into 5 shade levels (0..4).
    function _bucketize(int256[] memory x) internal pure returns (bytes memory lv) {
        lv = new bytes(1024);
        int256 t0 = WeightData.THRESH0;
        int256 t1 = WeightData.THRESH1;
        int256 t2 = WeightData.THRESH2;
        int256 t3 = WeightData.THRESH3;
        for (uint256 i = 0; i < 1024; ++i) {
            int256 v = x[i];
            uint8 l;
            if (v < t0) l = 0;
            else if (v < t1) l = 1;
            else if (v < t2) l = 2;
            else if (v < t3) l = 3;
            else l = 4;
            lv[i] = bytes1(l);
        }
    }

    /// @notice 1024 shade-level bytes (0=bg, 1..4 = outline..highlight).
    function bodyLevels(uint256 tokenId, uint8 fatLevel)
        public pure returns (bytes memory)
    {
        int256[] memory bufA = new int256[](8192);
        int256[] memory bufB = new int256[](2048);

        _latentAndCond(tokenId, fatLevel, bufA);

        FixedPointMath.denseLayerWithBias(
            bufA, bufB, WeightData.get_FC_WEIGHT(), WeightData.get_FC_BIAS(),
            WeightData.BIAS_SCALE_FC,
            40, 256);
        FixedPointMath.transposedConv2dFused(
            bufB, bufA, WeightData.get_CONVT1_WEIGHT(), WeightData.BIAS_CONVT1,
            WeightData.BIAS_SCALE_CONVT1,
            16, 16, 4, 4);
        FixedPointMath.transposedConv2dFused(
            bufA, bufB, WeightData.get_CONVT2_WEIGHT(), WeightData.BIAS_CONVT2,
            WeightData.BIAS_SCALE_CONVT2,
            16, 8, 8, 8);
        FixedPointMath.transposedConv2dFused(
            bufB, bufA, WeightData.get_CONVT3_WEIGHT(), WeightData.BIAS_CONVT3,
            WeightData.BIAS_SCALE_CONVT3,
            8, 8, 16, 16);
        FixedPointMath.conv1x1Fused(
            bufA, bufB, WeightData.get_CONV1X1_WEIGHT(), WeightData.BIAS_CONV1X1,
            WeightData.BIAS_SCALE_CONV1X1,
            8, 1, 32, 32);

        return _bucketize(bufB);
    }

    // ─────────────────────────── skin + palette ───────────────────────────

    /// @dev 4 body colors (outline, shadow, base, highlight) from the punk's
    ///      sampled skin: c*9/25, c*18/25, c, min(255, c*5/4) per channel.
    function _palette(bytes memory img) internal pure returns (uint24[4] memory p) {
        uint256 r = uint8(img[SKIN_OFFSET]);
        uint256 g = uint8(img[SKIN_OFFSET + 1]);
        uint256 b = uint8(img[SKIN_OFFSET + 2]);
        p[0] = _rgb(r * 9 / 25, g * 9 / 25, b * 9 / 25);
        p[1] = _rgb(r * 18 / 25, g * 18 / 25, b * 18 / 25);
        p[2] = _rgb(r, g, b);
        p[3] = _rgb(_min255(r * 5 / 4), _min255(g * 5 / 4), _min255(b * 5 / 4));
    }

    function _rgb(uint256 r, uint256 g, uint256 b) private pure returns (uint24) {
        return uint24((r << 16) | (g << 8) | b);
    }

    function _min255(uint256 v) private pure returns (uint256) {
        return v > 255 ? 255 : v;
    }

    // ─────────────────────────────── svg ──────────────────────────────────

    function svg(uint256 tokenId, uint8 fatLevel)
        public view override returns (string memory)
    {
        bytes memory img = PUNKS.punkImage(uint16(tokenId));
        bytes memory lv = bodyLevels(tokenId, fatLevel);
        uint24[4] memory pal = _palette(img);

        DynamicBufferLib.DynamicBuffer memory buf;
        buf.p('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 40" '
              'shape-rendering="crispEdges" width="640" height="800">'
              '<rect width="32" height="40" fill="', bytes(BG), '"/>');
        _bodyRects(buf, lv, pal);
        _punkRects(buf, img);
        buf.p("</svg>");
        return string(buf.data);
    }

    function _bodyRects(
        DynamicBufferLib.DynamicBuffer memory buf,
        bytes memory lv,
        uint24[4] memory pal
    ) internal pure {
        for (uint256 y = 0; y < 32; ++y) {
            uint256 x = 0;
            while (x < 32) {
                uint8 l = uint8(lv[y * 32 + x]);
                if (l == 0) { ++x; continue; }
                uint256 run = 1;
                while (x + run < 32 && uint8(lv[y * 32 + x + run]) == l) ++run;
                _rect(buf, x, y + BODY_Y, run, pal[l - 1], 255);
                x += run;
            }
        }
    }

    function _punkRects(
        DynamicBufferLib.DynamicBuffer memory buf,
        bytes memory img
    ) internal pure {
        for (uint256 y = 0; y < 24; ++y) {
            uint256 x = 0;
            while (x < 24) {
                uint256 o = (y * 24 + x) * 4;
                uint8 a = uint8(img[o + 3]);
                if (a == 0) { ++x; continue; }
                uint24 c = _rgb(uint8(img[o]), uint8(img[o + 1]), uint8(img[o + 2]));
                uint256 run = 1;
                while (x + run < 24) {
                    uint256 o2 = (y * 24 + x + run) * 4;
                    if (uint8(img[o2 + 3]) != a) break;
                    if (_rgb(uint8(img[o2]), uint8(img[o2 + 1]), uint8(img[o2 + 2])) != c) break;
                    ++run;
                }
                _rect(buf, x + PUNK_X, y, run, c, a);
                x += run;
            }
        }
    }

    function _rect(
        DynamicBufferLib.DynamicBuffer memory buf,
        uint256 x, uint256 y, uint256 w, uint24 color, uint8 alpha
    ) internal pure {
        buf.p('<rect x="', bytes(x.toString()),
              '" y="', bytes(y.toString()),
              '" width="', bytes(w.toString()));
        buf.p('" height="1" fill="#',
              bytes(LibString.toHexStringNoPrefix(uint256(color), 3)));
        if (alpha == 255) {
            buf.p('"/>');
        } else {
            buf.p('" fill-opacity="0.', bytes(_pct(alpha)), '"/>');
        }
    }

    function _pct(uint8 a) private pure returns (string memory) {
        // alpha/255 to two decimals, e.g. 0x80 -> "50"
        uint256 v = (uint256(a) * 100 + 127) / 255;
        return v < 10
            ? string(abi.encodePacked("0", v.toString()))
            : v.toString();
    }

    // ──────────────────────────── metadata ────────────────────────────────

    function _build(uint8 lvl) internal pure returns (string memory) {
        if (lvl == 0) return "OG Slim";
        if (lvl <= 5) return "Slim";
        if (lvl <= 10) return "Chubby";
        if (lvl <= 15) return "Fat";
        if (lvl <= 19) return "Huge";
        return "MEGAFAT ABSOLUTE UNIT";
    }

    function tokenURI(uint256 tokenId, uint8 fatLevel)
        external view override returns (string memory)
    {
        DynamicBufferLib.DynamicBuffer memory json;
        json.p(
            '{"name":"Fat Punk #', bytes(tokenId.toString()),
            '","description":"Fat Punk #', bytes(tokenId.toString())
        );
        json.p(
            '. Feed it burritos until it becomes an absolute unit.'
            ' Fully on-chain fatness, generated by a tiny neural network'
            ' living inside the contract.","attributes":['
        );
        _attributes(json, tokenId, fatLevel);
        json.p(
            '],"image":"data:image/svg+xml;base64,',
            bytes(Base64.encode(bytes(svg(tokenId, fatLevel)))),
            '"}'
        );
        return string(abi.encodePacked(
            "data:application/json;base64,", Base64.encode(json.data)));
    }

    function _attributes(
        DynamicBufferLib.DynamicBuffer memory buf,
        uint256 tokenId,
        uint8 fatLevel
    ) internal view {
        string memory raw = PUNKS.punkAttributes(uint16(tokenId));
        (
            string memory punkType,
            uint256 attributeCount,
            string[] memory accessories
        ) =
            _parsePunkAttributes(raw);

        _sortAccessoriesOpenSeaStyle(accessories);
        for (uint256 i = 0; i < accessories.length; ++i) {
            _stringTrait(buf, "Accessory", accessories[i], true);
        }
        _stringTrait(buf, "Accessory", string(abi.encodePacked(
            attributeCount.toString(), " attributes")), true);
        _stringTrait(buf, "Type", punkType, true);
        _uintTrait(buf, "Fatness Level", fatLevel, true);
        _stringTrait(buf, "Fatness Category", _build(fatLevel), false);
    }

    function _uintTrait(
        DynamicBufferLib.DynamicBuffer memory buf,
        string memory traitType,
        uint256 value,
        bool comma
    ) internal pure {
        buf.p(
            '{"trait_type":"', bytes(traitType),
            '","value":', bytes(value.toString()), "}"
        );
        if (comma) buf.p(",");
    }

    function _stringTrait(
        DynamicBufferLib.DynamicBuffer memory buf,
        string memory traitType,
        string memory value,
        bool comma
    ) internal pure {
        buf.p(
            '{"trait_type":"', bytes(traitType),
            '","value":"', bytes(value), '"}'
        );
        if (comma) buf.p(",");
    }

    function _parsePunkAttributes(string memory raw)
        internal pure returns (
            string memory punkType,
            uint256 attributeCount,
            string[] memory accessories
        )
    {
        bytes memory data = bytes(raw);
        uint256 accessoryCount;
        for (uint256 i = 0; i < data.length; ++i) {
            if (data[i] == ",") ++accessoryCount;
        }

        accessories = new string[](accessoryCount);
        uint256 start;
        uint256 part;
        for (uint256 i = 0; i <= data.length; ++i) {
            if (i != data.length && data[i] != ",") continue;
            if (part == 0) {
                punkType = _parseType(_slice(data, start, i));
            } else {
                uint256 accessoryStart = start;
                if (accessoryStart < i && data[accessoryStart] == " ") {
                    ++accessoryStart;
                }
                accessories[part - 1] = _slice(data, accessoryStart, i);
            }
            start = i + 1;
            ++part;
        }
        attributeCount = accessoryCount;
    }

    function _parseType(string memory rawType)
        private pure returns (string memory)
    {
        bytes memory data = bytes(rawType);
        uint256 lastSpace = type(uint256).max;
        for (uint256 i = 0; i < data.length; ++i) {
            if (data[i] == " ") lastSpace = i;
        }
        if (lastSpace == type(uint256).max || lastSpace + 1 >= data.length) {
            return rawType;
        }

        for (uint256 i = lastSpace + 1; i < data.length; ++i) {
            uint8 c = uint8(data[i]);
            if (c < 48 || c > 57) return rawType;
        }
        return _slice(data, 0, lastSpace);
    }

    function _sortAccessoriesOpenSeaStyle(string[] memory accessories)
        private pure
    {
        for (uint256 i = 1; i < accessories.length; ++i) {
            string memory value = accessories[i];
            uint256 priority = _accessoryPriority(value);
            uint256 j = i;
            while (
                j > 0 &&
                priority < _accessoryPriority(accessories[j - 1])
            ) {
                accessories[j] = accessories[j - 1];
                --j;
            }
            accessories[j] = value;
        }
    }

    function _accessoryPriority(string memory value)
        private pure returns (uint256)
    {
        bytes32 h = keccak256(bytes(value));

        // Small visual-category ordering to mirror CryptoPunks/OpenSea cards:
        // jewelry first, mouth/lips, eyes/eyewear, facial hair, then hair/hats.
        if (h == keccak256("Earring") || h == keccak256("Choker")) return 10;

        if (
            _contains(value, "Lipstick") ||
            h == keccak256("Smile") ||
            h == keccak256("Frown") ||
            h == keccak256("Cigarette") ||
            h == keccak256("Pipe") ||
            h == keccak256("Vape")
        ) return 20;

        if (
            _contains(value, "Eye") ||
            _contains(value, "Eyes") ||
            _contains(value, "Shades") ||
            _contains(value, "Glasses") ||
            h == keccak256("Eye Mask") ||
            h == keccak256("Eye Patch") ||
            h == keccak256("Welding Goggles") ||
            h == keccak256("VR")
        ) return 30;

        if (
            _contains(value, "Beard") ||
            _contains(value, "Mustache") ||
            h == keccak256("Goat") ||
            h == keccak256("Mole")
        ) return 40;

        return 50;
    }

    function _contains(string memory value, string memory needle)
        private pure returns (bool)
    {
        bytes memory haystack = bytes(value);
        bytes memory target = bytes(needle);
        if (target.length == 0 || target.length > haystack.length) return false;
        for (uint256 i = 0; i <= haystack.length - target.length; ++i) {
            bool found = true;
            for (uint256 j = 0; j < target.length; ++j) {
                if (haystack[i + j] != target[j]) {
                    found = false;
                    break;
                }
            }
            if (found) return true;
        }
        return false;
    }

    function _slice(bytes memory data, uint256 start, uint256 end)
        private pure returns (string memory)
    {
        bytes memory out = new bytes(end - start);
        for (uint256 i = start; i < end; ++i) {
            out[i - start] = data[i];
        }
        return string(out);
    }
}
