// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

/// @title  FixedPointMath — int8-weight neural primitives for EVM view-call inference
/// @notice Adapted (near-verbatim) from "Artificial After All" by Han
///         (github.com/hanrgba/artificialafterall, MIT license). Trained
///         weights are NOT reused; Fat Punks ships its own model.
///         Semantics: activations are wide int256 accumulators, never
///         rescaled between layers; each layer adds `int8(bias) * biasScale` and
///         (except conv1x1) applies ReLU. Weight layouts:
///           dense   : [out][in] bytes, row-major
///           convT   : per input channel, ceil(outCh/2) words of 32 bytes;
///                     word t = {16 kernel bytes for oc=2t}{16 for oc=2t+1}
///           conv1x1 : [oc] rows of inCh bytes (inCh <= 32)
///         convT is stride 2, kernel 4, padding 1: output[2*ih+kh-1][2*iw+kw-1]
///         accumulates input[ih][iw] * W[ic][oc][kh][kw] — identical to
///         PyTorch ConvTranspose2d(stride=2, padding=1, kernel_size=4).
library FixedPointMath {
    function denseLayerWithBias(
        int256[] memory input,
        int256[] memory output,
        bytes memory weight,
        bytes memory bias,
        int256 biasScale,
        uint256 inSize,
        uint256 outSize
    ) internal pure {
        assembly {
            let inputPtr := add(input, 32)
            let outputPtr := add(output, 32)
            let weightPtr := add(weight, 32)
            let biasPtr := add(bias, 32)
            let inSize4 := and(inSize, not(3))

            let oAddr := outputPtr
            for { let o := 0 } lt(o, outSize) { o := add(o, 1) } {
                let acc := 0
                let wAddr := add(weightPtr, mul(o, inSize))
                let wEnd := add(wAddr, inSize4)
                let iAddr := inputPtr

                for { } lt(wAddr, wEnd) { } {
                    let wWord := mload(wAddr)
                    acc := add(acc, mul(signextend(0, byte(0, wWord)), mload(iAddr)))
                    acc := add(acc, mul(signextend(0, byte(1, wWord)), mload(add(iAddr, 32))))
                    acc := add(acc, mul(signextend(0, byte(2, wWord)), mload(add(iAddr, 64))))
                    acc := add(acc, mul(signextend(0, byte(3, wWord)), mload(add(iAddr, 96))))
                    wAddr := add(wAddr, 4)
                    iAddr := add(iAddr, 128)
                }
                for { let i := inSize4 } lt(i, inSize) { i := add(i, 1) } {
                    acc := add(acc, mul(
                        signextend(0, byte(0, mload(add(weightPtr, add(mul(o, inSize), i))))),
                        mload(add(inputPtr, mul(i, 32)))
                    ))
                }

                acc := add(acc, mul(signextend(0, byte(0, mload(add(biasPtr, o)))), biasScale))
                if slt(acc, 0) { acc := 0 }
                mstore(oAddr, acc)
                oAddr := add(oAddr, 32)
            }
        }
    }

    function transposedConv2dFused(
        int256[] memory input,
        int256[] memory output,
        bytes memory weight,
        uint256 biasWord,
        int256 biasScale,
        uint256 inCh,
        uint256 outCh,
        uint256 inH,
        uint256 inW
    ) internal pure {
        uint256 outH = inH * 2;
        uint256 outW = inW * 2;

        assembly {
            let inputPtr := add(input, 32)
            let outputPtr := add(output, 32)
            let weightPtr := add(weight, 32)
            let inHW := mul(inH, inW)
            let outHW := mul(outH, outW)
            let outHW32 := mul(outHW, 32)

            let totalBytes := mul(mul(outCh, outHW), 32)
            calldatacopy(outputPtr, calldatasize(), totalBytes)

            let icWStride := mul(outCh, 16)

            for { let ic := 0 } lt(ic, inCh) { ic := add(ic, 1) } {
                let icInputBase := add(inputPtr, mul(mul(ic, inHW), 32))
                let icWBase := add(weightPtr, mul(ic, icWStride))

                for { let ih := 0 } lt(ih, inH) { ih := add(ih, 1) } {
                    let ih2 := mul(ih, 2)

                    for { let iw := 0 } lt(iw, inW) { iw := add(iw, 1) } {
                        let x := mload(add(icInputBase, mul(add(mul(ih, inW), iw), 32)))
                        if iszero(x) { continue }

                        let iw2 := mul(iw, 2)

                        for { let kh := 0 } lt(kh, 4) { kh := add(kh, 1) } {
                            let ohRaw := add(ih2, kh)
                            if or(iszero(ohRaw), gt(ohRaw, outH)) { continue }
                            let oh := sub(ohRaw, 1)
                            let ohOutW := mul(oh, outW)

                            for { let kw := 0 } lt(kw, 4) { kw := add(kw, 1) } {
                                let owRaw := add(iw2, kw)
                                if or(iszero(owRaw), gt(owRaw, outW)) { continue }
                                let ow := sub(owRaw, 1)
                                let kIdx := add(mul(kh, 4), kw)

                                let wPtr := add(icWBase, kIdx)
                                let wEnd := add(wPtr, mul(outCh, 16))
                                let oBase := add(outputPtr, mul(add(ohOutW, ow), 32))

                                for { } lt(wPtr, wEnd) { } {
                                    let wWord := mload(wPtr)
                                    let w0 := signextend(0, byte(0, wWord))
                                    let w1 := signextend(0, byte(16, wWord))

                                    mstore(oBase, add(mload(oBase), mul(w0, x)))
                                    let oBase1 := add(oBase, outHW32)
                                    mstore(oBase1, add(mload(oBase1), mul(w1, x)))

                                    wPtr := add(wPtr, 32)
                                    oBase := add(oBase1, outHW32)
                                }
                            }
                        }
                    }
                }
            }

            for { let oc := 0 } lt(oc, outCh) { oc := add(oc, 1) } {
                let biasVal := mul(signextend(0, byte(oc, biasWord)), biasScale)
                let p := add(outputPtr, mul(mul(oc, outHW), 32))
                let pEnd := add(p, mul(outHW, 32))
                for { } lt(p, pEnd) { p := add(p, 32) } {
                    let val := add(mload(p), biasVal)
                    if slt(val, 0) { val := 0 }
                    mstore(p, val)
                }
            }
        }
    }

    function conv1x1Fused(
        int256[] memory input,
        int256[] memory output,
        bytes memory weight,
        uint256 biasWord,
        int256 biasScale,
        uint256 inCh,
        uint256 outCh,
        uint256 H,
        uint256 W
    ) internal pure {
        uint256 hw = H * W;

        assembly {
            let inputPtr := add(input, 32)
            let outputPtr := add(output, 32)
            let weightPtr := add(weight, 32)
            let hw32 := mul(hw, 32)

            for { let oc := 0 } lt(oc, outCh) { oc := add(oc, 1) } {
                let bVal := mul(signextend(0, byte(oc, biasWord)), biasScale)
                let wWord := mload(add(weightPtr, mul(oc, inCh)))
                let oAddr := add(outputPtr, mul(mul(oc, hw), 32))

                for { let pos := 0 } lt(pos, hw) { pos := add(pos, 1) } {
                    let acc := 0
                    let iPtr := add(inputPtr, mul(pos, 32))
                    for { let ic := 0 } lt(ic, inCh) { ic := add(ic, 1) } {
                        acc := add(acc, mul(
                            signextend(0, byte(ic, wWord)),
                            mload(iPtr)
                        ))
                        iPtr := add(iPtr, hw32)
                    }
                    mstore(oAddr, add(acc, bVal))
                    oAddr := add(oAddr, 32)
                }
            }
        }
    }

    /// @dev Canonical Huffman decode. `table` layout: 61-byte header of
    ///      6-byte rows {codeLen, firstCode:2, count, symOff:2} zero-terminated,
    ///      then the symbol table. `compressed` must be zero-padded by >= 2
    ///      bytes past the last code so the 14-bit window read stays in bounds.
    function huffDecode(
        bytes memory compressed,
        bytes memory table,
        uint256 decompLen
    ) internal pure returns (bytes memory result) {
        result = new bytes(decompLen);

        assembly {
            let compPtr := add(compressed, 32)
            let tablePtr := add(table, 32)
            let resultPtr := add(result, 32)
            let symTablePtr := add(tablePtr, 61)

            let bitPos := 0
            let outPos := 0

            for { } lt(outPos, decompLen) { outPos := add(outPos, 1) } {
                let byteIdx := shr(3, bitPos)
                let bitOff := and(bitPos, 7)

                let raw := shr(232, mload(add(compPtr, byteIdx)))
                let window := and(shr(sub(24, add(bitOff, 14)), raw), 16383)

                let tPtr := tablePtr
                let matched := 0
                for { } 1 { } {
                    let codeLen := byte(0, mload(tPtr))
                    if iszero(codeLen) { break }

                    let firstCode := or(shl(8, byte(1, mload(tPtr))), byte(2, mload(tPtr)))
                    let count := byte(3, mload(tPtr))
                    let symOff := or(shl(8, byte(4, mload(tPtr))), byte(5, mload(tPtr)))

                    let code := shr(sub(14, codeLen), window)

                    if and(iszero(lt(code, firstCode)), lt(code, add(firstCode, count))) {
                        let sym := byte(0, mload(add(symTablePtr, add(symOff, sub(code, firstCode)))))
                        mstore8(add(resultPtr, outPos), sym)
                        bitPos := add(bitPos, codeLen)
                        matched := 1
                        break
                    }

                    tPtr := add(tPtr, 6)
                }
                if iszero(matched) { bitPos := add(bitPos, 1) }
            }
        }
    }
}
