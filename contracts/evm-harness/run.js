// Deploy MockCryptopunksData + FatPunksRenderer on a real EVM
// (@ethereumjs/vm) and exercise them. Outputs:
//   ../../ml/out/evm_results.json  — bodyLevels hex per (tokenId, level)
//   ../../ml/out/sample_chain.svg  — one rendered SVG for eyeballing
// Usage: node run.js [--seeds N]   (extra random tokenIds for verification)
const fs = require("fs");
const path = require("path");
const { createVM } = require("@ethereumjs/vm");
const { createEVM } = require("@ethereumjs/evm");
const { Common, Hardfork, Mainnet } = require("@ethereumjs/common");
const util = require("@ethereumjs/util");

const ART = JSON.parse(fs.readFileSync(path.join(__dirname, "artifacts.json")));
const FIX = JSON.parse(
  fs.readFileSync(path.join(__dirname, "../../ml/out/punk_fixtures.json"))
);
const OUT = path.join(__dirname, "../../ml/out");

const hexToBytes = (h) => util.hexToBytes(h.startsWith("0x") ? h : "0x" + h);
const bytesToHex = (b) => util.bytesToHex(b);

// ---- minimal ABI helpers (all args we use are uint/bytes/address) ----------
const { keccak256: keccak } = require("ethereum-cryptography/keccak");
function selector(sig) {
  return keccak(new TextEncoder().encode(sig)).slice(0, 4);
}
function encUint(n) {
  return hexToBytes(BigInt(n).toString(16).padStart(64, "0"));
}
function encAddress(a) {
  return hexToBytes(a.toString().slice(2).padStart(64, "0"));
}
function encBytesTail(b) {
  const lenWord = encUint(b.length);
  const padded = new Uint8Array(Math.ceil(b.length / 32) * 32);
  padded.set(b);
  return util.concatBytes(lenWord, padded);
}
function callData(sig, headParts, tailBytes) {
  // headParts: array of 32-byte words; tailBytes: optional single dynamic bytes arg appended
  let head = util.concatBytes(selector(sig), ...headParts);
  if (tailBytes) {
    const offset = encUint(headParts.length * 32 + 32);
    head = util.concatBytes(
      selector(sig),
      ...headParts,
      offset,
      encBytesTail(tailBytes)
    );
  }
  return head;
}
function decodeDynamic(ret) {
  // single dynamic return (bytes or string)
  const off = Number(util.bytesToBigInt(ret.slice(0, 32)));
  const len = Number(util.bytesToBigInt(ret.slice(off, off + 32)));
  return ret.slice(off + 32, off + 32 + len);
}

async function main() {
  const nSeeds = (() => {
    const i = process.argv.indexOf("--seeds");
    return i > 0 ? parseInt(process.argv[i + 1]) : 0;
  })();

  const common = new Common({ chain: Mainnet, hardfork: Hardfork.Shanghai });
  // dev harness: allow oversize deploys; the REAL EIP-170 gate is asserted
  // separately on the final trained build (compile.js prints sizes).
  const evm = await createEVM({ common, allowUnlimitedContractSize: true });
  const vm = await createVM({ common, evm });
  const caller = util.createAddressFromString(
    "0x1000000000000000000000000000000000000001"
  );

  async function deploy(name, ctorWords = []) {
    const data = util.concatBytes(hexToBytes(ART[name].bytecode), ...ctorWords);
    const r = await vm.evm.runCall({
      caller,
      data,
      gasLimit: 1_000_000_000n,
    });
    if (r.execResult.exceptionError)
      throw new Error(name + " deploy: " + r.execResult.exceptionError.error);
    return r.createdAddress;
  }

  async function call(to, data, gasLimit = 5_000_000_000n) {
    const r = await vm.evm.runCall({ caller, to, data, gasLimit });
    if (r.execResult.exceptionError)
      throw new Error(
        "call failed: " +
          r.execResult.exceptionError.error +
          " " +
          bytesToHex(r.execResult.returnValue || new Uint8Array())
      );
    return r;
  }

  // link + deploy the external WeightData library first
  const lib = await deploy("WeightData");
  const libHex = lib.toString().slice(2).toLowerCase();
  ART.FatPunksRenderer.bytecode = ART.FatPunksRenderer.bytecode.replace(
    /__\$[0-9a-f]{34}\$__/g,
    libHex
  );

  const mock = await deploy("MockCryptopunksData");
  for (const [idx, hx] of Object.entries(FIX)) {
    await call(
      mock,
      callData("setPunk(uint16,bytes)", [encUint(idx)], hexToBytes(hx))
    );
  }
  const renderer = await deploy("FatPunksRenderer", [encAddress(mock)]);
  console.log(
    "deployed; renderer", ART.FatPunksRenderer.deployedSize,
    "B + WeightData lib", ART.WeightData.deployedSize, "B");

  // ---- bodyLevels over fixtures x levels, plus extra random seeds ----------
  const results = {};
  const fixIds = Object.keys(FIX).map(Number);
  const levels = [0, 20];
  const jobs = [];
  for (const id of fixIds) for (const l of levels) jobs.push([id, l]);
  let rng = 12345n;
  for (let i = 0; i < nSeeds; i++) {
    rng = (rng * 6364136223846793005n + 1442695040888963407n) & ((1n << 64n) - 1n);
    jobs.push([Number(rng % 10000n), Number(rng % 21n)]);
  }
  let gasBody = 0n;
  let done = 0;
  for (const [id, l] of jobs) {
    const r = await call(
      renderer,
      callData("bodyLevels(uint256,uint8)", [encUint(id), encUint(l)])
    );
    gasBody = r.execResult.executionGasUsed;
    results[`${id}_${l}`] = Buffer.from(
      decodeDynamic(r.execResult.returnValue)
    ).toString("hex");
    if (++done % 10 === 0) {
      fs.writeFileSync(path.join(OUT, "evm_results.json"), JSON.stringify(results));
      console.log(`progress ${done}/${jobs.length}`);
    }
  }
  fs.writeFileSync(path.join(OUT, "evm_results.json"), JSON.stringify(results));
  console.log(`bodyLevels: ${jobs.length} calls, last gas ${gasBody}`);

  // ---- one full svg + tokenURI for visual + gas ----------------------------
  const rs = await call(
    renderer,
    callData("svg(uint256,uint8)", [encUint(0), encUint(20)])
  );
  const svgStr = Buffer.from(decodeDynamic(rs.execResult.returnValue)).toString();
  fs.writeFileSync(path.join(OUT, "sample_chain.svg"), svgStr);
  console.log("svg(0,20): gas", rs.execResult.executionGasUsed, "len", svgStr.length);

  const rt = await call(
    renderer,
    callData("tokenURI(uint256,uint8)", [encUint(0), encUint(20)])
  );
  const uri = Buffer.from(decodeDynamic(rt.execResult.returnValue)).toString();
  const json = JSON.parse(
    Buffer.from(uri.split("base64,")[1], "base64").toString()
  );
  console.log("tokenURI(0,20): gas", rt.execResult.executionGasUsed,
    "| name:", json.name, "| attrs:", JSON.stringify(json.attributes));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
