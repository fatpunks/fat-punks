// Compile Fat Punks contracts with solc 0.8.17 (soljson) — no Foundry needed.
// Artifacts land in evm-harness/artifacts.json.
const fs = require("fs");
const path = require("path");
const solc = require("solc");

const ROOT = path.join(__dirname, "..");

const ENTRY = [
  "src/FatPunksRenderer.sol",
  "src/MockCryptopunksData.sol",
  "src/FatPunks.sol",
];

const sources = {};
for (const f of ENTRY) {
  sources[f] = { content: fs.readFileSync(path.join(ROOT, f), "utf8") };
}

const input = {
  language: "Solidity",
  sources,
  settings: {
    remappings: [
      "solady/=lib/solady/",
      "forge-std/=lib/forge-std/src/",
      "ERC721A/=lib/ERC721A/contracts/",
      "openzeppelin-contracts/=lib/openzeppelin-contracts/contracts/",
      "utility-contracts/=lib/utility-contracts/src/",
      "solmate/=lib/solmate/src/",
      "seadrop/=lib/seadrop/src/",
    ],
    optimizer: { enabled: true, runs: 200 },
    evmVersion: "london",
    viaIR: true,
    outputSelection: {
      "*": { "*": ["abi", "evm.bytecode.object", "evm.deployedBytecode.object"] },
    },
  },
};

function findImports(p) {
  try {
    return { contents: fs.readFileSync(path.join(ROOT, p), "utf8") };
  } catch (e) {
    return { error: "not found: " + p };
  }
}

const out = JSON.parse(
  solc.compile(JSON.stringify(input), { import: findImports })
);

let fatal = false;
for (const e of out.errors || []) {
  if (e.severity === "error") fatal = true;
  console.log(`[${e.severity}] ${e.formattedMessage}`);
}
if (fatal) process.exit(1);

const artifacts = {};
for (const file of Object.keys(out.contracts || {})) {
  for (const name of Object.keys(out.contracts[file])) {
    const c = out.contracts[file][name];
    artifacts[name] = {
      abi: c.abi,
      bytecode: "0x" + c.evm.bytecode.object,
      deployedSize: c.evm.deployedBytecode.object.length / 2,
    };
  }
}
fs.writeFileSync(
  path.join(__dirname, "artifacts.json"),
  JSON.stringify(artifacts)
);
for (const [n, a] of Object.entries(artifacts)) {
  if (a.deployedSize > 0)
    console.log(
      `${n}: deployed ${a.deployedSize} bytes` +
        (a.deployedSize > 24576 ? "  ** EXCEEDS EIP-170 **" : "")
    );
}
