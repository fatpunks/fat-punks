// FatPunks token behavior, executed on a real EVM (@ethereumjs/vm) against
// the REAL renderer + WeightData library + mock punk store.
// Mirrors test/FatPunks.t.sol so "tests written" becomes "tests executed".
// Usage: node test_token.js
const fs = require("fs");
const path = require("path");
const { createVM } = require("@ethereumjs/vm");
const { createEVM } = require("@ethereumjs/evm");
const { Common, Hardfork, Mainnet } = require("@ethereumjs/common");
const { createBlock } = require("@ethereumjs/block");
const util = require("@ethereumjs/util");
const { keccak256: keccak } = require("ethereum-cryptography/keccak");

const ART = JSON.parse(fs.readFileSync(path.join(__dirname, "artifacts.json")));
const FIX = JSON.parse(
  fs.readFileSync(path.join(__dirname, "../../ml/out/punk_fixtures.json"))
);

const hexToBytes = (h) => util.hexToBytes(h.startsWith("0x") ? h : "0x" + h);
const sel = (sig) => keccak(new TextEncoder().encode(sig)).slice(0, 4);
const selHex = (sig) => util.bytesToHex(sel(sig));
const encUint = (n) => hexToBytes(BigInt(n).toString(16).padStart(64, "0"));
const encAddr = (a) => hexToBytes(a.toString().slice(2).padStart(64, "0"));
const AFFINE_A = 7321n;
const AFFINE_B = 7900n;
const PUNK_SUPPLY = 10000n;
const permutedTokenId = (ordinal) =>
  (BigInt(ordinal) * AFFINE_A + AFFINE_B) % PUNK_SUPPLY;

function callData(sig, words) {
  return util.concatBytes(sel(sig), ...words);
}
function decodeDynamic(ret) {
  const off = Number(util.bytesToBigInt(ret.slice(0, 32)));
  const len = Number(util.bytesToBigInt(ret.slice(off, off + 32)));
  return ret.slice(off + 32, off + 32 + len);
}

let passed = 0, failed = 0;
function ok(cond, name) {
  if (cond) { passed++; console.log("  ok", name); }
  else { failed++; console.log("  FAIL", name); }
}

async function main() {
  const common = new Common({ chain: Mainnet, hardfork: Hardfork.Shanghai });
  const evm = await createEVM({ common, allowUnlimitedContractSize: true });
  const vm = await createVM({ common, evm });

  const owner = util.createAddressFromString("0x1000000000000000000000000000000000000001");
  const seaDrop = util.createAddressFromString("0x00000000000000000000000000000000000005ea");
  const alice = util.createAddressFromString("0x00000000000000000000000000000000000a11ce");
  const bob   = util.createAddressFromString("0x0000000000000000000000000000000000000b0b");

  // fund the actors
  for (const a of [owner, seaDrop, alice, bob]) {
    await vm.evm.stateManager.putAccount(
      a, new util.Account(0n, 10n ** 21n /* 1000 ETH */)
    );
  }

  let now = 1_750_000_000n;
  const blockAt = (ts) =>
    createBlock({ header: { number: 1n, timestamp: ts, gasLimit: 1_000_000_000n } }, { common });

  async function raw(caller, to, data, value = 0n) {
    return vm.evm.runCall({
      caller, to, data, value,
      gasLimit: 6_000_000_000n,
      block: blockAt(now),
    });
  }
  async function call(caller, to, data, value = 0n) {
    const r = await raw(caller, to, data, value);
    if (r.execResult.exceptionError)
      throw new Error(
        "revert [" + r.execResult.exceptionError.error + "] " + util.bytesToHex(r.execResult.returnValue || new Uint8Array())
      );
    return r;
  }
  async function expectRevert(p, errSig, name) {
    const r = await p;
    const err = r.execResult.exceptionError;
    if (!err) return ok(false, name + " (did not revert)");
    if (!errSig) return ok(true, name);
    const got = util.bytesToHex(r.execResult.returnValue.slice(0, 4));
    ok(got === selHex(errSig), name + ` (${got})`);
  }
  async function deploy(caller, name, ctorTail = new Uint8Array(0)) {
    const data = util.concatBytes(hexToBytes(ART[name].bytecode), ctorTail);
    const r = await vm.evm.runCall({ caller, data, gasLimit: 6_000_000_000n, block: blockAt(now) });
    if (r.execResult.exceptionError)
      throw new Error(name + " deploy failed: " + r.execResult.exceptionError.error);
    return r.createdAddress;
  }

  // ── stack: WeightData lib → link renderer → mock+fixtures → renderer → token
  const lib = await deploy(owner, "WeightData");
  ART.FatPunksRenderer.bytecode = ART.FatPunksRenderer.bytecode.replace(
    /__\$[0-9a-f]{34}\$__/g, lib.toString().slice(2).toLowerCase());
  const mock = await deploy(owner, "MockCryptopunksData");
  for (const [idx, hx] of Object.entries(FIX)) {
    const b = hexToBytes(hx);
    const tail = util.concatBytes(
      encUint(idx), encUint(64),
      encUint(b.length), b, new Uint8Array((32 - (b.length % 32)) % 32));
    await call(owner, mock, util.concatBytes(sel("setPunk(uint16,bytes)"), tail));
  }
  const renderer = await deploy(owner, "FatPunksRenderer", encAddr(mock));
  // constructor(address[] allowedSeaDrop, IFatRenderer): [off 0x40][renderer][len 1][seaDrop]
  const nft = await deploy(owner, "FatPunks",
    util.concatBytes(encUint(0x40), encAddr(renderer), encUint(1), encAddr(seaDrop)));
  await call(owner, nft, callData("setMaxSupply(uint256)", [encUint(10_000)]));
  console.log("deployed FatPunks at", nft.toString());

  const mintSeaDrop = (minter, quantity) =>
    callData("mintSeaDrop(address,uint256)", [encAddr(minter), encUint(quantity)]);
  const totalSupply = async () => {
    const rr = await call(owner, nft, callData("totalSupply()", []));
    return util.bytesToBigInt(rr.execResult.returnValue);
  };
  const contractPermutedTokenId = async (ordinal) => {
    const rr = await call(owner, nft, callData("permutedTokenId(uint256)", [encUint(ordinal)]));
    return util.bytesToBigInt(rr.execResult.returnValue);
  };
  const mintOne = async (who) => {
    const id = permutedTokenId(await totalSupply());
    await call(seaDrop, nft, mintSeaDrop(who, 1));
    return id;
  };
  const mintMany = async (who, quantity) => {
    const first = await totalSupply();
    const ids = Array.from({ length: Number(quantity) }, (_, i) =>
      permutedTokenId(first + BigInt(i)));
    await call(seaDrop, nft, mintSeaDrop(who, quantity));
    return ids;
  };

  console.log("\nSeaDrop minting");
  await expectRevert(
    raw(alice, nft, mintSeaDrop(alice, 1)),
    "OnlyAllowedSeaDrop()", "non-SeaDrop mintSeaDrop reverts");
  ok((await contractPermutedTokenId(0)) === 7900n, "affine ordinal 0 -> token 7900");
  ok((await contractPermutedTokenId(1)) === 5221n, "affine ordinal 1 -> token 5221");
  ok((await contractPermutedTokenId(2)) === 2542n, "affine ordinal 2 -> token 2542");
  ok((await contractPermutedTokenId(100)) === 0n, "affine ordinal 100 -> token 0");
  await call(seaDrop, nft, mintSeaDrop(alice, 3));
  let r = await call(owner, nft, callData("ownerOf(uint256)", [encUint(permutedTokenId(0))]));
  ok(util.bytesToHex(r.execResult.returnValue).endsWith("a11ce"), "permuted token 7900 minted by SeaDrop");
  r = await call(owner, nft, callData("ownerOf(uint256)", [encUint(permutedTokenId(2))]));
  ok(util.bytesToHex(r.execResult.returnValue).endsWith("a11ce"), "permuted token 2542 minted by SeaDrop");
  await expectRevert(raw(owner, nft, callData("ownerOf(uint256)", [encUint(0)])), null, "token 0 not minted before ordinal 100");
  r = await call(owner, nft, callData("totalSupply()", []));
  ok(util.bytesToBigInt(r.execResult.returnValue) === 3n, "totalSupply 3");
  r = await call(owner, nft, callData("balanceOf(address)", [encAddr(alice)]));
  ok(util.bytesToBigInt(r.execResult.returnValue) === 3n, "balanceOf alice 3");
  r = await call(owner, nft, callData("getMintStats(address)", [encAddr(alice)]));
  ok(util.bytesToBigInt(r.execResult.returnValue.slice(0, 32)) === 3n, "getMintStats minterNumMinted 3");
  ok(util.bytesToBigInt(r.execResult.returnValue.slice(32, 64)) === 3n, "getMintStats totalSupply 3");
  ok(util.bytesToBigInt(r.execResult.returnValue.slice(64, 96)) === 10000n, "getMintStats maxSupply 10000");
  const exactClaim = util.concatBytes(sel("claim(uint256[])"), encUint(0x20), encUint(1), encUint(635));
  await expectRevert(
    raw(alice, nft, exactClaim),
    null, "public exact-index claim unavailable");

  console.log("\nfattening");
  const fatLevel = async (id) => {
    const rr = await call(owner, nft, callData("fatLevel(uint256)", [encUint(id)]));
    return util.bytesToBigInt(rr.execResult.returnValue);
  };
  const lastFattened = async (id) => {
    const rr = await call(owner, nft, callData("lastFattened(uint256)", [encUint(id)]));
    return util.bytesToBigInt(rr.execResult.returnValue);
  };
  const feed = (id) => callData("feed(uint256)", [encUint(id)]);
  const feedAll = (ids) => util.concatBytes(
    sel("feedAll(uint256[])"), encUint(0x20), encUint(ids.length),
    ...ids.map(encUint));
  const xfer = (from, to, id) => callData(
    "transferFrom(address,address,uint256)", [encAddr(from), encAddr(to), encUint(id)]);
  const fedTopic = util.bytesToHex(keccak(new TextEncoder().encode("Fed(uint256,uint8)")));
  const fedLogs = (r) => (r.execResult.logs || [])
    .filter((l) => util.bytesToHex(l[1][0]) === fedTopic);

  const feedId = permutedTokenId(0);
  ok((await fatLevel(feedId)) === 0n, "mint does not fatten");
  r = await call(owner, nft, callData("MAX_FAT_LEVEL()", []));
  ok(util.bytesToBigInt(r.execResult.returnValue) === 20n, "MAX_FAT_LEVEL is 20");
  r = await call(owner, nft, callData("fattenCooldown()", []));
  ok(util.bytesToBigInt(r.execResult.returnValue) === 86400n, "cooldown defaults to 24h");
  await expectRevert(
    raw(owner, nft, callData("setFattenCooldown(uint64)", [encUint(0)])),
    "InvalidFattenCooldown()", "zero cooldown reverts");
  ok((await lastFattened(feedId)) === 0n, "lastFattened starts at 0");

  const fed = await call(alice, nft, feed(feedId));
  ok((await fatLevel(feedId)) === 1n, "feed fattens to 1");
  ok(fedLogs(fed).length === 1, "Fed event emitted");
  ok((await lastFattened(feedId)) === now, "lastFattened records feed time");

  await call(alice, nft, xfer(alice, bob, feedId));
  ok((await fatLevel(feedId)) === 1n, "transfer does not fatten");
  await expectRevert(raw(alice, nft, feed(feedId)), "NotPunkOwner()", "previous owner cannot feed after transfer");
  await expectRevert(raw(bob, nft, feed(feedId)), "FeedOnCooldown()", "new owner still obeys cooldown");
  now += 86_401n;
  await call(bob, nft, feed(feedId));
  ok((await fatLevel(feedId)) === 2n, "new owner feeds after cooldown");

  const operatorId = await mintOne(alice);
  await call(alice, nft, callData("setApprovalForAll(address,bool)", [encAddr(bob), encUint(1)]));
  await expectRevert(raw(bob, nft, feed(operatorId)), "NotPunkOwner()", "approved operator cannot feed");

  console.log("\ncooldown and cap");
  const cooldownId = await mintOne(alice);
  const firstCooldown = await call(alice, nft, feed(cooldownId));
  ok((await fatLevel(cooldownId)) === 1n, "owner feed works");
  ok(fedLogs(firstCooldown).length === 1, "owner feed emits Fed");
  await expectRevert(raw(alice, nft, feed(cooldownId)), "FeedOnCooldown()", "early second feed reverts");
  now += 86_401n;
  await call(alice, nft, feed(cooldownId));
  ok((await fatLevel(cooldownId)) === 2n, "after cooldown: feed works");

  const maxId = await mintOne(bob);
  for (let i = 0; i < 20; i++) {
    await call(bob, nft, feed(maxId));
    now += 86_401n;
  }
  ok((await fatLevel(maxId)) === 20n, "cap at 20 after 20 feeds");
  await expectRevert(raw(bob, nft, feed(maxId)), "AlreadyMaxFat()", "feed at level 20 reverts");

  console.log("\nbatch feeding");
  const [batchFirst, batchSecond, batchThird] = await mintMany(alice, 3);
  const batch = await call(alice, nft, feedAll([batchFirst, batchSecond]));
  ok(fedLogs(batch).length === 2, "feedAll emits Fed per fed punk");
  ok((await fatLevel(batchFirst)) === 1n && (await fatLevel(batchSecond)) === 1n, "feedAll feeds multiple owned punks");
  const dup = await call(alice, nft, feedAll([batchThird, batchThird]));
  ok(fedLogs(dup).length === 1, "duplicate batch id emits once");
  ok((await fatLevel(batchThird)) === 1n, "duplicate batch id does not double-feed");
  await call(alice, nft, feedAll([batchFirst, maxId, 4040]));
  ok((await fatLevel(batchFirst)) === 1n, "feedAll skips cooldown id");
  ok((await fatLevel(maxId)) === 20n, "feedAll skips maxed or unowned id");

  for (let i = 0; i < 19; i++) {
    now += 86_401n;
    await call(alice, nft, feed(batchFirst));
  }
  ok((await fatLevel(batchFirst)) === 20n, "SeaDrop minted punk reaches level 20");

  console.log("\ntokenURI through the real renderer");
  let supply = await totalSupply();
  if (supply < 100n) {
    await call(seaDrop, nft, mintSeaDrop(bob, 100n - supply));
  }
  const fixtureId = await mintOne(bob);
  ok(fixtureId === 0n, "fixture token 0 minted at affine ordinal 100");
  for (let i = 0; i < 20; i++) {
    await call(bob, nft, feed(fixtureId));
    now += 86_401n;
  }

  r = await call(owner, nft, callData("tokenURI(uint256)", [encUint(fixtureId)]));
  const uri = Buffer.from(decodeDynamic(r.execResult.returnValue)).toString();
  ok(uri.startsWith("data:application/json;base64,"), "tokenURI is base64 json");
  const meta = JSON.parse(Buffer.from(uri.split("base64,")[1], "base64").toString());
  const lvl = await fatLevel(fixtureId);
  ok(BigInt(meta.attributes[0].value) === lvl, `Fatness attr matches state (${lvl})`);
  ok(meta.attributes[1].value === "MEGAFAT ABSOLUTE UNIT", "level 20 build is MEGAFAT ABSOLUTE UNIT");
  ok(meta.description.includes("burritos"), "metadata describes burrito feeding");
  ok(!meta.description.includes("transfer makes"), "metadata no longer describes transfer fattening");
  ok(meta.image.startsWith("data:image/svg+xml;base64,"), "image is inline svg");
  await expectRevert(raw(owner, nft, callData("tokenURI(uint256)", [encUint(404)])), null, "tokenURI nonexistent reverts");

  console.log("\nrenderer lock");
  await call(owner, nft, callData("setRenderer(address)", [encAddr(mock)]));
  await call(owner, nft, callData("setRenderer(address)", [encAddr(renderer)]));
  await call(owner, nft, callData("lockRenderer()", []));
  await expectRevert(
    raw(owner, nft, callData("setRenderer(address)", [encAddr(mock)])),
    "RendererIsLocked()", "setRenderer after lock reverts");

  console.log(`\n${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(1); });
