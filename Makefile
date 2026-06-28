# Fat Punks verification targets.
.PHONY: train-float train-qat export compile verify strips parity app

train-float:
	cd ml && python3 train.py --phase float --epochs 500 --resume

train-qat:
	cd ml && python3 train.py --phase qat --epochs 200

export:
	cd ml && python3 export.py --ckpt out/ckpt_qat.pt

compile:
	cd contracts/evm-harness && node compile.js

verify:
	cd contracts/evm-harness && node run.js --seeds 300
	cd ml && python3 verify/pure_forward.py --check
	cd contracts/evm-harness && node test_token.js

strips:
	cd ml && python3 verify/render_strip.py

parity:        # same check through Foundry's EVM (needs forge)
	cd contracts && forge script script/ParityDump.s.sol
	cd ml && python3 verify/pure_forward.py --check

app:
	cd app && npm install && npm run build
