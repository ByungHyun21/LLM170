#!/usr/bin/env bash
# rawcuda 리눅스 검증 하니스 — scripts/verify_cuda.bat 미러 (plans/124).
# 워크트리 루트에서 실행해야 한다(상대 자산 경로 계약 — 셔임이 보장).
# 네거티브 컨트롤은 반드시 비영 exit + NEG-DETECTED 마커여야 한다
# (검증기의 검증 — 원장 17). 셔임이 없으면 rustc 단독 빌드(계약 — std 외 크레이트 금지).
set -u
cd "$(dirname "$0")/.."
PROBE=target/cuda_probe
D27=${D27:-../models/Qwen3.8-27B-exl3-5.00bpw}
G4=${G4:-../models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_M.gguf}
export LLM170_CUDA_Q4_GGUF="$G4"
export LLM170_CUDA_EXL3_27_CONFIG=${LLM170_CUDA_EXL3_27_CONFIG:-/tmp/ulw-cuda/fixture/cfg27.json}
export LLM170_CUDA_EXL3_35_CONFIG=${LLM170_CUDA_EXL3_35_CONFIG:-/tmp/ulw-cuda/fixture/cfg35moe.json}
export LLM170_CUDA_EXL3_FN_CONFIG=${LLM170_CUDA_EXL3_FN_CONFIG:-/tmp/ulw-cuda/fixture/cfgfn.json}
fail=0

if [ ! -x "$PROBE" ]; then
  rustc --edition 2024 -O scripts/cuda_probe_shim.rs -o "$PROBE" || exit 1
fi

run() { # 긍정 프로브 — 영 exit가 PASS
  local name=$1; shift
  "$PROBE" "$@" > "target/cuda_$name.out" 2>&1
  local e=$?
  tail -1 "target/cuda_$name.out" | sed "s/^/[$name] /"
  if [ $e -ne 0 ]; then fail=1; echo "[$name] FAIL exit=$e"; fi
}
neg() { # 네거티브 컨트롤 — 비영 exit + NEG-DETECTED가 PASS
  local name=$1; shift
  "$PROBE" "$@" > "target/cuda_$name.out" 2>&1
  local e=$?
  if [ $e -eq 0 ]; then
    fail=1; echo "[$name] NEG FAILED TO FAIL (exit 0 — 검증기 결함)"
  elif ! grep -q NEG-DETECTED "target/cuda_$name.out"; then
    fail=1; echo "[$name] NEG 마커 없음 (exit=$e — 검증기 결함)"
  else
    echo "[$name] NEG-DETECTED (정상)"
  fi
}

run smoke smoke
run gemv gemv
neg gemv-neg gemv-neg
run norm norm
neg norm-neg norm-neg
run gemm2 gemm2
neg gemm2-neg gemm2-neg
run gdn gdn "$D27" "$D27"
neg gdn-neg gdn-neg "$D27"
run attn attn "$D27" "$D27"
neg attn-neg attn-neg "$D27"
run ew ew
run argmax argmax
neg argmax-neg argmax-neg
export LLM170_CUDA_Q4_GGUF="$G4"
run q4-dequant q4-dequant
run q4-gemv q4-gemv
run q4-gemm q4-gemm
neg q4-neg q4-neg
run moe moe
neg moe-neg moe-neg
run mtp mtp "$D27"
neg mtp-neg mtp-neg "$D27"
echo "[skip] hc|fn|qsa|ple|mtp-frame|fn-chain|ds4-* — Windows측 FN EXL3 픽스처 필요(보류, 2026-10-07)"

if [ $fail -eq 0 ]; then
  echo "verify_cuda: ALL PASS"
else
  echo "verify_cuda: FAILURES PRESENT"
fi
exit $fail
