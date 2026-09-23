#!/usr/bin/env bash
# 매칭 스코어카드 — 양 모델 × (pp512/4k/16k, tg@4k/16k) 를 우리 엔진과 llama.cpp 로
# 같은 조건에서 연속 측정한다. llama 는 llama-bench 로 동일 파라미터.
#
# 사용: scripts/scorecard.sh [llama|ours|all]
set -uo pipefail
cd "$(dirname "$0")/.."
# llama 빌드는 rocm-7.2.2 로 컴파일 — 런타임도 7.2.2 로 매칭(2026-09-23).
export LD_LIBRARY_PATH=/opt/rocm-7.2.2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm-7.2.2/lib/rocblas/library

M27=/home/yoon/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL.gguf
MFN=/home/yoon/models/qwen3.8-Flash-Next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
# 2026-09-23: llama.cpp b1ff4ca23 빌드(source/llama.cpp) — qwen4exp 가 마스터에
# 있어 단일 빌드로 두 모델 커버. 구 -fit/-lm mmap 는 신 CLI에서 제거/행업.
SRC=source/llama.cpp
LBHIP=$SRC/build-hip/bin/llama-bench
LBVK=$SRC/build-vk/bin/llama-bench
LB=${LLM170_LB:-$LBHIP}
LBFN=${LLM170_LBFN:-$LBHIP}
WHICH=${1:-all}

run_ours() {
  local name=$1 model=$2; shift 2
  for pt in 512 4096 16384; do
    timeout 1800 ./target/release/llm170 bench --model "$model" --pp $pt --tg 0 --reps 1 \
      --ctx 20480 --backend gpu 2>/dev/null | grep -E "\| pp" | sed "s/^/[ours $name] /"
  done
  for ctx in 4096 16384; do
    timeout 1800 ./target/release/llm170 bench --model "$model" --pp 512 --tg 128 --reps 1 \
      --ctx $((ctx+512)) --backend gpu 2>/dev/null | grep -E "\| tg" | sed "s/^/[ours $name tg@$ctx] /"
  done
}

run_llama() {
  local name=$1 model=$2 bench=$3; shift 3
  local extra=""
  # FN: 임베딩만 CPU — 신 CLI: -lm mmap 는 4파트 모델에서 로드 행업(b1ff4ca23),
  # -fit off 는 제거됨(구 fa-off). fa=auto 기본이 최선.
  [[ "$name" == "FN" ]] && extra="-ot per_layer_token_embd=CPU"
  timeout 3600 "$bench" -m "$model" -ngl 99 $extra -p 512,4096,16384 -n 0 -r 1 2>/dev/null \
    | sed "s/^/[llama $name] /"
  for ctx in 4096 16384; do
    timeout 3600 "$bench" -m "$model" -ngl 99 $extra -p 512 -n 128 -d $ctx -r 1 2>/dev/null \
      | sed "s/^/[llama $name tg@$ctx] /"
  done
}

case $WHICH in
  ours) run_ours 27B "$M27"; run_ours FN "$MFN" ;;
  llama) run_llama 27B "$M27" "$LB"; run_llama FN "$MFN" "$LBFN" ;;
  all) run_ours 27B "$M27"; run_ours FN "$MFN"; run_llama 27B "$M27" "$LB"; run_llama FN "$MFN" "$LBFN" ;;
esac
