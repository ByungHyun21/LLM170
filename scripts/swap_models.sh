#!/usr/bin/env bash
# 모델 스왑 스모크 — plans/cuda-models.md §10-7 (P0-4, 2026-10-08)
#
# 판정(각 모델 1회 기동 — 2026-10-08 사용자 논의로 A→B→A 왕복 제거:
# 프로세스 재기동 구조의 왕복은 결정론 재확인뿐이라 검출력이 없다.
# 무회귀 토큰 감시는 gate-exl3.sh가 담당):
#   1. 기동 배너 1줄(# boot: model=… format=… runtime=… offload=… ctx=… slots=… attach=…)
#      존재 + format 정합 — "무엇으로 도는지"를 로그만으로 판정.
#   2. A(GGUF qwen35)·B(EXL3) 기동 + 고정 프롬프트 24토큰 생성 — 같은 절차·같은 플래그.
#   3. C(W4A16) 명시 거부(정체불명 실패 금지 — B22) — W4A16 로더(P2 §3.5) 도입 전까지
#      거부가 정답. 도입 후 이 행을 "기동+토큰" 판정으로 전환한다.
# 새 지원 조합(모델×런타임)이 계획 표(§10)에 행으로 추가되면 여기에도 스모크를 추가한다.
# 사용법: scripts/swap_models.sh [백엔드=cpu|cuda]
set -u
cd "$(dirname "$0")/.."
BACKEND=${1:-cuda}
BIN=./target/release/llm170
PORT=18210
PROMPT='[148678, 65233, 202419]'   # 고정 토큰 프롬프트(엔진 무관)
NP=24

M_GGUF=../models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_M.gguf
M_EXL3=../models/Qwen3.8-27B-exl3-5.00bpw
M_W416=../models/Qwen3.8-27B-W4A16-AutoRound

fail=0
note() { echo "[swap] $*"; }

# serve를 포어그라운드로 띄우고(타임아웃 캡) 대기→요청→수집.
# 인자: 모델경로 추가플래그 아웃파일 로그파일
run_serve() {
  local model=$1 extra=$2 out=$3 log=$4
  : > "$log"
  ( for i in $(seq 1 60); do
      sleep 5
      grep -m1 "listening" "$log" >/dev/null 2>&1 && break
    done
    curl -s -m 600 "http://127.0.0.1:$PORT/v1/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"prompt\":$PROMPT,\"max_tokens\":$NP,\"temperature\":0}" > "$out"
    pkill -9 -x llm170 ) &
  timeout 420 "$BIN" serve --backend "$BACKEND" --model "$model" --ctx 1024 --slots 1 \
    --port $PORT $extra > "$log" 2>&1
  wait
}

check_banner() {
  local log=$1 model=$2 fmt=$3
  local b
  b=$(grep -m1 "^# boot: " "$log" || true)
  if [ -z "$b" ]; then
    echo "[swap] FAIL: 배너 없음($model)"; fail=1; return
  fi
  local want_fmt="format=$fmt "
  case "$b" in
    *"$want_fmt"*) note "배너 OK: $b" ;;
    *) echo "[swap] FAIL: 배너 format 불일치: $b (기대 $fmt)"; fail=1 ;;
  esac
}

tokens_of() { python3 -c "
import json
try:
  d=json.load(open('$1'))
  toks=d.get('tokens') or (d['choices'][0].get('tokens') if 'choices' in d else None)
  if not toks: raise KeyError('tokens')
  print(','.join(map(str,toks)))
except Exception as e:
  print('ERR',e)
"; }

# ── 1. A(GGUF qwen35) 기동 + 토큰 ──
note "[1/3] A(27B GGUF) 기동 — backend=$BACKEND"
run_serve "$M_GGUF" "" /tmp/swap_a.json /tmp/swap_a.log
check_banner /tmp/swap_a.log "$M_GGUF" gguf
A=$(tokens_of /tmp/swap_a.json)
case "$A" in ERR*) echo "[swap] FAIL: GGUF 요청 실패: $A"; fail=1;; *) note "GGUF 토큰 OK: $(echo "$A" | head -c 60)";; esac

# ── 2. B(EXL3) 기동 + 토큰 ──
note "[2/3] B(EXL3 27B) 기동"
run_serve "$M_EXL3" "" /tmp/swap_b.json /tmp/swap_b.log
check_banner /tmp/swap_b.log "$M_EXL3" exl3
B=$(tokens_of /tmp/swap_b.json)
case "$B" in ERR*) echo "[swap] FAIL: EXL3 요청 실패: $B"; fail=1;; *) note "EXL3 토큰 OK: $(echo "$B" | head -c 60)";; esac

# ── 3. C(W4A16) — 명시 에러 판정(기동 거부) ──
note "[3/3] C(W4A16) 명시 에러 판정"
timeout 60 "$BIN" serve --backend "$BACKEND" --model "$M_W416" --ctx 1024 --slots 1 \
  --port $PORT > /tmp/swap_c.log 2>&1
rc=$?
if grep -q "미지원 포맷(W4A16" /tmp/swap_c.log; then
  note "W4A16 명시 에러 OK (rc=$rc)"
else
  echo "[swap] FAIL: W4A16 명시 에러 없음(로그 확인) — /tmp/swap_c.log"; fail=1
  head -3 /tmp/swap_c.log
fi

if [ $fail -eq 0 ]; then
  echo "[swap] swap_models: ALL PASS (A→B→C 각 1회, backend=$BACKEND)"
else
  echo "[swap] swap_models: FAILURES PRESENT"
fi
exit $fail
