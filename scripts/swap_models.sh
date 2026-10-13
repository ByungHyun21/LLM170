#!/usr/bin/env bash
# W4A16 스모크 — 참조(CPU)·GPU 체인·serve 종단 + 명시 거부 판정.
# 2026-10-08 단일 트랙(W3-3: GPU 서빙 착륙).
#
# 판정:
#   1. A(27B): 참조(CPU) greedy 토큰열 골든 접두 일치 — 오라클.
#   2. B(27B): GPU 체인(w4a16-gpu) 토큰열 골든 접두 일치.
#   3. C(27B): serve HTTP 종단 — 배너 runtime=cuda + 골든 접두 + 배치 불변성
#      (동시 2요청 = 슬롯 0·1 배치 체인 토큰 = 단일 경로 골든).
#   4. D(35B-A3B INT4 g32): 참조(CPU) MoE 오라클 — W4-1(2026-10-08) g32/bf16
#      전문가 + 라우터 top-8·shared 게이트. 골든 접두 일치 + 로드 완전성.
#   5. D2(35B) GPU 체인(MoE) — w4a16-gpu 골든 접두(플레인 bf16 + 전문가 상주).
#   6. D3(35B) 다중 청크(35토큰 = 32+3) — 배치 프리필 t≤32의 CPU/GPU 골든.
#      (2026-10-09 교훈: 3토큰 케이스는 단일 청크만 탄다 — 다중 청크를 상시
#       검증한다. 이 케이스가 gdn_exp 도메인 결함을 잡는 그물이다.)
#   7. E(FN FP8PLE): 자원 가드 명시 거부(120GiB > 호스트).
#   8. G(27B): 장문 600·4000토큰 — lim>256(attn 분할)·다중 청크 경로 골든.
#   9. G2(35B): 장문 600·4000토큰 — 동일(35B MoE).
#  10. H(스펙 옵트인): --spec 4 골든 접두(27B/35B) — 검증=t=1 등가·보정 경로 회귀.
#  6b. D4(35B): 스트리밍 serve(--moe streaming) 골든 — 전문가 LRU 캐시 경로 판정.
#      [버그수정 2026-10-13] t≤8 스테이징 삽입이 가중치 업로드를 빠뜨려(유령 슬롯)
#      스트리밍 출력이 오염됐다(워밍업/프리필 → t=1 디코드까지 전파) — 상시 판정.
# 장문 골든(600/4000)은 GPU 동결(2회 재현) — CPU 대조는 35토큰 한계(D3).
# 경로 회귀 검출용(값 변경 시 원인 판정 후 재동결).
# [KVQ 채택 2026-10-10] int8 KV가 단일 경로 — 구 F(KVQ) 섹션은 기본 경로와
# 동일해져 제거. 35B 600 골든은 int8 채택값으로 재동결(2회 재현, eval.md).
# 사용법: scripts/swap_models.sh
set -u
cd "$(dirname "$0")/.."
BIN=./target/release/llm170
GOLDEN="156037,16072,154029,209495,30"
# W4-1 MoE(35B-A3B g32) — 참조 골든 접두(2026-10-09 확정).
GOLDEN35="156037,16072,154029,156504,30"
# 장문 골든(2026-10-10 GPU 동결): lim>256·다중 청크.
P600=$(python3 -c "print(','.join(str(1000+i) for i in range(600)))")
P4000=$(python3 -c "print(','.join(str(1000+i) for i in range(4000)))")
GOLDEN27_600="220,197,320,198,197,320,198,197,320"
GOLDEN27_4K="291,806,1184,2354,597,749,477,13,198"
GOLDEN35_600="198,248046,198,248045,248068,271,248069,271,2064" # 2026-10-10 FLA-8(__expf) 재동결(2회 재현) — 플립: "\nHere's a"→"\n\n?\n\nIt"
GOLDEN35_4K="291,806,81,2177,248046,198,248045,248068,271" # 2026-10-13 GDN 상태 f16+행-스케일 채택 재동결(2회 재현 결정적) — pos9 271↔198 근접동률(10-12 플립의 역복귀). 게이트: mix 8K/16K/32K −0.20/−0.39/−0.51%(CI 0 포함)·골든 나머지 전부 불변.
# 다중 청크(35토큰)골든 — CPU·GPU 동일 실측(2026-10-09).
GOLDEN35L="271,248068,271,248069,271,168951,227596,149285,65233"
P35L="148678,65233,202419,156037,16072,154029,156504,30,149285,65233,195939,149820,152091,150868,54581,155180,177992,12434,170819,149820,152091,16869,201523,229231,204817,13,153343,51643,155497,171699,25845,159667,32762,189291,13"
# A2: 동시 실행 충돌 방지 — 실행별 런 디렉터리·포트(고정 18210/고정 파일명 해소).
mkdir -p /tmp/opencode
RUN=$(mktemp -d /tmp/opencode/w4a16-smoke.XXXXXX)
PORT=$((18210 + ($$ % 400)))

M_27B=../models/Qwen3.8-27B-W4A16-AutoRound
M_35B=../models/Qwen3.6-35B-A3B-INT4-W4A16
M_FN=../models/Qwen3.8-Flash-Next-W4A16-FP8PLE

fail=0
note() { echo "[w4a16] $*"; }

# ── 1. A(27B) 참조 토큰 ──
note "[1/10] A(27B) 참조 실행 — w4a16-ref"
timeout 900 "$BIN" w4a16-ref "$M_27B" --prompt-tokens 148678,65233,202419 --n-predict 8 --ctx 1024 \
  > "$RUN/a.out" 2> "$RUN/a.log"
A=$(grep -m1 '^tokens:' "$RUN/a.out" | sed 's/^tokens: //')
case "$A" in
  "$GOLDEN"*) note "참조 토큰 OK: $(echo "$A" | head -c 60)";;
  "") echo "[w4a16] FAIL: 참조 실행 실패 — ${RUN}/a.log"; fail=1;;
  *) echo "[w4a16] FAIL: 골든 접두 불일치: $(echo "$A" | head -c 60)"; fail=1;;
esac

# ── 2. B(27B) GPU 체인 ──
note "[2/10] B(27B) GPU 체인 — w4a16-gpu"
timeout 900 "$BIN" w4a16-gpu "$M_27B" --prompt-tokens 148678,65233,202419 --n-predict 8 --ctx 1024 \
  > "$RUN/b.out" 2> "$RUN/b.log"
B=$(grep -m1 '^ tokens:' "$RUN/b.out" | sed 's/^ tokens: //')
case "$B" in
  "$GOLDEN"*) note "GPU 토큰 OK: $(echo "$B" | head -c 60)";;
  "") echo "[w4a16] FAIL: GPU 체인 실패 — ${RUN}/b.log"; fail=1;;
  *) echo "[w4a16] FAIL: GPU 골든 불일치: $(echo "$B" | head -c 60)"; fail=1;;
esac

# ── 3. C(27B) serve HTTP 종단 ──
note "[3/10] C(27B) serve HTTP — runtime=cuda + 골든 접두"
: > "$RUN/c.log"
"$BIN" serve --model "$M_27B" --ctx 1024 --port "$PORT" > "$RUN/c.log" 2>&1 &
SERVE_PID=$!
# A2: 기기 전체 pkill 금지 — 이 실행의 서버 PID만 종료(개발 서버 보호).
trap 'kill "$SERVE_PID" 2>/dev/null' EXIT
for i in $(seq 1 100); do
  sleep 3
  grep -m1 "listening" "$RUN/c.log" >/dev/null 2>&1 && break
done
curl -s -m 600 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt":[148678,65233,202419],"max_tokens":8,"temperature":0}' > "$RUN/c.json"
# [배치 불변성 2026-10-13] 동시 2요청 — 슬롯 0·1이 배치 체인으로 묶여도 토큰은
# 단일 경로와 동일해야 한다. 가드 배경: 상태 f16+스케일 채택이 단일 경로에만
# 배선돼 np·다중 슬롯이 illegal access로 죽던 회귀(스케일 슬롯 기저 128× OOB +
# 배치 체인 sc 누락) — 슬롯 0만 검사하던 종전 스모크가 통과시켰다.
curl -s -m 600 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt":[148678,65233,202419],"max_tokens":8,"temperature":0}' > "$RUN/c2a.json" &
C2A=$!
curl -s -m 600 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt":[148678,65233,202419],"max_tokens":8,"temperature":0}' > "$RUN/c2b.json" &
C2B=$!
wait "$C2A" "$C2B"
kill "$SERVE_PID" 2>/dev/null
wait "$SERVE_PID" 2>/dev/null
trap - EXIT
if grep -q "runtime=cuda" "$RUN/c.log"; then note "배너 OK(runtime=cuda)"; else
  echo "[w4a16] FAIL: 배너 runtime!=cuda — ${RUN}/c.log"; fail=1
fi
C=$(python3 -c "
import json
try:
    d=json.load(open('"$RUN/c.json"'))
    t=d.get('tokens') or (d['choices'][0].get('tokens') if 'choices' in d else None)
    print(','.join(map(str,t)) if t else 'ERR')
except Exception:
    print('ERR')
")
case "$C" in
  "$GOLDEN"*) note "serve 토큰 OK: $(echo "$C" | head -c 60)";;
  *) echo "[w4a16] FAIL: serve 응답 불일치: $(echo "$C" | head -c 60)"; fail=1;;
esac
for J in c2a c2b; do
  CJ=$(python3 -c "
import json
try:
    d=json.load(open('"$RUN/$J.json"'))
    t=d.get('tokens') or (d['choices'][0].get('tokens') if 'choices' in d else None)
    print(','.join(map(str,t)) if t else 'ERR')
except Exception:
    print('ERR')
")
  case "$CJ" in
    "$GOLDEN"*) note "배치 불변성 OK($J)";;
    *) echo "[w4a16] FAIL: 배치 불변성 불일치($J): $(echo "$CJ" | head -c 60)"; fail=1;;
  esac
done

# ── 4. D(35B INT4 g32) 참조 오라클(W4-1 MoE) ──
note "[4/10] D(35B INT4 g32) 참조 실행 — w4a16-ref (MoE g32/bf16)"
timeout 2400 "$BIN" w4a16-ref "$M_35B" --prompt-tokens 148678,65233,202419 --n-predict 8 --ctx 1024 \
  > "$RUN/d.out" 2> "$RUN/d.log"
D=$(grep -m1 '^tokens:' "$RUN/d.out" | sed 's/^tokens: //')
case "$D" in
  "$GOLDEN35"*) note "35B 참조 토큰 OK: $(echo "$D" | head -c 60)";;
  "") echo "[w4a16] FAIL: 35B 참조 실패 — ${RUN}/d.log"; fail=1;;
  *) echo "[w4a16] FAIL: 35B 골든 불일치: $(echo "$D" | head -c 60)"; fail=1;;
esac

# ── 5. D2(35B INT4 g32) GPU 체인(MoE) ──
note "[5/10] D2(35B) GPU 체인 — w4a16-gpu (MoE)"
timeout 1800 "$BIN" w4a16-gpu "$M_35B" --prompt-tokens 148678,65233,202419 --n-predict 8 --ctx 1024 \
  > "$RUN/d2.out" 2> "$RUN/d2.log"
D2=$(grep -m1 '^ tokens:' "$RUN/d2.out" | sed 's/^ tokens: //')
case "$D2" in
  "$GOLDEN35"*) note "35B GPU 토큰 OK: $(echo "$D2" | head -c 60)";;
  "") echo "[w4a16] FAIL: 35B GPU 실패 — ${RUN}/d2.log"; fail=1;;
  *) echo "[w4a16] FAIL: 35B GPU 골든 불일치: $(echo "$D2" | head -c 60)"; fail=1;;
esac

# ── 6. D3(35B) 다중 청크(35토큰) — 배치 프리필 t≤32 CPU/GPU 골든 ──
note "[6/10] D3(35B) 다중 청크 — 참조(CPU) + GPU (배치 t≤32)"
timeout 2400 "$BIN" w4a16-ref "$M_35B" --prompt-tokens "$P35L" --n-predict 8 --ctx 1024 \
  > "$RUN/d3a.out" 2> "$RUN/d3a.log"
DA=$(grep -m1 '^tokens:' "$RUN/d3a.out" | sed 's/^tokens: //')
case "$DA" in
  "$GOLDEN35L"*) note "긴 프롬프트 참조 OK: $(echo "$DA" | head -c 50)";;
  "") echo "[w4a16] FAIL: 긴 참조 실패 — ${RUN}/d3a.log"; fail=1;;
  *) echo "[w4a16] FAIL: 긴 참조 골든 불일치: $(echo "$DA" | head -c 50)"; fail=1;;
esac
timeout 900 "$BIN" w4a16-gpu "$M_35B" --prompt-tokens "$P35L" --n-predict 8 --ctx 1024 \
  > "$RUN/d3b.out" 2> "$RUN/d3b.log"
DB=$(grep -m1 '^ tokens:' "$RUN/d3b.out" | sed 's/^ tokens: //')
case "$DB" in
  "$GOLDEN35L"*) note "긴 프롬프트 GPU OK: $(echo "$DB" | head -c 50)";;
  "") echo "[w4a16] FAIL: 긴 GPU 실패 — ${RUN}/d3b.log"; fail=1;;
  *) echo "[w4a16] FAIL: 긴 GPU 골든 불일치: $(echo "$DB" | head -c 50)"; fail=1;;
esac

# ── 6b. D4(35B) 스트리밍 serve — 전문가 LRU 캐시 경로 골든 ──
note "[6b] D4(35B) 스트리밍 serve — --moe streaming 골든(캐시 경로)"
: > "$RUN/d4.log"
"$BIN" serve --model "$M_35B" --ctx 1024 --moe streaming --spec off --port "$PORT" > "$RUN/d4.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null' EXIT
for i in $(seq 1 100); do
  sleep 3
  grep -m1 "listening" "$RUN/d4.log" >/dev/null 2>&1 && break
done
curl -s -m 600 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
  -d '{"prompt":[148678,65233,202419],"max_tokens":8,"temperature":0}' > "$RUN/d4.json"
kill "$SERVE_PID" 2>/dev/null
wait "$SERVE_PID" 2>/dev/null
trap - EXIT
D4=$(python3 -c "
import json
try:
    d=json.load(open('"$RUN/d4.json"'))
    t=d.get('tokens') or (d['choices'][0].get('tokens') if 'choices' in d else None)
    print(','.join(map(str,t)) if t else 'ERR')
except Exception:
    print('ERR')
")
case "$D4" in
  "$GOLDEN35"*) note "35B 스트리밍 토큰 OK: $(echo "$D4" | head -c 60)";;
  *) echo "[w4a16] FAIL: 35B 스트리밍 골든 불일치: $(echo "$D4" | head -c 60)"; fail=1;;
esac

# ── 7. E(FN FP8PLE) 자원 가드 명시 거부 ──
note "[7/10] E(FN FP8PLE) 자원 가드 명시 거부 판정"
if timeout 60 "$BIN" w4a16-load "$M_FN" > "$RUN/e.log" 2>&1; then
  echo "[w4a16] FAIL: 가드가 통과시킴(120GiB) — ${RUN}/e.log"; fail=1
elif grep -qE "insufficient resources|rsrc-guard" "$RUN/e.log"; then
  note "가드 거부 OK: $(head -1 "$RUN/e.log")"
else
  echo "[w4a16] FAIL: 가드 외 사유 — ${RUN}/e.log"; fail=1
fi

# ── 9. G(27B) 장문 600·4000토큰 — lim>256·다중 청크 ──
note "[8/10] G(27B) 장문 600/4000토큰 GPU 골든"
timeout 900 "$BIN" w4a16-gpu "$M_27B" --prompt-tokens "$P600" --n-predict 8 --ctx 4096 \
  > "$RUN/g1.out" 2> "$RUN/g1.log"
G1=$(grep -m1 '^ tokens:' "$RUN/g1.out" | sed 's/^ tokens: //')
case "$G1" in
  "$GOLDEN27_600"*) note "27B 600 토큰 OK";;
  "") echo "[w4a16] FAIL: 27B 600 실패 — ${RUN}/g1.log"; fail=1;;
  *) echo "[w4a16] FAIL: 27B 600 골든 불일치: $(echo "$G1" | head -c 50)"; fail=1;;
esac
timeout 1200 "$BIN" w4a16-gpu "$M_27B" --prompt-tokens "$P4000" --n-predict 8 --ctx 8192 \
  > "$RUN/g2.out" 2> "$RUN/g2.log"
G2=$(grep -m1 '^ tokens:' "$RUN/g2.out" | sed 's/^ tokens: //')
case "$G2" in
  "$GOLDEN27_4K"*) note "27B 4000 토큰 OK";;
  "") echo "[w4a16] FAIL: 27B 4000 실패 — ${RUN}/g2.log"; fail=1;;
  *) echo "[w4a16] FAIL: 27B 4000 골든 불일치: $(echo "$G2" | head -c 50)"; fail=1;;
esac

# ── 10. G2(35B) 장문 600·4000토큰 ──
note "[9/10] G2(35B) 장문 600/4000토큰 GPU 골든"
timeout 900 "$BIN" w4a16-gpu "$M_35B" --prompt-tokens "$P600" --n-predict 8 --ctx 4096 \
  > "$RUN/g3.out" 2> "$RUN/g3.log"
G3=$(grep -m1 '^ tokens:' "$RUN/g3.out" | sed 's/^ tokens: //')
case "$G3" in
  "$GOLDEN35_600"*) note "35B 600 토큰 OK";;
  "") echo "[w4a16] FAIL: 35B 600 실패 — ${RUN}/g3.log"; fail=1;;
  *) echo "[w4a16] FAIL: 35B 600 골든 불일치: $(echo "$G3" | head -c 50)"; fail=1;;
esac
timeout 1200 "$BIN" w4a16-gpu "$M_35B" --prompt-tokens "$P4000" --n-predict 8 --ctx 8192 \
  > "$RUN/g4.out" 2> "$RUN/g4.log"
G4=$(grep -m1 '^ tokens:' "$RUN/g4.out" | sed 's/^ tokens: //')
case "$G4" in
  "$GOLDEN35_4K"*) note "35B 4000 토큰 OK";;
  "") echo "[w4a16] FAIL: 35B 4000 실패 — ${RUN}/g4.log"; fail=1;;
  *) echo "[w4a16] FAIL: 35B 4000 골든 불일치: $(echo "$G4" | head -c 50)"; fail=1;;
esac

# ── 10. H(스펙 옵트인) --spec 4 골든 접두 ──
note "[10/10] H(스펙) --spec 4 골든 접두 — 27B/35B"
timeout 900 "$BIN" w4a16-gpu "$M_27B" --prompt-tokens 148678,65233,202419 --n-predict 8 --ctx 1024 --spec 4 \
  > "$RUN/h1.out" 2> "$RUN/h1.log"
H1=$(grep -m1 '^ tokens:' "$RUN/h1.out" | sed 's/^ tokens: //')
case "$H1" in
  "$GOLDEN"*) note "27B spec 토큰 OK: $(echo "$H1" | head -c 60)";;
  "") echo "[w4a16] FAIL: 27B spec 실행 실패 — ${RUN}/h1.log"; fail=1;;
  *) echo "[w4a16] FAIL: 27B spec 골든 불일치: $(echo "$H1" | head -c 60)"; fail=1;;
esac
timeout 900 "$BIN" w4a16-gpu "$M_35B" --prompt-tokens 148678,65233,202419 --n-predict 8 --ctx 1024 --spec 4 \
  > "$RUN/h2.out" 2> "$RUN/h2.log"
H2=$(grep -m1 '^ tokens:' "$RUN/h2.out" | sed 's/^ tokens: //')
case "$H2" in
  "$GOLDEN35"*) note "35B spec 토큰 OK: $(echo "$H2" | head -c 60)";;
  "") echo "[w4a16] FAIL: 35B spec 실행 실패 — ${RUN}/h2.log"; fail=1;;
  *) echo "[w4a16] FAIL: 35B spec 골든 불일치: $(echo "$H2" | head -c 60)"; fail=1;;
esac

note "[12/12] K(스펙 항등) — 128토큰 서브: 스펙 on/off 토큰 동일 + 라운드 발화"
# [비결정성 회귀 가드 2026-10-13·커밋 45ae1b08] 융합(ngram) 보정 롤백 오프바이원이
# 스펙 발화 시 후속 토큰을 오염시켰다 — 스펙은 무손실이어야 하므로, 같은 요청의
# 스펙 on/off 출력이 **토큰 단위로 동일**해야 한다. 라운드 미발화면 검사 무효
# (spec_rounds ≥ 1 확인). SPEC_OFF는 진단 플래그(성능 경로만 끔 — 무손실).
K_OK=0
for pair in "on:" "off:LLM170_SPEC_OFF=1"; do
  mode="${pair%%:*}"; envs="${pair#*:}"
  env $envs "$BIN" serve --model "$M_27B" --ctx 4096 --port "$PORT" \
    > "$RUN/k-$mode.log" 2>&1 &
  SERVE_PID=$!
  trap 'kill "$SERVE_PID" 2>/dev/null' EXIT
  for i in $(seq 1 100); do
    sleep 3
    grep -m1 "listening" "$RUN/k-$mode.log" >/dev/null 2>&1 && break
  done
  curl -s -m 600 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d '{"prompt":[148678,65233,202419],"max_tokens":128,"temperature":0}' > "$RUN/k-$mode.json"
  curl -s "http://127.0.0.1:$PORT/stats" > "$RUN/k-$mode.stats" 2>/dev/null
  kill "$SERVE_PID" 2>/dev/null
  wait "$SERVE_PID" 2>/dev/null
  trap - EXIT
done
KK=$(python3 - "$RUN" <<'EOF'
import json, sys
run = sys.argv[1]
def toks(mode):
    d = json.load(open(f"{run}/k-{mode}.json"))
    return d.get("tokens") or []
def rounds(mode):
    try:
        d = json.load(open(f"{run}/k-{mode}.stats"))
        return d["serving"]["spec_rounds"]
    except Exception:
        return -1
t_on, t_off = toks("on"), toks("off")
r_on = rounds("on")
if not t_on or not t_off:
    print("FAIL 출력 없음")
elif t_on != t_off:
    i = next((i for i, (a, b) in enumerate(zip(t_on, t_off)) if a != b), min(len(t_on), len(t_off)))
    print(f"FAIL 스펙 항등 위반 idx {i} on={t_on[i:i+3]} off={t_off[i:i+3]}")
elif r_on < 1:
    print("FAIL 라운드 미발화 — 검사 무효")
else:
    print(f"OK rounds={r_on} tokens={len(t_on)}")
EOF
)
case "$KK" in
  OK*) note "27B 스펙 항등 OK ($KK)";;
  *) echo "[w4a16] FAIL: 스펙 항등 — $KK"; fail=1;;
esac

note "[11/11] J(수치 지문) — 128토큰 창 PPL 동결 대조 (Q2 · 서브토큰 드리프트)"
FP="docs/eval/fingerprint.tsv"   # 로컬 원장(docs/ — 푸시 제외)
for pair in "$M_27B:27B" "$M_35B:35B"; do
  d="${pair%:*}"; n="${pair#*:}"
  jout=$(timeout 900 "$BIN" w4a16-eval ppl "$d" --corpus docs/eval/corpus-en.txt \
          --limit 128 --json 2> "$RUN/j-$n.log" | tail -1)
  if [ -z "$jout" ]; then
    echo "[w4a16] FAIL: $n 수치 지문 실행 실패 — ${RUN}/j-$n.log"; fail=1; continue
  fi
  got=$(printf '%s' "$jout" | python3 -c \
    'import json,sys; d=json.loads(sys.stdin.read()); print(f"{d["ppl"]:.9f} {d["nll_mean"]:.9f}")')
  ref=$(awk -v m="$n" '!/^#/ && $1==m {print $4" "$3}' "$FP")
  if [ -z "$ref" ]; then
    echo "[w4a16] FAIL: $n 지문 기준 없음(${FP})"; fail=1; continue
  fi
  if python3 -c "
import sys
g=sys.argv[1].split(); r=sys.argv[2].split()
ok = abs(float(g[0])-float(r[0])) <= 1e-6 and abs(float(g[1])-float(r[1])) <= 1e-6
sys.exit(0 if ok else 1)
" "$got" "$ref"; then
    note "$n 지문 OK: ppl=$(printf '%s' "$got" | cut -d' ' -f1) (기준 $(printf '%s' "$ref" | cut -d' ' -f1))"
  else
    echo "[w4a16] FAIL: $n 수치 지문 초과 — got [$got] ≠ ref [$ref] (수치 변경 시 재동결 필요)"
    fail=1
  fi
done

if [ $fail -eq 0 ]; then
  echo "[w4a16] smoke: ALL PASS (27B 참조·GPU·serve·장문·spec / 35B MoE 참조+GPU·장문·spec·스트리밍(D4) / E 가드 거부 / 스펙 항등 / 수치 지문) — int8 KV 단일 경로"
else
  echo "[w4a16] smoke: FAILURES PRESENT"
fi
exit $fail
