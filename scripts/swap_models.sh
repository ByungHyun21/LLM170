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
M_G4=../models/Gemma-4-26B-A4B-it-QAT-W4A16

fail=0
SUMMARY=()
note() { echo "[w4a16] $*"; SUMMARY+=("$*"); }

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

# ── 2. B(27B) GPU 체인 — [검수 통합 2026-10-14] [3b]의 600토큰 프로브가 같은
# 골든 접두를 판정한다(모델 로드 1회 절감) — 별도 8토큰 런 제거.
note "[2/10] B(27B) GPU 체인 — [3b] 600토큰 런에 흡수(골든 접두 동일 계약)"

# ── 3a. serve kv-mem 가드 (W1+W3 — 필수·최소·호스트 여유 상한·--ctx 폐지) ──
note "[3a/10] serve --kv-mem 가드 — 무 auto·최소·호스트 여유 상한 거부·--ctx 폐지"
# [2026-10-16 수정] 종전 "23G 거부" 기대는 W1(VRAM 단일 풀) 시절 값 — W3 호스트
# 2계층에서 kv-mem > VRAM은 정상(초과분 = 핀드 호스트 아레나). 그 검사가
# timeout 없이 serve 대기에 물려 W1 작성 후 첫 실행에서 스모크가 정지했다.
# 새 계약: ① 실제 상한 = 호스트부 ≤ 가용 RAM − 예약(100G 그물) ② VRAM 초과분은
# 호스트 분할 수용(기동 확인 후 종료) ③ 모든 프로브 timeout.
if timeout 120 "$BIN" serve --model "$M_27B" --port 18098 > "$RUN/sv0.log" 2>&1; then
  echo "[w4a16] FAIL: --kv-mem 누락이 통과됨"; fail=1
elif grep -q -- "--kv-mem" "$RUN/sv0.log"; then note "kv-mem 필수 거부 OK: $(head -1 "$RUN/sv0.log")"
else
  echo "[w4a16] FAIL: kv-mem 누락 사유 불명 — ${RUN}/sv0.log"; fail=1
fi
if timeout 120 "$BIN" serve --model "$M_27B" --kv-mem 1M --port 18098 > "$RUN/sv1.log" 2>&1; then
  echo "[w4a16] FAIL: --kv-mem 1M(블록 미만)이 통과됨"; fail=1
elif grep -q "kv-mem 가드 거부" "$RUN/sv1.log"; then note "kv-mem 최소 거부 OK"
else
  echo "[w4a16] FAIL: kv-mem 최소 거부 사유 불명 — ${RUN}/sv1.log"; fail=1
fi
if timeout 120 "$BIN" serve --model "$M_27B" --kv-mem 100G --port 18098 > "$RUN/sv4.log" 2>&1; then
  echo "[w4a16] FAIL: --kv-mem 100G(호스트 여유 초과)가 통과됨"; fail=1
elif grep -q "kv-mem 가드 거부" "$RUN/sv4.log"; then note "kv-mem 상한(호스트 여유) 거부 OK: $(grep -m1 'kv-mem 가드 거부' "$RUN/sv4.log" | cut -c1-80)"
else
  echo "[w4a16] FAIL: kv-mem 상한 거부 사유 불명 — ${RUN}/sv4.log"; fail=1
fi
if timeout 120 "$BIN" serve --model "$M_27B" --kv-mem 8G --ctx 4096 --port 18098 > "$RUN/sv2.log" 2>&1; then
  echo "[w4a16] FAIL: serve --ctx가 통과됨(폐지 계약 위반)"; fail=1
elif grep -q "ctx는 폐지" "$RUN/sv2.log"; then note "--ctx 폐지 거부 OK"
else
  echo "[w4a16] FAIL: --ctx 폐지 사유 불명 — ${RUN}/sv2.log"; fail=1
fi
# ② W3 분할 수용 — kv-mem 12G(VRAM부 상한 2.7G, 초과분 = 호스트 2계층).
: > "$RUN/sv3.log"
"$BIN" serve --model "$M_27B" --kv-mem 12G --port 18103 > "$RUN/sv3.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null' EXIT
for i in $(seq 1 100); do
  sleep 3
  grep -m1 "listening" "$RUN/sv3.log" >/dev/null 2>&1 && break
done
kill "$SERVE_PID" 2>/dev/null
wait "$SERVE_PID" 2>/dev/null
trap - EXIT
if grep -m1 "listening" "$RUN/sv3.log" >/dev/null 2>&1 && grep -q "# kv-tier:" "$RUN/sv3.log"; then
  note "kv-mem 초과분 호스트 분할 수용 OK: $(grep -m1 '# kv-tier:' "$RUN/sv3.log" | cut -c1-80)"
else
  echo "[w4a16] FAIL: kv-mem 12G 호스트 분할 수용 — ${RUN}/sv3.log"; fail=1
fi

# ── 3. C(27B) serve HTTP 종단 ──
note "[3/10] C(27B) serve HTTP — runtime=cuda + 골든 접두"
: > "$RUN/c.log"
"$BIN" serve --model "$M_27B" --kv-mem 128M --port "$PORT" > "$RUN/c.log" 2>&1 &
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

# ── 3b. 그래프 replay KV 블록(512) 경계 교차 — 디코드 중 성장 경로 ──
# [2026-10-14] 그래프 경로가 kv_ensure(온디맨드 성장)를 우회해 경계 교차 시
# 미할당 테이블 엔트리를 판독(illegal access — np2 tg768 실측)하던 결함의 가드.
# 프리필이 블록을 선확보하는 장문 배터리·스모크 경로에서는 잠복했던 클래스.
note "[3b] 그래프 경로 KV 경계 교차 — 단문→600토큰(단일 그래프+배치 그래프)"
timeout 900 "$BIN" w4a16-gpu "$M_27B" --prompt-tokens 148678,65233,202419 --n-predict 600 --ctx 1024 \
  > "$RUN/c3.out" 2> "$RUN/c3.log"
C3=$(grep -m1 '^ tokens:' "$RUN/c3.out" | sed 's/^ tokens: //')
case "$C3" in
  "$GOLDEN"*) note "단일 그래프 교차 OK + GPU 골든 접두([2] 흡수) ($(echo "$C3" | awk -F, '{print NF}')토큰)";;
  "") echo "[w4a16] FAIL: 단일 그래프 교차 실패(크래시?) — ${RUN}/c3.log"; fail=1;;
  *) echo "[w4a16] FAIL: 단일 그래프 교차 골든 불일치: $(echo "$C3" | head -c 60)"; fail=1;;
esac
timeout 900 "$BIN" bench --model "$M_27B" --np 2 --tg 600 --reps 1 --ctx 2048 --np-check 600 \
  > "$RUN/c3b.out" 2> "$RUN/c3b.log"
if grep -q "^np2(tg600)" "$RUN/c3b.out" && grep -q "^np2-check | 단일 경로 등가 OK" "$RUN/c3b.out"; then
  note "배치 그래프 교차 OK + np 자체 검증(단일 등가) OK ($(grep -m1 '^np2' "$RUN/c3b.out" | cut -d'|' -f3 | sed 's/^ *//'))"
  note "  np2 해시: $(grep -m1 '^np2-hash' "$RUN/c3b.out" | cut -d'|' -f2 | sed 's/^ *//')"
else
  echo "[w4a16] FAIL: 배치 그래프 교차/자체 검증 실패(크래시?) — ${RUN}/c3b.log"; fail=1
fi

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

# ── 5. D2+D3(35B) GPU 체인 + 다중 청크(GPU) — [검수 통합 2026-10-14] 시나리오
# 2종(단문·P35L)을 모델 로드 1회로. 시나리오별 `s0/s1-tokens:` 판정.
note "[5/10] D2+D3(35B) GPU 체인 — 다중 시나리오(단문+P35L·MoE)"
timeout 1800 "$BIN" w4a16-gpu "$M_35B" --prompt-tokens 148678,65233,202419 --prompt-tokens "$P35L" \
  --n-predict 8 --ctx 1024 > "$RUN/d2.out" 2> "$RUN/d2.log"
D2=$(grep -m1 '^ s0-tokens:' "$RUN/d2.out" | sed 's/^ s0-tokens: //')
D3G=$(grep -m1 '^ s1-tokens:' "$RUN/d2.out" | sed 's/^ s1-tokens: //')
case "$D2" in
  "$GOLDEN35"*) note "35B GPU 토큰 OK: $(echo "$D2" | head -c 60)";;
  "") echo "[w4a16] FAIL: 35B GPU 실패 — ${RUN}/d2.log"; fail=1;;
  *) echo "[w4a16] FAIL: 35B GPU 골든 불일치: $(echo "$D2" | head -c 60)"; fail=1;;
esac
case "$D3G" in
  "$GOLDEN35L"*) note "긴 프롬프트 GPU OK: $(echo "$D3G" | head -c 50)";;
  "") echo "[w4a16] FAIL: 긴 GPU 실패 — ${RUN}/d2.log"; fail=1;;
  *) echo "[w4a16] FAIL: 긴 GPU 골든 불일치: $(echo "$D3G" | head -c 50)"; fail=1;;
esac

# ── 6. D3(35B) 다중 청크(35토큰) 참조(CPU) ──
note "[6/10] D3(35B) 다중 청크 참조(CPU) — 배치 프리필 t≤32 골든"
timeout 2400 "$BIN" w4a16-ref "$M_35B" --prompt-tokens "$P35L" --n-predict 8 --ctx 1024 \
  > "$RUN/d3a.out" 2> "$RUN/d3a.log"
DA=$(grep -m1 '^tokens:' "$RUN/d3a.out" | sed 's/^tokens: //')
case "$DA" in
  "$GOLDEN35L"*) note "긴 프롬프트 참조 OK: $(echo "$DA" | head -c 50)";;
  "") echo "[w4a16] FAIL: 긴 참조 실패 — ${RUN}/d3a.log"; fail=1;;
  *) echo "[w4a16] FAIL: 긴 참조 골든 불일치: $(echo "$DA" | head -c 50)"; fail=1;;
esac

# ── 6b. D4(35B) 스트리밍 serve — 전문가 LRU 캐시 경로 골든 ──
note "[6b] D4(35B) 스트리밍 serve — --moe streaming 골든(캐시 경로)"
: > "$RUN/d4.log"
"$BIN" serve --model "$M_35B" --kv-mem 128M --moe streaming --spec off --port "$PORT" > "$RUN/d4.log" 2>&1 &
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

# ── 8. G(27B) 장문 600·4000토큰 — lim>256·다중 청크
# [검수 통합 2026-10-14] 2시나리오 1런(모델 로드 1회) — s0=600·s1=4000.
note "[8/10] G(27B) 장문 600/4000토큰 GPU 골든(2시나리오 1런)"
timeout 1800 "$BIN" w4a16-gpu "$M_27B" --prompt-tokens "$P600" --prompt-tokens "$P4000" \
  --n-predict 8 --ctx 8192 > "$RUN/g.out" 2> "$RUN/g.log"
G1=$(grep -m1 '^ s0-tokens:' "$RUN/g.out" | sed 's/^ s0-tokens: //')
G2=$(grep -m1 '^ s1-tokens:' "$RUN/g.out" | sed 's/^ s1-tokens: //')
case "$G1" in
  "$GOLDEN27_600"*) note "27B 600 토큰 OK";;
  "") echo "[w4a16] FAIL: 27B 600 실패 — ${RUN}/g.log"; fail=1;;
  *) echo "[w4a16] FAIL: 27B 600 골든 불일치: $(echo "$G1" | head -c 50)"; fail=1;;
esac
case "$G2" in
  "$GOLDEN27_4K"*) note "27B 4000 토큰 OK";;
  "") echo "[w4a16] FAIL: 27B 4000 실패 — ${RUN}/g.log"; fail=1;;
  *) echo "[w4a16] FAIL: 27B 4000 골든 불일치: $(echo "$G2" | head -c 50)"; fail=1;;
esac

# ── 10. G2(35B) 장문 600·4000토큰 — [검수 통합] 2시나리오 1런.
note "[9/10] G2(35B) 장문 600/4000토큰 GPU 골든(2시나리오 1런)"
timeout 1800 "$BIN" w4a16-gpu "$M_35B" --prompt-tokens "$P600" --prompt-tokens "$P4000" \
  --n-predict 8 --ctx 8192 > "$RUN/g2.out" 2> "$RUN/g2.log"
G3=$(grep -m1 '^ s0-tokens:' "$RUN/g2.out" | sed 's/^ s0-tokens: //')
G4=$(grep -m1 '^ s1-tokens:' "$RUN/g2.out" | sed 's/^ s1-tokens: //')
case "$G3" in
  "$GOLDEN35_600"*) note "35B 600 토큰 OK";;
  "") echo "[w4a16] FAIL: 35B 600 실패 — ${RUN}/g2.log"; fail=1;;
  *) echo "[w4a16] FAIL: 35B 600 골든 불일치: $(echo "$G3" | head -c 50)"; fail=1;;
esac
case "$G4" in
  "$GOLDEN35_4K"*) note "35B 4000 토큰 OK";;
  "") echo "[w4a16] FAIL: 35B 4000 실패 — ${RUN}/g2.log"; fail=1;;
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

note "[12/12] K(스펙 항등) — 128토큰 서브: 스펙 on/off 토큰 동일 + 라운드 발화 + 골든 접두"
# [비결정성 회귀 가드 2026-10-13·커밋 45ae1b08] 융합(ngram) 보정 롤백 오프바이원이
# 스펙 발화 시 후속 토큰을 오염시켰다 — 스펙은 무손실이어야 하므로, 같은 요청의
# 스펙 on/off 출력이 **토큰 단위로 동일**해야 한다. 라운드 미발화면 검사 무효
# (spec_rounds ≥ 1 확인). SPEC_OFF는 진단 플래그(성능 경로만 끔 — 무손실).
K_OK=0
for pair in "on:" "off:LLM170_SPEC_OFF=1"; do
  mode="${pair%%:*}"; envs="${pair#*:}"
  env $envs "$BIN" serve --model "$M_27B" --kv-mem 512M --port "$PORT" \
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
KK=$(python3 - "$RUN" "$GOLDEN" <<'EOF'
import json, sys
run = sys.argv[1]
gold = [int(x) for x in sys.argv[2].split(",")]
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
elif t_on[:len(gold)] != gold:
    # [무료 다중 검증 2026-10-15] on==off만으로는 둘이 함께 드리프트해도 통과한다 —
    # 접두를 동결 골든과 대조(이미 수집된 토큰 — 추가 로드 0; H는 프로브 경로,
    # 여기는 서브+스펙 경로의 골든 셀).
    print(f"FAIL 골든 접두 불일치(스펙 on): got={t_on[:len(gold)]} want={gold}")
else:
    print(f"OK rounds={r_on} tokens={len(t_on)} golden-prefix OK")
EOF
)
case "$KK" in
  OK*) note "27B 스펙 항등 OK ($KK)";;
  *) echo "[w4a16] FAIL: 스펙 항등 — $KK"; fail=1;;
esac

note "[11/11] J(수치 지문) — 128토큰 창 PPL 동결 대조 (Q2 · 서브토큰 드리프트)"
FP="docs/eval/fingerprint.tsv"   # 로컬 원장(docs/ — 푸시 제외)
# [gemma4 2026-10-14] 세 번째 = gemma4 — 비채팅 프로토콜(add_bos 강제 → --bos 2).
for pair in "$M_27B:27B:" "$M_35B:35B:" "$M_G4:gemma4:--bos 2"; do
  d="${pair%%:*}"; rest="${pair#*:}"; n="${rest%%:*}"; extra="${rest#*:}"
  jout=$(timeout 900 "$BIN" w4a16-eval ppl "$d" --corpus docs/eval/corpus-en.txt \
          --limit 128 $extra --json 2> "$RUN/j-$n.log" | tail -1)
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

# ── G4. gemma4-26B-A4B(QAT W4A16 g32) — P3 v0 서빙 경로 ──
# [gemma4 P3 2026-10-14] CPU 오라클 동일 실행 대조(마진 인지 게이트) + infer 골든.
# 프롬프트 = tokenize --chat '{"messages":[{"role":"user","content":"안녕하세요"}]}'.
G4_PROMPT="2,105,2364,107,61659,106,107,105,4368,107,100,45518,107,101"
G4_GOLDEN="61659,236888,131147,237372,199825,236761,94485,237293,220757"
if [ -d "$M_G4" ]; then
  echo "=== G4. gemma4-26B-A4B v0 체인(GPU↔CPU 동일 실행 · 채팅) ==="
  A=$(timeout 1800 "$BIN" w4a16-gpu "$M_G4" --prompt-tokens "$G4_PROMPT" --n-predict 9 --ctx 128 2>&1)
  echo "$A" | grep -E "GPU tokens|판정:" | sed 's/^/  /'
  case "$(echo "$A" | grep -o 'GPU tokens: [0-9,]*')" in
    "GPU tokens: $G4_GOLDEN"*) note "G4 게이트 골든 OK";;
    *) echo "[w4a16] FAIL: G4 게이트 토큰 — $(echo "$A" | grep -o 'GPU tokens: [0-9,]*')"; fail=1;;
  esac
  case "$(echo "$A" | grep -o '판정: 일치')" in
    "판정: 일치") note "G4 판정 OK(일치)";;
    *) echo "[w4a16] FAIL: G4 판정 — $(echo "$A" | grep '판정:')"; fail=1;;
  esac
  echo "=== G4f. gemma4 np16 배치 등가 — t≤16 커널 상시 그물 ==="
  # [2.N 2026-10-15] 15ba3702가 호스트 청크 상한만 올리고 gptq4의 t-row GEMV
  # 케이스(9..16)를 빠뜨린 회귀의 검출 그물 — 케이스 부재면 t=16 조건 커널이
  # 조용히 무기록(no-op)되어 np16-check가 단일 경로 등가 FAIL로 터진다.
  # 디코드 배치 t=16은 서빙(다중 슬롯 동시 배치)에서 실제로 타는 경로다.
  A=$(timeout 900 "$BIN" bench --model "$M_G4" --pp 16 --tg 8 --np 16 --ctx 1024 2>&1)
  case "$A" in
    *"np16-check | 단일 경로 등가 OK"*) note "G4f np16 등가 OK";;
    *) echo "[w4a16] FAIL: G4f np16 — $(echo "$A" | grep -o 'np16-check.*' | head -1)"; fail=1;;
  esac
  echo "=== G4c. gemma4 HTTP 서브(채팅 · 아치 렌더·정지 토큰) ==="
  # [W1] --ctx(serve) 폐지 — kv-mem ≥ 블록 1개(512토큰)면 충분(요청은 소형).
  "$BIN" serve --model "$M_G4" --kv-mem 128M --port 18099 > "$RUN/g4c.log" 2>&1 &
  G4_PID=$!
  # A2 규율: 이 실행의 PID만 종료(전역 pkill 금지).
  trap 'kill "$G4_PID" 2>/dev/null' EXIT
  for i in $(seq 1 100); do
    sleep 3
    grep -m1 "listening" "$RUN/g4c.log" >/dev/null 2>&1 && break
  done
  curl -s -m 600 "http://127.0.0.1:18099/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"gemma4","messages":[{"role":"user","content":"안녕하세요"}],"max_tokens":12,"temperature":0}' > "$RUN/g4c.json"
  kill "$G4_PID" 2>/dev/null
  wait "$G4_PID" 2>/dev/null
  trap - EXIT
  D=$(python3 -c "
import json
try:
    d=json.load(open('"$RUN/g4c.json"'))
    print(','.join(map(str,d['tokens'])) if d.get('tokens') else 'ERR')
except Exception:
    print('ERR')
")
  case "$D" in
    "$G4_GOLDEN"*) note "G4c HTTP 골든 OK: $(echo "$D" | head -c 60)";;
    *) echo "[w4a16] FAIL: G4c HTTP 토큰 — $(echo "$D" | head -c 60)"; fail=1;;
  esac
  echo "=== G4b. gemma4 infer(서빙 경로) ==="
  B=$(timeout 900 "$BIN" infer --model "$M_G4" --prompt-tokens "$G4_PROMPT" --n-predict 9 --ctx 128 2>&1)
  C=$(echo "$B" | grep -o '"token":[0-9]*' | cut -d: -f2 | paste -sd, -)
  case "$C" in
    "$G4_GOLDEN"*) note "G4b infer 골든 OK: $(echo "$C" | head -c 60)";;
    *) echo "[w4a16] FAIL: G4b infer 토큰 — $(echo "$C" | head -c 60)"; fail=1;;
  esac
  # G4d: 스트리밍 전문가(비상주 — 저하 사다리) HTTP 경로. [2026-10-14] t=1
  # 단일 청크 전환이 t행 MoE(g4_moe_host_t — 스트리밍 미지원)로 폴백을
  # 오염시킨 회귀를 잡는 그물 — 비상주 구성은 v0 호스트 체인(g4_step)이 정본.
  echo "=== G4d. gemma4 HTTP 서브(--moe streaming — 저하 사다리) ==="
  "$BIN" serve --model "$M_G4" --kv-mem 128M --moe streaming --port 18100 > "$RUN/g4d.log" 2>&1 &
  G4_PID=$!
  trap 'kill "$G4_PID" 2>/dev/null' EXIT
  for i in $(seq 1 100); do
    sleep 3
    grep -m1 "listening" "$RUN/g4d.log" >/dev/null 2>&1 && break
  done
  curl -s -m 600 "http://127.0.0.1:18100/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"model":"gemma4","messages":[{"role":"user","content":"안녕하세요"}],"max_tokens":12,"temperature":0}' > "$RUN/g4d.json"
  kill "$G4_PID" 2>/dev/null
  wait "$G4_PID" 2>/dev/null
  trap - EXIT
  D=$(python3 -c "
import json
try:
    d=json.load(open('"$RUN/g4d.json"'))
    print(','.join(map(str,d['tokens'])) if d.get('tokens') else 'ERR')
except Exception:
    print('ERR')
")
  case "$D" in
    "$G4_GOLDEN"*) note "G4d 스트리밍 골든 OK: $(echo "$D" | head -c 60)";;
    *) echo "[w4a16] FAIL: G4d 스트리밍 토큰 — $(echo "$D" | head -c 60)"; fail=1;;
  esac
fi

# ── G4e. 접두 재사용 정합 — 콜드↔웜 토큰 동일 + 부분 재사용 발화 ──
# [2026-10-14] 재사용 Stage 1(엔진 pos 복원·finish 불변식 클램프·빈 프리필
# 가드)의 상시 그물: 동일 프롬프트를 (콜드 B1) → (절단 A·max8) → (웜 B2)
# 순서로 보내 B1==B2(토큰열)와 `# prefix-cache` 발화를 판정한다. qwen 동반
# (체크포인트 부분 경로 — 공유 assign_slot 편집의 무영향 확인).
cat > "$RUN/g4e_cmp.py" <<'PYCMP'
import json, sys
def toks(p):
    with open(p) as f:
        return json.load(f).get("tokens", [])
t1, t2 = toks(sys.argv[1]), toks(sys.argv[2])
sys.exit(0 if t1 and t1 == t2 else 1)
PYCMP
python3 - "$RUN" <<'PYG4E'
import json, sys
run = sys.argv[1]
t = open('docs/eval/corpus-long.txt', encoding='utf-8').read()[:6000]
for name, mt in [("rB", 16), ("rA", 8)]:
    body = {"model": "m", "messages": [{"role": "user", "content": t}],
            "max_tokens": mt, "temperature": 0}
    open(f'{run}/reuse-{name}.json', 'w', encoding='utf-8').write(json.dumps(body))
PYG4E
for pair in "$M_G4:gemma4:18101" "$M_27B:qwen27:18102"; do
  d="${pair%%:*}"; rest="${pair#*:}"; n="${rest%%:*}"; port="${rest#*:}"
  "$BIN" serve --model "$d" --kv-mem 512M --port "$port" > "$RUN/g4e-$n.log" 2>&1 &
  G4E_PID=$!
  trap 'kill "$G4E_PID" 2>/dev/null' EXIT
  for i in $(seq 1 100); do
    sleep 3
    grep -m1 "listening" "$RUN/g4e-$n.log" >/dev/null 2>&1 && break
  done
  curl -s -m 600 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    --data-binary "@$RUN/reuse-rB.json" > "$RUN/g4e-$n-b1.json"
  curl -s -m 600 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    --data-binary "@$RUN/reuse-rA.json" > "$RUN/g4e-$n-a.json"
  curl -s -m 600 "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
    --data-binary "@$RUN/reuse-rB.json" > "$RUN/g4e-$n-b2.json"
  kill "$G4E_PID" 2>/dev/null
  wait "$G4E_PID" 2>/dev/null
  trap - EXIT
  if python3 "$RUN/g4e_cmp.py" "$RUN/g4e-$n-b1.json" "$RUN/g4e-$n-b2.json"; then
    if grep -q "# prefix-cache: slot" "$RUN/g4e-$n.log"; then
      note "G4e $n 재사용 정합 OK($(grep -c '# prefix-cache' "$RUN/g4e-$n.log")회 발화)"
    else
      echo "[w4a16] FAIL: G4e $n — 콜드==웜이나 재사용 미발화(캐시 경로 미진입)"; fail=1
    fi
  else
    echo "[w4a16] FAIL: G4e $n — 콜드↔웜 토큰 불일치"; fail=1
  fi
done

echo "[w4a16] ── 검수 요약(${#SUMMARY[@]}항목) ──"
for s in "${SUMMARY[@]}"; do echo "[w4a16]   $s"; done
if [ $fail -eq 0 ]; then
  echo "[w4a16] smoke: ALL PASS (27B 참조·GPU(600 흡수)·serve·장문(2시나리오 1런)·spec·경계 교차+np 자체 검증 / 35B MoE 참조+GPU(2시나리오)·장문(2시나리오)·spec·스트리밍(D4) / gemma4 v0 게이트+infer(G4)+np16 등가 / E 가드·kv-mem 가드 거부 / 스펙 항등 / 수치 지문) — int8 KV 단일 경로"
else
  echo "[w4a16] smoke: FAILURES PRESENT"
fi
exit $fail
