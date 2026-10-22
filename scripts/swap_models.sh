#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
BIN=./target/release/llm170
gval() { awk -F'\t' -v k="$1" '$1==k{print $2}' scripts/goldens.tsv; }
GOLDEN=$(gval serve27.prefix)
GOLDENFN5=$(gval fn5.prefix)
GOLDEN35=$(gval moe35.prefix)
GOLDEN27_600=$(gval serve27.long600)
GOLDEN27_4K=$(gval serve27.long4k)
GOLDEN35_600=$(gval moe35.long600)
GOLDEN35_4K=$(gval moe35.long4k)
GOLDEN35L=$(gval moe35.chunk35)
P35L="148678,65233,202419,156037,16072,154029,156504,30,149285,65233,195939,149820,152091,150868,54581,155180,177992,12434,170819,149820,152091,16869,201523,229231,204817,13,153343,51643,155497,171699,25845,159667,32762,189291,13"
mkdir -p /tmp/opencode
RUN=$(mktemp -d /tmp/opencode/w4a16-smoke.XXXXXX)
PORT=$((18210 + ($$ % 400)))

SECTIONS=all
if [ $# -ge 1 ] && [ "$1" = "--fast" ]; then
    set -- "--section" "3a,c"
fi
if [ "$1" = "--section" ] && [ $# -eq 2 ]; then
    SECTIONS="${2// /}"
else
    echo "사용: $0 [--fast | --section <키,키,...>] — 키: a,3a,c,3b,d,d2,d3,d4,e,g,g2,h,k,j,g4,g4e" >&2
    exit 2
fi
if [ "$SECTIONS" != all ]; then
    if [ -z "${SECTIONS//,/}" ]; then
        echo "빈 섹션 이름" >&2
        exit 2
    fi
    for k in ${SECTIONS//,/ }; do
        case ",a,3a,c,3b,d,d2,d3,d4,e,g,g2,h,k,j,g4,g4e," in
            *",$k,"*) ;;
            *) echo "unknown section: $k (키: a,3a,c,3b,d,d2,d3,d4,e,g,g2,h,k,j,g4,g4e)" >&2; exit 2;;
        esac
    done
    echo "[w4a16] section 필터: $SECTIONS (나머지 섹션 스킵)"
fi
want() {
    [ "$SECTIONS" = all ] && return 0
    case ",$SECTIONS," in
        *",$1,"*) return 0;;
    esac
    return 1
}

M_27B=../models/Qwen3.8-27B-W4A16-AutoRound
M_35B=../models/Qwen3.6-35B-A3B-INT4-W4A16
M_FN=../models/Qwen3.8-Flash-Next-W4A16-FP8PLE
M_G4=../models/Gemma-4-26B-A4B-it-QAT-W4A16

fail=0
SUMMARY=()
note() { echo "[w4a16] $*"; SUMMARY+=("$*"); }

if want a; then
    note "[1/10] A(27B) 참조 실행 — w4a16-ref"
    timeout 900 "$BIN" w4a16-ref "$M_27B" --prompt-tokens 148678,65233,202419 --n-predict 8 --ctx 1024 \
      > "$RUN/a.out" 2> "$RUN/a.log"
    A=$(grep -m1 '^tokens:' "$RUN/a.out" | sed 's/^tokens: //')
    case "$A" in
      "$GOLDEN"*) note "참조 토큰 OK: $(echo "$A" | head -c 60)";;
      "") echo "[w4a16] FAIL: 참조 실행 실패 — ${RUN}/a.log"; fail=1;;
      *) echo "[w4a16] FAIL: 골든 접두 불일치: $(echo "$A" | head -c 60)"; fail=1;;
    esac
fi

want 3b && note "[2/10] B(27B) GPU 체인 — [3b] 600토큰 런에 흡수(골든 접두 동일 계약)"

if want 3a; then
    note "[3a/10] serve --kv-mem 가드 — 무 auto·최소·호스트 여유 상한 거부·--ctx 폐지"
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
    SV4_G=$(( ( $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0) / 1024
              + $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) ) * 2 ))
    if timeout 120 "$BIN" serve --model "$M_27B" --kv-mem "${SV4_G}G" --port 18098 > "$RUN/sv4.log" 2>&1; then
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
    : > "$RUN/sv3.log"
    LLM170_KV_VRAM_MB=2048 "$BIN" serve --model "$M_27B" --kv-mem 12G --port 18103 > "$RUN/sv3.log" 2>&1 &
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
fi

if want c; then
    note "[3/10] C(27B) serve HTTP — runtime=cuda + 골든 접두"
    : > "$RUN/c.log"
    "$BIN" serve --model "$M_27B" --kv-mem 256M --port "$PORT" > "$RUN/c.log" 2>&1 &
    SERVE_PID=$!
    trap 'kill "$SERVE_PID" 2>/dev/null' EXIT
    for i in $(seq 1 100); do
      sleep 3
      grep -m1 "listening" "$RUN/c.log" >/dev/null 2>&1 && break
    done
    curl -s -m 600 "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d '{"prompt":[148678,65233,202419],"max_tokens":8,"temperature":0}' > "$RUN/c.json"
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
fi

if want 3b; then
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
fi

if want d; then
    note "[4/10] D(35B INT4 g32) 참조 실행 — w4a16-ref (MoE g32/bf16)"
    timeout 2400 "$BIN" w4a16-ref "$M_35B" --prompt-tokens 148678,65233,202419 --n-predict 8 --ctx 1024 \
      > "$RUN/d.out" 2> "$RUN/d.log"
    D=$(grep -m1 '^tokens:' "$RUN/d.out" | sed 's/^tokens: //')
    case "$D" in
      "$GOLDEN35"*) note "35B 참조 토큰 OK: $(echo "$D" | head -c 60)";;
      "") echo "[w4a16] FAIL: 35B 참조 실패 — ${RUN}/d.log"; fail=1;;
      *) echo "[w4a16] FAIL: 35B 골든 불일치: $(echo "$D" | head -c 60)"; fail=1;;
    esac
fi

if want d2; then
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
fi

if want d3; then
    note "[6/10] D3(35B) 다중 청크 참조(CPU) — 배치 프리필 t≤32 골든"
    timeout 2400 "$BIN" w4a16-ref "$M_35B" --prompt-tokens "$P35L" --n-predict 8 --ctx 1024 \
      > "$RUN/d3a.out" 2> "$RUN/d3a.log"
    DA=$(grep -m1 '^tokens:' "$RUN/d3a.out" | sed 's/^tokens: //')
    case "$DA" in
      "$GOLDEN35L"*) note "긴 프롬프트 참조 OK: $(echo "$DA" | head -c 50)";;
      "") echo "[w4a16] FAIL: 긴 참조 실패 — ${RUN}/d3a.log"; fail=1;;
      *) echo "[w4a16] FAIL: 긴 참조 골든 불일치: $(echo "$DA" | head -c 50)"; fail=1;;
    esac
fi

if want d4; then
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
fi

if want e; then
    note "[7/10] E(FN FP8PLE) stage-1 식별 + serve 자원 가드 거부"
    if timeout 60 "$BIN" w4a16-load "$M_FN" > "$RUN/e.log" 2>&1; then
      if grep -q "FN 1단계 식별 통과" "$RUN/e.log" && grep -q "arch=qwen4_exp(FN)" "$RUN/e.log"; then
        note "stage-1 식별 통과 OK: $(grep -m1 '판정:' "$RUN/e.log")"
      else
        echo "[w4a16] FAIL: stage-1 판정 문구 불명 — ${RUN}/e.log"; fail=1
      fi
    else
      echo "[w4a16] FAIL: w4a16-load 실패(stage-1 통과 기대) — ${RUN}/e.log"; fail=1
    fi
    if timeout 60 "$BIN" serve --model "$M_FN" --kv-mem 1G --port 18099 > "$RUN/e2.log" 2>&1; then
      echo "[w4a16] FAIL: serve 가드가 통과시킴(148.7GiB) — ${RUN}/e2.log"; fail=1
    elif grep -q "insufficient resources" "$RUN/e2.log"; then
      note "serve 가드 거부 OK: $(head -1 "$RUN/e2.log")"
    elif grep -q "미지원 아키텍처" "$RUN/e2.log"; then
      note "serve 아치 거부 OK(FN serve 미배선): $(head -1 "$RUN/e2.log")"
    else
      echo "[w4a16] FAIL: serve 가드 외 사유(락 대기 포함) — ${RUN}/e2.log"; fail=1
    fi
    if timeout 420 "$BIN" w4a16-gpu "$M_FN" --prompt-tokens 148678,65233,202419,771,99012 \
      > "$RUN/e3.out" 2> "$RUN/e3.log"; then
      E3=$(grep -m1 'argmax ' "$RUN/e3.out" | grep -o 'argmax [0-9]*' | awk '{print $2}')
      case "$E3" in
        "$GOLDENFN5") note "FN 5톤 GPU 골든 OK: argmax $E3";;
        "") echo "[w4a16] FAIL: FN 5톤 argmax 판독 불가 — ${RUN}/e3.log"; fail=1;;
        *) echo "[w4a16] FAIL: FN 5톤 골든 불일치: $E3 ≠ $GOLDENFN5 — ${RUN}/e3.log"; fail=1;;
      esac
    else
      echo "[w4a16] FAIL: FN 5톤 GPU 실행 실패 — ${RUN}/e3.log"; fail=1
    fi
fi

if want g; then
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
fi

if want g2; then
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
fi

if want h; then
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
fi

if want k; then
    note "[12/12] K(스펙 항등) — 128토큰 서브: 스펙 on/off 토큰 동일 + 라운드 발화 + 골든 접두"
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
    print(f"FAIL 골든 접두 불일치(스펙 on): got={t_on[:len(gold)]} want={gold}")
else:
    print(f"OK rounds={r_on} tokens={len(t_on)} golden-prefix OK")
EOF
    )
    case "$KK" in
      OK*) note "27B 스펙 항등 OK ($KK)";;
      *) echo "[w4a16] FAIL: 스펙 항등 — $KK"; fail=1;;
    esac
fi

if want j; then
    note "[11/11] J(수치 지문) — 128토큰 창 PPL 동결 대조 (Q2 · 서브토큰 드리프트)"
    FP="scripts/eval-data/fingerprint.tsv"
    for pair in "$M_27B:27B:" "$M_35B:35B:" "$M_G4:gemma4:--bos 2"; do
      d="${pair%%:*}"; rest="${pair#*:}"; n="${rest%%:*}"; extra="${rest#*:}"
      jout=$(timeout 900 "$BIN" w4a16-eval ppl "$d" --corpus scripts/eval-data/corpus-en.txt \
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
fi

if want g4; then
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
      A=$(timeout 900 "$BIN" bench --model "$M_G4" --pp 16 --tg 8 --np 16 --ctx 1024 2>&1)
      case "$A" in
        *"np16-check | 단일 경로 등가 OK"*) note "G4f np16 등가 OK";;
        *) echo "[w4a16] FAIL: G4f np16 — $(echo "$A" | grep -o 'np16-check.*' | head -1)"; fail=1;;
      esac
      echo "=== G4c. gemma4 HTTP 서브(채팅 · 아치 렌더·정지 토큰) ==="
      "$BIN" serve --model "$M_G4" --kv-mem 128M --port 18099 > "$RUN/g4c.log" 2>&1 &
      G4_PID=$!
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
fi

if want g4e; then
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
t = open('scripts/eval-data/corpus-long.txt', encoding='utf-8').read()[:6000]
for name, mt in [("rB", 16), ("rA", 8)]:
    body = {"model": "m", "messages": [{"role": "user", "content": t}],
            "max_tokens": mt, "temperature": 0}
    open(f'{run}/reuse-{name}.json', 'w', encoding='utf-8').write(json.dumps(body))
PYG4E
    for pair in "$M_G4:gemma4:18101:1G" "$M_27B:qwen27:18102:512M"; do
      d="${pair%%:*}"; rest="${pair#*:}"; n="${rest%%:*}"; rest="${rest#*:}"; port="${rest%%:*}"; kv="${rest#*:}"
      "$BIN" serve --model "$d" --kv-mem "$kv" --port "$port" > "$RUN/g4e-$n.log" 2>&1 &
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
fi

echo "[w4a16] ── 검수 요약(${#SUMMARY[@]}항목) ──"
for s in "${SUMMARY[@]}"; do echo "[w4a16]   $s"; done
if [ $fail -eq 0 ]; then
    echo "[w4a16] smoke: ALL PASS (27B 참조·GPU(600 흡수)·serve·장문(2시나리오 1런)·spec·경계 교차+np 자체 검증 / 35B MoE 참조+GPU(2시나리오)·장문(2시나리오)·spec·스트리밍(D4) / gemma4 v0 게이트+infer(G4)+np16 등가 / E stage-1 식별+serve 가드 거부·kv-mem 가드 거부 / 스펙 항등 / 수치 지문) — int8 KV 단일 경로 [section=$SECTIONS]"
else
    echo "[w4a16] smoke: FAILURES PRESENT [section=$SECTIONS]"
fi
exit $fail
