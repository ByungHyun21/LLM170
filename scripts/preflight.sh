#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0

echo "== 1/6 주석 0 =="
if out=$(python3 scripts/comment_check.py 2>&1); then
    echo "OK"
else
    echo "FAIL"; printf '%s\n' "$out" | head -20; fail=1
fi

echo "== 2/6 rustfmt =="
if cargo fmt --all -- --check >/dev/null 2>&1; then
    echo "OK"
else
    echo "FAIL — cargo fmt --all 적용 후 재시도"; cargo fmt --all -- --check | head -10; fail=1
fi

echo "== 3/6 clippy =="
if cargo clippy --workspace --all-targets -- -D warnings >/dev/null 2>&1; then
    echo "OK"
else
    echo "FAIL:"; cargo clippy --workspace --all-targets -- -D warnings 2>&1 | grep -E "^(error|warning)" | sort | uniq -c | head -10; fail=1
fi

echo "== 4/6 cargo 경고 0 (dev check — 릴리스는 빌드 게이트가 담당) =="
w=$(cargo check --workspace --all-targets 2>&1 | grep -c "^warning" || true)
if [[ "$w" -eq 0 ]]; then echo "OK"; else echo "FAIL — warning ${w}건"; fail=1; fi

echo "== 5/6 env 동치 (108 P1) =="
if cargo build --release -q -p llm170-server >/dev/null 2>&1; then
    if ! ./target/release/llm170 diag envcheck >/dev/null; then
        ./target/release/llm170 diag envcheck; fail=1
    else
        echo "OK"
    fi
else
    echo "FAIL — 빌드 오류"; fail=1
fi

echo "== 6/6 스테일 fatbin 가드 =="
A=crates/backend-gpu/src/rawcuda/assets
stale=0
for cu in "$A"/*.cu; do
    fb="${cu%.cu}.fatbin"
    if [[ ! -f "$fb" || "$cu" -nt "$fb" ]]; then
        echo "FAIL — $(basename "$cu") 가 fatbin보다 새로움(리빌드: scripts/build_cuda_kernels.sh)"; stale=1
    fi
done
for n in gptq4 norm ew gdn attn embed; do
    fb="$A/$n.fatbin"
    if [[ ! -f "$fb" || "$A/cast_common.cuh" -nt "$fb" ]]; then
        echo "FAIL — cast_common.cuh 가 $n.fatbin보다 새로움(리빌드 필요)"; stale=1
    fi
done
for n in gptq4 moe; do
    fb="$A/$n.fatbin"
    if [[ ! -f "$fb" || "$A/g4_common.cuh" -nt "$fb" ]]; then
        echo "FAIL — g4_common.cuh 가 $n.fatbin보다 새로움(리빌드 필요)"; stale=1
    fi
done
if [ ! -f "$A/FATBINS.manifest" ]; then
    echo "FAIL — FATBINS.manifest 부재(scripts/build_cuda_kernels.sh로 생성)"; stale=1
else
    while read -r s f n; do
        [ -n "${n:-}" ] || continue
        if [ "$n" = marlin ]; then
            cs=$(cat "$A/marlin.cu" $A/marlin/*.h $A/marlin/*.cuh $A/marlin/core/*.hpp \
                $A/marlin/torch/headeronly/util/Exception.h | sha256sum | cut -d' ' -f1)
        elif [ "$n" = marlin_g32 ]; then
            cs=$(cat "$A/marlin_g32.cu" $A/marlin/*.h $A/marlin/*.cuh $A/marlin/core/*.hpp \
                $A/marlin/torch/headeronly/util/Exception.h | sha256sum | cut -d' ' -f1)
        elif [ "$n" = marlin_moe ]; then
            cs=$(cat "$A/marlin_moe.cu" $A/marlin_moe/*.h \
                $A/marlin_moe/quantization/marlin/*.h* $A/marlin_moe/core/*.hpp \
                $A/marlin_moe/torch/headeronly/util/Exception.h | sha256sum | cut -d' ' -f1)
        else
            cs=$(cat "$A/$n.cu" "$A/cast_common.cuh" "$A/g4_common.cuh" | sha256sum | cut -d' ' -f1)
        fi
        cf=$(sha256sum "$A/$n.fatbin" | cut -d' ' -f1)
        if [ "$cs" != "$s" ]; then
            echo "FAIL — $n: 소스가 매니페스트 빌드와 다름(리빌드: scripts/build_cuda_kernels.sh)"; stale=1
        fi
        if [ "$cf" != "$f" ]; then
            echo "FAIL — $n: fatbin 내용이 매니페스트 기록과 다름(외부 변경/복원?)"; stale=1
        fi
    done < "$A/FATBINS.manifest"
fi
if [[ "$stale" -eq 0 ]]; then echo "OK"; else fail=1; fi

if [[ "$fail" -eq 0 ]]; then echo "== 프리플라이트 PASS =="; else echo "== 프리플라이트 FAIL =="; exit 1; fi
