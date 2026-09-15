#!/usr/bin/env python3
"""verify_serve용 프롬프트 스토어 생성 — 우리 서버 /tokenize 만 사용 (llama 불필요).

사용: python3 scripts/build_prompt_store.py <model.gguf> <out.json> [port]
"""
import json, os, subprocess, sys, time, urllib.request, signal

MODEL = sys.argv[1]
OUT = sys.argv[2]
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 18099

PARAS = {
    "long1": ("The quick brown fox jumps over the lazy dog. Pack my box with "
              "five dozen liquor jugs. How vexingly quick daft zebras jump! "
              "Sphinx of black quartz, judge my vow. ", 2300, 120),
    "long2": ("서울은 대한민국의 수도이며 한강이 도시를 가로지른다. 부산은 "
              "대한민국 제2의 도시로 항구 도시로 발전했다. 대전은 과학 도시로서 "
              "대덕연구단지를 품고 있다. 광주는 예술의 도시로 알려져 있다. ", 1900, 160),
    "long3": ("The Amazon rainforest spans nine countries in South America and "
              "produces roughly twenty percent of the world's oxygen supply. "
              "Its river carries more water than any other river on Earth. ", 2100, 160),
    "long4": (" Quantum computers manipulate qubits using superposition and "
              "entanglement, promising speedups for factoring and search. "
              "Error correction remains the central engineering challenge. ", 1700, 160),
}

def main():
    env = dict(os.environ)
    proc = subprocess.Popen(
        ["target/release/llm170", "serve", "--model", MODEL,
         "--port", str(PORT), "--backend", "gpu"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    base = f"http://127.0.0.1:{PORT}"
    try:
        for _ in range(300):
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError("serve 조기 종료")
                time.sleep(2)
        def tok(text):
            req = urllib.request.Request(
                f"{base}/tokenize",
                data=json.dumps({"content": text}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())["tokens"]
        prompts = {}
        for key, (para, target, cap) in PARAS.items():
            ids = None
            for k in range(2, cap):
                t = tok(para * k)
                if len(t) >= target:
                    ids = t
                    break
            assert ids, f"{key} 조립 실패"
            prompts[key] = ids
            print(f"{key}: {len(ids)} tok", flush=True)
        with open(OUT, "w") as f:
            json.dump({"prompts": prompts, "base": {}}, f)
        print(f"완료 → {OUT}")
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    main()
