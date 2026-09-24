# Filesystem Corruption Incident — 2026-09-25

## Symptom

Model files (4-part, 111 GiB total) intermittently returned `ENOENT` from
`open(2)` while `stat(2)` and directory listing succeeded. All processes were
affected equally (Python, Rust, shell). System files and smaller models on the
same filesystem remained accessible. A forced `fsck` (`touch /forcefsck` +
reboot) fully repaired the damage; plain reboots did not.

## Root Cause Chain

1. **Trigger (primary)**: `llama-server.service` crash loop at boot. The unit
   mmaps a 111 GiB model and requests ~96 GiB VRAM on a 30 GiB-RAM host with
   96 GiB carve-out contention; it exited every ~8 s and systemd restarted it
   (67+ restart cycles observed). Each cycle issued massive random I/O and a
   GPU initialization that faulted. Weeks of this accumulated silent metadata
   damage. Now `systemctl disable llama-server`.
2. **Aggravator**: experimental MoE tile kernels (`LLM170_VK_Q4KSG2=1`) hit a
   GPU fault during prefill. Each observed GPU fault was followed within
   seconds by filesystem access failures. Suspected mechanism: in-flight DMA
   (page-cache writeback / journal commit) disturbed by GPU fault handling —
   no `EXT4-fs error` lines were logged (silent corruption), consistent with
   partially committed journal state that alternated on replay.
3. **Workload stress**: 111 GiB mmap streaming through a 30 GiB page cache
   during sustained pp4096 benchmarking (1.1M+ major faults) amplifies both
   effects.

## Prevention (implemented)

1. **Preflight gate** (`scripts/fs-preflight.sh`): 5–10 rapid open probes of
   the model file before any heavy run; aborts on instability. Wired into
   `scripts/gate-flash-next.sh` and the `bench` subcommand (3-probe Rust check
   with recovery instructions on failure).
2. **GPU fault sources quarantined**: the crashing sg2 kernel path stays
   env-gated (`LLM170_VK_Q4KSG2`, default off). The pre-existing
   `rows <= 8192` guard on sg1 remains.
3. **Crash-loop services disabled**: `llama-server.service` disabled;
   `deploy-web-1` container stopped.
4. **Recovery procedure** documented in `FS_RECOVERY.md`:
   `sudo touch /forcefsck && sudo reboot` (effective), or live-USB
   `fsck.ext4 -f -y` for deep repair.

## Recommendations

- Run `sudo smartctl -t long /dev/nvme0n1` once to rule out NVMe degradation.
- Re-enable `llama-server` only after fixing its VRAM allocation failure
  (it crashed before serving; the crash loop, not serving, caused damage).
- Treat any `fs-preflight` failure as a hard stop; continuing benchmarks on a
  flapping inode risks compounding the damage.
