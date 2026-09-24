# 파일시스템 복구 절차 (2026-09-25)

## 증상
- FN 모델 파일(111GB, 4개 파트)의 inode 손상 — stat은 성공하나 open()이 ENOENT
- 원본(/home/yoon/models/qwen3.8-Next/)과 fresh 복사본 모두 동일 손상
- llama-server.service 크래시 루프가 대량 I/O로 손상 유발(현재 disabled)

## 복구 절차
1. 재부팅 → 라이브 USB 부팅 (Ubuntu 설치 USB의 "Try Ubuntu")
2. 터미널에서:
   ```bash
   # LVM 볼륨 활성화
   sudo vgchange -ay
   
   # ext4 파일시스템 검사 (마운트 해제 상태에서)
   sudo fsck.ext4 -f /dev/mapper/ubuntu--vg-ubuntu--lv
   
   # 손상 inode 복구 시도
   sudo fsck.ext4 -f -y /dev/mapper/ubuntu--vg-ubuntu--lv
   ```
3. fsck 후 재부팅
4. 모델 파일 재확인: `ls -la /home/yoon/models/qwen3.8-Flash-Next/`
5. 여전히 손상 시 모델 재다운로드 필요 (~111GB)

## 재발 방지
- llama-server.service는 disabled 상태 유지 (크래시 루프가 I/O 폭탄)
- 모델 mmap 서빙 시 RAM 30GB + 111GB 모델 = 극심한 페이지 캐시 스래싱
- 권장: 모델 서빙 전 mmap 프리페치로 순차 판독 (vmtouch -t)

## 2026-09-25 01:50 추가 분석

### GPU fault ↔ FS 상관관계
- sg2 커널 크래시 직후 FS 즉시 차단 (10분+)
- GPU fault → IOMMU 상태 오염 → NVMe DMA 실패 추정
- sg2 테스트 자체가 FS를 악화시키는 악순환

### 재다운로드 후에도 재발
- 재다운로드 완료 1분 후 즉시 손상 (새 inode 포함)
- llama-server 정지·deploy-web-1 정지 후에도 지속
- 파일시스템 자체(fsck 필요) 또는 NVMe 컨트롤러 문제

### 최종 권고
1. 라이브 USB로 부팅
2. `sudo fsck.ext4 -f -y /dev/mapper/ubuntu--vg-ubuntu--lv`
3. 재부팅 후 `sudo smartctl -t long /dev/nvme0n1` (백그라운드 장기 테스트)
4. fsck 후에도 재발 시 NVMe 교체 검토
