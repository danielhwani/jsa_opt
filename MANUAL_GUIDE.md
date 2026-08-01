# 수동 실행 가이드 (Claude Code 없이 터미널에서 직접 실행하기)

이 문서는 2026-07-26~27 세션에서 Claude Code로 진행했던 작업들
(JSA+Conv 학습 → TensorRT 벤치마크 → 이미지/지표 비교)을 사용자가 직접
터미널에서 재현할 수 있도록 정리한 가이드입니다.

## 0. 사전 준비

```bash
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t joint_sa .
```

**호스트 사양 주의사항** (이 프로젝트가 경량화 이식된 기준 환경):
- GPU: GTX 1650 (4GB VRAM)
- RAM: 7.7GB, **스왑 0B**

스왑이 없는 상태에서 컨테이너가 호스트 RAM을 무제한으로 쓰면, 컨테이너가 아니라
호스트 시스템 전체가 멈출 수 있습니다 (실제로 이번 세션 이전에 한 번 발생했던
프리즈의 원인 후보). 아래 "안전 모드"처럼 `--memory`/`--memory-swap`을 항상
지정하는 것을 권장합니다.

## 1. 간단 버전: 인터랙티브 컨테이너

```bash
bash run_docker.sh
```

컨테이너 안에서:

```bash
bash scripts/generate_dataset.sh                       # 데이터셋 없을 때만
bash scripts/train.sh                                  # 원본 JSA 학습
bash scripts/train_conv.sh                              # JSA+Conv 학습
bash scripts/run_benchmark_original_jsa_pth_vs_trt.sh   # TensorRT 벤치마크 (JSA + Conv 둘 다)
bash scripts/compare_conv.sh                            # 이미지+지표 비교 (아래 4절 참고)
```

`run_docker.sh`는 메모리 상한이 걸려 있지 않으므로, 4GB VRAM/무스왑 환경에서는
학습이나 TensorRT 변환 중 OOM이 나면 시스템 전체가 영향을 받을 위험이 있습니다.
가능하면 2절의 "안전 모드"를 사용하세요.

## 2. 안전 모드 (권장, 이번 세션에서 검증된 방식)

컨테이너를 백그라운드(`-d`)로 띄우고, 메모리 상한 + `PYTHONUNBUFFERED=1`을 걸어
로그가 실시간으로 나오게 하는 방식입니다. 공통 패턴:

```bash
docker run -d --name <NAME> \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  -v "$(pwd):/workspace" \
  --shm-size=2g \
  --memory=6g --memory-swap=6g \
  joint_sa \
  bash -lc "<COMMAND>"

docker logs -f <NAME>          # 실시간 로그 확인
docker wait <NAME>              # 종료 코드 대기
docker rm -f <NAME>             # 정리
```

> `PYTHONUNBUFFERED=1`이 없으면 TTY 없는 `docker run`에서 Python stdout이
> 완전 버퍼링되어, 실제로는 정상 진행 중인데 로그만 멈춘 것처럼 보일 수 있습니다
> (이번 세션에서 실제로 이 때문에 정상 프로세스를 "행(hang)"으로 오판해
> 강제종료한 적이 있습니다 — 아래 3절 참고).

### 2-1. JSA+Conv 학습

```bash
COMMAND="cd /workspace/codes && python conv_train.py"
```

완료되면 체크포인트가 여기 생성됩니다:
`data/jsaCNN_classroom_overfit/__checkpoints__/epoch_jsaCNN_classroom_overfit_best.pth`

이 프로젝트의 4GB 카드 기준으로는 학습 세트가 매우 작아 (train npz 4장) 100 epoch가
1분 내외로 끝나고, GPU 메모리도 1.5GB를 넘지 않습니다.

### 2-2. TensorRT 벤치마크 — JSA(원본)

```bash
COMMAND="cd /workspace && python codes/benchmark_original_jsa_trt.py \
  --repo-root /workspace --model-kind jsa --config-module config \
  --height 128 --width 128 --workspace-gb 1 --warmup 20 --iters 100 \
  --out-dir /workspace/benchmark_results/engine"
```

### 2-3. TensorRT 벤치마크 — JSA+Conv

```bash
COMMAND="cd /workspace && python codes/benchmark_original_jsa_trt.py \
  --repo-root /workspace --model-kind conv --config-module config_cnn \
  --height 128 --width 128 --workspace-gb 1 --warmup 20 --iters 100 \
  --out-dir /workspace/benchmark_results/engine"
```

두 벤치마크 모두 체크포인트를 명시적으로 지정하지 않으면 각 config의
`load_epoch`(둘 다 현재 `"best"`)을 기준으로 자동으로
`epoch_<task>_best.pth`를 찾습니다. 결과는
`benchmark_results/engine/*.json`에 PyTorch 원본/export-patch/torch2trt 평균
시간과 정확도(max abs diff)가 저장됩니다.

`scripts/run_benchmark_original_jsa_pth_vs_trt.sh`는 위 두 단계를 순서대로
실행하는 래퍼입니다 (`HEIGHT`/`WIDTH`/`WORKSPACE_GB` 환경변수로 조절 가능). 다만
이 스크립트는 JSA를 다시 컴파일한 뒤 Conv를 이어서 돌리므로, JSA 엔진이 이미
있고 Conv만 다시 돌리고 싶다면 위 2-3 명령만 단독으로 실행하는 편이 빠릅니다.

## 3. 로그가 조용할 때 hang인지 확인하는 법

TensorRT의 tactic profiling 단계는 실제로 진행 중이어도 수십~수백 초간 로그
출력이 없을 수 있습니다. 로그 무출력만으로 강제종료하지 말고, 아래로 GPU/CPU
활동을 같이 확인하세요:

```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
docker stats --no-stream <NAME>
```

GPU 사용률이나 컨테이너 CPU%가 0에 가깝지 않다면 정상 진행 중입니다.

## 4. 이미지 비교 (GT / Noisy / JSA / JSA+Conv)

**별도 스크립트를 새로 만들 필요는 없습니다.** 이번 세션에서 이미
`codes/compare_jsa_vs_conv.py`를 작성해뒀고, 그 얇은 실행 래퍼로
`scripts/compare_conv.sh`도 추가했습니다.

```bash
COMMAND="cd /workspace/codes && python compare_jsa_vs_conv.py --view-index 0"
```

또는 인터랙티브 컨테이너 안에서:

```bash
bash scripts/compare_conv.sh --view-index 0
```

옵션:
- `--view-index N`: 테스트 세트에서 몇 번째 뷰를 비교할지 (자연 정렬 순서, 기본값 0
  → `classroom_overfit_jsa_512_0000`). 뷰가 더 있다면 1, 2 ... 로 바꿔서 실행하면 됩니다.
- `--out-dir DIR`: 결과 저장 위치 (기본값 `outputs/inference_classroom/`)

실행하면:
- `outputs/inference_classroom/<view>_compare.png` — GT/Noisy/JSA/JSA+Conv 4분할
  라벨링 이미지 (PSNR/SSIM/추론시간 캡션 포함)
- `outputs/inference_classroom/<view>_{gt,noisy,jsa,conv}.png` — 개별 PNG
- 콘솔에 PSNR/SSIM/추론시간 markdown 표 출력

내부적으로 `eval.py`의 `tiled_forward` (128px 타일, 24px 코사인 블렌딩 오버랩)를
그대로 재사용하므로, 학습/평가 때와 동일한 방식으로 512×512 이미지를 추론합니다.

## 5. 결과 파일 위치 정리

| 산출물 | 경로 |
|---|---|
| JSA 체크포인트 | `data/jsa_classroom_overfit/__checkpoints__/` |
| JSA+Conv 체크포인트 | `data/jsaCNN_classroom_overfit/__checkpoints__/` |
| JSA TensorRT 엔진/결과 | `benchmark_results/engine/jsa_torch2trt_exportpatch_fp16_1x3x128x128.*` |
| JSA+Conv TensorRT 엔진/결과 | `benchmark_results/engine/jsa_conv_torch2trt_exportpatch_fp16_1x3x128x128.*` |
| 이미지/지표 비교 결과 | `outputs/inference_classroom/` |
| staircase2 데이터셋 (input/target) | `data/__train_scenes__/staircase2_overfit_jsa/`, `data/__test_scenes__/staircase2_overfit_jsa/` |

## 6. 새 씬으로 데이터셋 생성하기 (예: staircase2)

`classroom` 말고 다른 씬으로 데이터셋을 만들고 싶을 때의 절차입니다.
`scenes/staircase2/`가 그 예시로 이미 포함되어 있습니다
([Benedikt Bitterli's Rendering Resources](https://benedikt-bitterli.me/resources/)의
Mitsuba 3 호환 씬, CC-BY 라이선스).

### 6-1. 씬 준비

자기완결적인 Mitsuba 3 XML 씬(`version="3.0.0"`, 자체 `models/`, `textures/` 포함)을
`scenes/<name>/<name>.xml` 형태로 저장소에 배치합니다.

### 6-2. 카메라 파라미터 추출

씬 XML의 `<sensor><transform name="to_world"><matrix value="..."/>`에서 4x4 행렬을 뽑아
origin(마지막 열)/up(2번째 열)/forward(3번째 열)/fov를 계산합니다:

```python
import numpy as np
vals = [float(v) for v in "<matrix value 16개 숫자>".split()]
M = np.array(vals).reshape(4, 4)
origin = M[:3, 3]
up = M[:3, 1]
target = origin + M[:3, 2]   # origin + forward
```

> **주의**: `up`/`forward` 성분이 `-4.21474e-08`처럼 과학적 표기법 음수로 나오면
> `--base-up` 같은 `nargs=3` 인자에 그대로 넣지 마세요. Python argparse의 음수
> 판별 정규식(`^-\d+$|^-\d*\.\d+$`)이 `e-08` 표기를 못 알아봐서 옵션 플래그로
> 오인해 `error: expected 3 arguments`로 실패합니다. 이런 값은 사실상 0이므로
> `0.0`으로 반올림해서 넣으면 됩니다 (staircase2도 이 문제를 겪어서 up을
> `0.0 1.0 0.0`으로 고쳤습니다).

### 6-3. 생성 스크립트 작성

`scripts/generate_dataset.sh`를 복사해서 `--scene`/`--name`/카메라 파라미터만
바꾼 `scripts/generate_dataset_<name>.sh`를 만듭니다. `scripts/generate_dataset_staircase2.sh`는
해상도/뷰수/spp를 환경변수로 오버라이드할 수 있게 되어 있어, 저해상도 스모크
테스트와 풀 해상도 본 생성에 같은 스크립트를 재사용합니다:

```bash
# GPU 초기화만 빠르게 확인하는 스모크 테스트
WIDTH=128 HEIGHT=128 NUM_VIEWS=1 INPUT_SPP=1 AOV_SPP=1 REF_SPP=64 \
  bash scripts/generate_dataset_staircase2.sh

# 풀 해상도 (기본값: 512x512, 2 views, input/aov 4spp, ref 4096spp)
bash scripts/generate_dataset_staircase2.sh
```

### 6-4. 안전 모드로 실행 (2절과 동일한 패턴)

```bash
docker run -d --name staircase2_gen \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 \
  --memory=6g --memory-swap=6g --shm-size=2g \
  -v "$(pwd):/workspace" -w /workspace \
  joint_sa \
  bash scripts/generate_dataset_staircase2.sh

docker logs -f staircase2_gen     # 실시간 로그 (input+aov, ref chunk 진행률)
docker wait staircase2_gen        # 종료 코드 대기
docker rm staircase2_gen          # 정리
```

`ref-spp=4096`는 `--ref-chunk-spp`(기본 512) 단위로 청크 렌더링되며 각 청크마다
`elapsed`/`eta`가 로그로 출력됩니다. 512×512 기준 뷰당 GT 렌더링에 약 2.5~3분이
걸리고 그 사이 로그가 19초 간격으로 나오므로, 그보다 오래 조용하면 3절의
`nvidia-smi`/`docker stats`로 실제 진행 여부를 확인하세요.

결과는 `data/__train_scenes__/<name>/`, `data/__test_scenes__/<name>/`에
`input`/`target` EXR + `input_npz`(`color`+`aux` 7채널: albedo 3 + depth 1 + normal 3)
/`target_npz`로 저장되고, 마지막에 `config["task"]`/`trainDatasetDirectory`/
`testDatasetDirectory` 값이 콘솔에 출력됩니다 — 이 값을 `config.py`/`config_cnn.py`에
반영하면 새 씬으로 바로 학습할 수 있습니다.
