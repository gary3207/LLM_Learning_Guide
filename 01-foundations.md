# 01. 실습의 바닥 다지기 — Linux·Python·PyTorch와 GPU

[← 이전: 전체 로드맵](00-roadmap.md) · [목차](README.md) · [다음: 첫 텍스트 분류기 →](02-first-classifier.md)

이 장에서는 Linux 터미널의 기본 명령을 익히고, 실습용 Python 환경을 분리한 뒤 PyTorch로 GPU 계산을 확인합니다. 모델 학습에 앞서 **명령을 실행하는 컴퓨터, 선택된 Python, 계산 장치**를 구분하는 것이 목표입니다.

> **환경 예시:** OCI A10 VM, Ubuntu 24.04 계열, Python 3.12 환경을 기준으로 설명합니다. 아래 버전과 출력은 실행 예시이며 최신 권장 버전 목록이 아닙니다. OS·패키지·드라이버·GPU에 따라 설치 방법과 출력이 달라집니다. 이미 동작하는 환경에서는 설치나 드라이버 변경을 반복하지 않습니다.

## 1. PC·VM·터미널 구분하기

| 이름 | 역할 |
|---|---|
| 로컬 PC | SSH 접속을 시작하는 컴퓨터 |
| 원격 VM | Linux 명령과 PyTorch 코드를 실행하는 컴퓨터 |
| SSH 터미널의 `$` 프롬프트 | Linux 명령 입력 위치. 프롬프트 자체는 복사하지 않음 |
| Python의 `>>>` 프롬프트 | Python 코드 입력 위치. Bash 명령을 넣는 자리가 아님 |
| `(.venv)` | 현재 터미널에서 프로젝트 가상환경이 활성화됐다는 표시 |
| `~/llm-lab/source` | VM 안에 만들 실습 폴더. 저장소의 `source/`와 별도 위치 |

`bash` 블록은 **SSH로 접속한 Linux VM의 Bash 터미널**에서 실행합니다. Windows PowerShell에 그대로 붙여 넣는 명령이 아닙니다. VM 생성과 SSH 키 설정은 이 장의 범위 밖이며, 접속 가능한 환경을 준비한 뒤 진행합니다.

## 2. 운영체제·GPU·드라이버 확인하기

### 2.1 설치보다 조회가 먼저

Ubuntu VM에서 다음 명령을 실행합니다.

```bash
cat /etc/os-release
nvidia-smi
lspci -nnk -d 10de:
dpkg -l | grep -i nvidia
```

| 명령 | 확인할 내용 |
|---|---|
| `cat /etc/os-release` | 운영체제 이름과 버전 |
| `nvidia-smi` | NVIDIA GPU·드라이버 상태 |
| `lspci -nnk -d 10de:` | NVIDIA PCI 장치와 커널 관련 정보 |
| `dpkg -l` | 설치된 패키지 목록 |
| `\| grep -i nvidia` | 앞 출력에서 NVIDIA 관련 줄을 대소문자 구분 없이 검색 |

드라이버 관리 도구가 준비되지 않은 환경의 **실행 예시**입니다.

```text
PRETTY_NAME="Ubuntu 24.04.5 LTS"
Command 'nvidia-smi' not found, but can be installed with:
00:04.0 3D controller [0302]: NVIDIA Corporation GA102GL [A10] [10de:2236] (rev a1)
        Kernel modules: nvidiafb, nouveau
```

**`nvidia-smi`가 없다고 GPU가 없는 것은 아닙니다.** PCI 조회에 A10이 보인다면 장치는 인식되고 있지만 관리 도구나 드라이버 구성을 따로 점검해야 합니다. 패키지 검색에 일치하는 줄이 없다면 설치 여부부터 확인합니다. 장치 존재, 드라이버 동작, Python에서의 GPU 사용은 각각 다른 검사입니다.

### 2.2 드라이버가 필요한 경우

> 아래 명령은 시스템을 변경하고 재부팅합니다. NVIDIA 드라이버가 필요한 Ubuntu 환경이며 해당 VM 이미지의 설치 정책과 호환성을 확인한 경우에만 사용합니다. 이미 `nvidia-smi`가 정상 동작하면 이 단계를 진행하지 않습니다. 작업 중인 프로세스를 정리하고 재접속 방법을 확보합니다.

Ubuntu의 드라이버 선택 도구를 사용하는 설치 경로의 예입니다. 한 줄씩 성공 여부를 확인합니다.

```bash
sudo apt update
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers install --gpgpu
sudo reboot
```

- `sudo`: 관리자 권한으로 실행합니다.
- `apt update`: 설치 가능한 패키지 목록을 갱신합니다. 전체 패키지 업그레이드 명령은 아닙니다.
- `ubuntu-drivers-common`: 드라이버 선택·설치 도구를 준비합니다.
- `--gpgpu`: GPU 연산용 드라이버 설치 경로를 선택합니다.
- `reboot`: VM을 재부팅합니다. SSH 연결이 끊기므로 부팅 후 다시 접속합니다.

재접속 뒤에도 `nvidia-smi`를 찾지 못한다면 드라이버를 무조건 다시 설치하지 말고 패키지를 조회합니다.

```bash
dpkg -l | grep -E 'nvidia|linux-modules-nvidia'
```

설치된 드라이버 계열을 보여주는 **실행 예시**입니다.

```text
ii  libnvidia-compute-595-server:amd64  595.91.07-0ubuntu0.24.04.1  amd64  NVIDIA libcompute package
```

**595-server 계열 드라이버와 커널 모듈은 설치되어 있고 같은 계열의 관리 도구만 빠진 경우**에는 다음처럼 도구를 추가할 수 있습니다. 다른 계열을 사용한다면 아래 패키지명을 그대로 적용하지 않습니다. 이 패키지 하나만으로 전체 드라이버가 설치되는 것도 아닙니다.

```bash
sudo apt install nvidia-utils-595-server
nvidia-smi
```

정상 동작 시 **실행 예시**입니다.

```text
NVIDIA-SMI 595.91.07              Driver Version: 595.91.07      CUDA Version: 13.2
0  NVIDIA A10
0MiB / 23028MiB
No running processes found
```

`CUDA Version: 13.2`는 드라이버가 지원하는 CUDA 버전 표시입니다. **CUDA Toolkit이나 PyTorch 설치를 확인한 결과가 아닙니다.** GPU 이름·메모리·프로세스 표시는 환경과 사용 상태에 따라 달라집니다.

## 3. Linux에서 위치와 파일 다루기

### 3.1 계정·현재 위치·목록

```bash
whoami
pwd
ls -la
```

`whoami`는 접속 계정 이름, `pwd`는 현재 작업 폴더, `ls -la`는 숨김 항목을 포함한 상세 목록을 보여줍니다.

- `~`: 현재 계정의 홈 폴더입니다.
- `.`와 `..`: 현재 폴더와 상위 폴더입니다.
- `-l`: 상세 정보, `-a`: 숨김 항목까지 표시하는 옵션입니다.
- 이름 앞의 `.`: 숨김 항목입니다. `.ssh`, `.bashrc`, `.profile` 등이 해당합니다.
- 상세 목록 첫 글자의 `d`: 디렉터리, `-`: 일반 파일입니다.

SSH 개인키·접속 설정·명령 기록에는 민감한 정보가 있을 수 있습니다. 내용을 학습 자료나 공개 저장소에 복사하지 않습니다.

### 3.2 폴더 생성과 이동

```bash
mkdir -p ~/llm-lab/source
cd ~/llm-lab/source
pwd
touch linux-practice.txt
ls -l
cat linux-practice.txt
```

| 명령 | 의미 |
|---|---|
| `mkdir -p …` | 중간 폴더까지 생성. 이미 있으면 그대로 사용 |
| `cd …` | 현재 작업 위치 변경 |
| `pwd` | 현재 위치 출력 |
| `touch …` | 파일이 없으면 빈 파일 생성. 기존 파일의 내용은 지우지 않음 |
| `ls -l` | 파일 크기 등 상세 정보 확인 |
| `cat …` | 파일 내용 읽기 |

`mkdir`는 폴더를 만들 뿐 현재 위치를 바꾸지 않습니다. 생성 뒤에도 `pwd`가 홈 폴더를 가리킨다면 `cd`를 실행했는지 확인합니다. 새 빈 파일은 크기가 0바이트이며, `cat`이 표시할 내용이 없는 것이 정상입니다. 기존 파일이 있다면 결과는 그 파일 내용에 따라 달라집니다.

### 3.3 내용 쓰기와 추가하기

다음 명령은 연습 파일에 내용을 기록합니다. **첫 줄은 기존 내용을 덮어씁니다.** 대상 파일을 확인한 뒤 실행합니다.

```bash
echo "Hello Linux" > linux-practice.txt
cat linux-practice.txt
echo "I am learning Linux for LLM practice." >> linux-practice.txt
cat linux-practice.txt
```

`echo`는 문자열을 출력합니다. `>`는 출력을 파일에 덮어쓰고, `>>`는 파일 끝에 추가합니다. 마지막 `cat`의 **실행 예시**입니다.

```text
Hello Linux
I am learning Linux for LLM practice.
```

## 4. Python과 Git 명령 확인하기

```bash
python3 --version
git --version
```

**실행 예시:**

```text
Python 3.12.3
git version 2.43.0
```

버전 출력은 실행 파일을 사용할 수 있다는 뜻입니다. 프로젝트 패키지가 설치되어 있다는 뜻이나 Git 저장소가 초기화됐다는 뜻은 아닙니다. 이 장에서는 Git 설치 여부만 확인하며, 저장소 생성·커밋·원격 게시 방법은 다루지 않습니다.

### 변수와 출력 연습

Bash에서 `python3`를 실행합니다. `>>>`가 나오면 다음 Python 코드를 한 줄씩 입력합니다. `>>>` 자체는 입력하지 않습니다.

```python
message = "Hello LLM"
print(message)
epochs = 3
print(epochs)
print(epochs + 1)
exit()
```

- `message`, `epochs`: 값을 가리키는 변수 이름입니다.
- `=`: 오른쪽 값을 왼쪽 변수에 대입합니다.
- 따옴표 안의 글: 문자열입니다.
- `print`: 값을 화면에 출력합니다.
- `epochs`: 이후 전체 데이터를 몇 번 학습할지 나타내는 이름으로 사용합니다.
- `exit()`: Python 대화형 환경을 종료하고 Bash로 돌아갑니다.

출력에서 `Hello LLM`, `3`, `4`를 확인합니다. Bash의 폴더 이동 명령과 Python 코드는 서로 다른 입력 환경에서 실행합니다.

## 5. 실습 전용 Python 가상환경 만들기

### 5.1 어떤 Python에 PyTorch가 있는가?

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('GPU 사용 가능:', torch.cuda.is_available())"
```

`-c`는 뒤 문자열을 Python 코드로 실행합니다. 해당 Python에 PyTorch가 없다면 다음 오류가 발생할 수 있습니다.

```text
ModuleNotFoundError: No module named 'torch'
```

이는 **이 명령이 선택한 Python 환경에서** `torch`를 찾지 못했다는 뜻입니다. GPU 고장이나 모든 Python 환경의 상태를 뜻하지 않습니다. 시스템 Python에 패키지를 섞어 설치하기보다 프로젝트 전용 환경을 사용합니다.

### 5.2 생성과 활성화

```bash
cd ~/llm-lab/source
python3 -m venv .venv
source .venv/bin/activate
```

- `python3 -m venv .venv`: Python의 `venv` 기능으로 `.venv` 폴더에 환경을 생성합니다.
- `source .venv/bin/activate`: 현재 Bash 터미널에서 사용할 Python과 관련 경로를 바꿉니다.
- 가상환경은 패키지를 분리합니다. VM이나 GPU 드라이버를 새로 만드는 기능은 아닙니다.

다음 오류가 발생하면 활성화보다 **생성 실패의 원인**을 먼저 해결합니다.

```text
The virtual environment was not created successfully because ensurepip is not
available.
-bash: .venv/bin/activate: No such file or directory
```

Ubuntu의 Python 3.12에서 필요한 `venv` 패키지가 없는 경우의 복구 예입니다. 다른 Python 버전이라면 그 버전에 맞는 패키지를 확인합니다.

```bash
sudo apt update && sudo apt install python3.12-venv
python3 -m venv .venv && source .venv/bin/activate
python -m pip --version
```

`&&`는 앞 명령이 성공했을 때만 다음 명령을 실행합니다. 생성 실패 뒤에 활성화 스크립트를 임의로 만들거나, 원인 확인 없이 폴더를 삭제하지 않습니다.

`python -m pip`는 **현재 선택된 Python에 속한 pip**를 실행합니다. 다음은 **실행 예시**입니다. `<VM>`은 환경마다 달라지는 계정 경로 부분을 나타냅니다. 경로가 프로젝트의 `.venv` 안인지 확인합니다.

```text
pip 24.0 from /home/<VM>/llm-lab/source/.venv/lib/python3.12/site-packages/pip (python 3.12)
```

## 6. PyTorch로 GPU 계산 확인하기

### 6.1 패키지 설치

실행 위치는 `~/llm-lab/source`이며 `.venv`가 활성화되어 있어야 합니다. 다음은 **CUDA 12.8용 PyTorch 패키지 저장소를 사용하는 설치 예시**입니다. GPU·드라이버·Python과 호환되는 배포본인지 확인한 환경에 적용합니다.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

버전 번호를 고정하지 않은 명령이므로 실행 시점에 따라 설치 버전이 달라질 수 있습니다. 아래 출력의 버전을 최신 권장 버전이나 항상 설치되는 버전으로 해석하지 않습니다.

### 6.2 GPU 검사 전체 코드

[소스 파일: gpu_check.py](source/01-foundations/gpu_check.py)

다음 블록 전체를 Bash에 입력합니다.

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("GPU 사용 가능:", torch.cuda.is_available())
assert torch.cuda.is_available(), "PyTorch에서 GPU를 사용할 수 없습니다."

print("GPU:", torch.cuda.get_device_name(0))
x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
print("GPU 계산 결과:", (x * 2).cpu().tolist())
PY
```

| 코드 | 역할 |
|---|---|
| `import torch` | PyTorch 기능 불러오기. 학습된 AI를 가져오는 것은 아님 |
| `torch.__version__` | 사용 중인 패키지 버전 출력 |
| `torch.cuda.is_available()` | 현재 Python에서 CUDA 사용이 가능한지 확인 |
| `assert …` | GPU 사용 조건이 맞지 않으면 오류 발생 |
| `get_device_name(0)` | 첫 GPU 이름 조회 |
| `torch.tensor(…, device="cuda")` | 숫자 묶음인 텐서를 GPU에 생성 |
| `x * 2` | 텐서의 각 값에 2를 곱함 |
| `.cpu().tolist()` | CPU로 옮긴 뒤 Python 리스트로 변환 |

`python - <<'PY'`와 마지막 `PY`는 Bash가 여러 줄을 Python에 전달하는 here-document 문법입니다. `.py` 파일에는 이 두 줄을 넣지 않습니다. 링크한 소스 파일은 Python 본문을 제공합니다.

**실행 예시:**

```text
UserWarning: Failed to initialize NumPy: No module named 'numpy'
PyTorch: 2.11.0+cu128
GPU 사용 가능: True
GPU: NVIDIA A10
GPU 계산 결과: [2.0, 4.0, 6.0]
```

NumPy가 없는 환경에서는 위 경고가 나타날 수 있습니다. 경고와 별개로 마지막 계산 결과까지 나오는지 확인합니다. 드라이버 표시의 CUDA 13.2와 PyTorch 배포본의 `cu128`은 같은 의미의 버전 표기가 아닙니다. 숫자의 일치 여부만 판단하지 말고 호환 조건과 실제 연산을 확인합니다.

### 6.3 NumPy 설치와 배열 변환 검사

NumPy가 없다면 같은 가상환경에 설치합니다.

```bash
python -m pip install numpy
```

[소스 파일: numpy_check.py](source/01-foundations/numpy_check.py)

```bash
python - <<'PY'
import numpy as np
import torch

x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
result = (x * 2).cpu().numpy()
np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])
print("GPU 연산·NumPy 변환 성공:", result)
PY
```

- `import numpy as np`: NumPy를 `np`라는 이름으로 사용합니다.
- `.cpu().numpy()`: GPU 결과를 CPU로 옮긴 뒤 NumPy 배열로 바꿉니다.
- `np.testing.assert_array_equal`: 배열이 기대한 값과 같은지 검사합니다. 다르면 성공 문구에 도달하지 않습니다.

**실행 예시:**

```text
GPU 연산·NumPy 변환 성공: [2. 4. 6.]
```

**이 계산에는 모델도, 가중치 수정도 없습니다.** 숫자 계산이 되는지 확인한 것이며, 모델 학습 성공을 뜻하지 않습니다.

### 6.4 문제가 생기면 확인할 순서

1. `ModuleNotFoundError`: 가상환경 활성화와 `python -m pip --version`의 경로를 확인합니다.
2. `nvidia-smi` 실패: Python 설치보다 GPU 노출·드라이버·관리 도구 상태부터 확인합니다.
3. `nvidia-smi`는 정상인데 CUDA 사용이 `False`: 현재 Python의 PyTorch 배포본과 드라이버 호환성, VM의 GPU 접근 조건을 확인합니다.
4. `ensurepip` 오류 뒤 활성화 실패: 활성화 파일을 만드는 대신 `venv` 생성에 필요한 패키지를 확인합니다.
5. NumPy 경고: 같은 Python에 NumPy가 있는지 확인하고 변환 검사 결과를 봅니다.

오류 메시지의 첫 원인을 읽고 해당 계층부터 점검합니다. 설치 명령을 전부 반복하는 것은 진단을 대신하지 못합니다.

## 7. PyTorch·모델·CUDA의 역할

| 개념 | 설명 | 예 |
|---|---|---|
| Python | 코드를 쓰는 언어 | 변수·함수·반복문 |
| PyTorch | 숫자 계산·모델·미분·최적화 도구 | `torch.tensor`, `nn.Linear` |
| 텐서 | 숫자를 담고 계산하는 자료구조 | `[1.0, 2.0, 3.0]`을 담은 `x` |
| 모델 | 입력에서 예측 점수를 만드는 계산 구조 | 다음 장의 선형 분류기 |
| 가중치·편향 | 학습으로 바뀌는 조절값 | `Linear`의 파라미터 |
| 자동 미분 | 손실에 대한 파라미터의 기울기 계산 | `loss.backward()` |
| 옵티마이저 | 기울기를 바탕으로 파라미터 수정 | `optimizer.step()` |
| GPU | 계산 장치 | NVIDIA A10 |
| CUDA | NVIDIA GPU 계산에 쓰는 기술 | `device="cuda"` |

학습 흐름은 **예측 → 정답과 비교 → 손실 계산 → 기울기 계산 → 가중치 수정 → 반복**입니다. `backward()`는 기울기를 계산하고, `step()`이 실제 값을 수정합니다. PyTorch를 설치했다고 모든 Python 코드가 GPU에서 실행되지는 않습니다. 텐서와 모델을 사용할 장치에 배치해야 합니다.

작은 분류기로 원리를 배우는 데는 CPU도 사용할 수 있습니다. 지원되는 AMD GPU의 ROCm, Mac GPU의 MPS 같은 경로도 있지만 하드웨어·운영체제·패키지별 지원 조합은 따로 확인해야 합니다. 이 가이드는 NVIDIA CUDA 경로를 사용합니다. 다음 장의 GPU 필수 검사는 **이 가이드가 선택한 실행 조건**이지 모든 모델 학습의 필수조건은 아닙니다.

## 8. 새 터미널에서 이어서 사용하기

기존 가상환경은 다시 만들지 않고 활성화합니다.

```bash
cd ~/llm-lab/source
source .venv/bin/activate
```

이 명령은 현재 터미널의 Python 선택을 바꿉니다. 새 접속이라는 이유만으로 패키지 설치·드라이버 설치·가상환경 생성을 처음부터 반복하지 않습니다.

## 9. 완료 체크리스트와 복습 질문

- [ ] 로컬 PC와 SSH로 접속한 VM을 구분한다.
- [ ] `mkdir`, `cd`, `pwd`를 생성·이동·위치 확인으로 구분한다.
- [ ] 빈 파일의 내용이 표시되지 않는 이유를 설명한다.
- [ ] `>`의 덮어쓰기 위험과 `>>`의 추가 동작을 구분한다.
- [ ] pip 경로가 프로젝트의 `.venv` 안인지 확인한다.
- [ ] GPU 검사와 GPU→NumPy 변환 검사를 통과한다.
- [ ] 드라이버 동작, PyTorch 설치, GPU 계산, 모델 학습을 서로 구분한다.

**스스로 답해 보기**

1. `mkdir`를 실행했는데 `pwd`가 바뀌지 않는 이유는 무엇인가?
2. `python -m pip --version`의 경로에서 무엇을 확인해야 하는가?
3. 가상환경 생성 실패 뒤에 활성화 오류가 따라오는 이유는 무엇인가?
4. `CUDA Version: 13.2`가 PyTorch 설치 증거가 아닌 이유는 무엇인가?
5. `nvidia-smi`가 정상인데 GPU 검사가 실패하면 무엇부터 확인해야 하는가?
6. PyTorch·모델·GPU는 각각 어떤 역할인가?
7. GPU 곱셈 검사에는 왜 학습이 없다고 말할 수 있는가?

[← 이전: 전체 로드맵](00-roadmap.md) · [목차](README.md) · [다음: 첫 텍스트 분류기 →](02-first-classifier.md)
