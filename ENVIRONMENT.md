# 项目环境说明

## 设备概览

```mermaid
graph TD
    subgraph 硬件层["🖥️ 硬件层"]
        OS["Windows 11 Home China<br/>10.0.26200"]
        GPU["NVIDIA GeForce RTX 4060<br/>Laptop GPU<br/>8GB GDDR6"]
        CPU["x64 AMD64"]
    end

    subgraph 驱动层["🔧 驱动层"]
        Driver["NVIDIA Driver 581.80"]
        CUDA_API["CUDA API 13.0"]
    end

    subgraph Python环境["🐍 Python 环境"]
        direction LR
        subgraph 系统环境["系统 Python"]
            SysPy["Python 3.14<br/>(已卸载 torch)"]
        end
        subgraph GPU环境["torch_env (conda)"]
            CondaPy["Python 3.10.20"]
            Torch["PyTorch 2.6.0+cu124<br/>✅ CUDA 可用"]
            TorchVision["TorchVision 0.27.0"]
        end
    end

    GPU --> Driver
    Driver --> CUDA_API
    CUDA_API --> Torch
    CondaPy --> Torch
    CondaPy --> TorchVision
```

## 环境调用关系

```mermaid
flowchart LR
    subgraph PackageManager["包管理器"]
        Conda["Miniconda3<br/>C:/ProgramData/miniconda3/"]
    end

    subgraph Envs["虚拟环境"]
        Base["base<br/>Python 3.12"]
        TorchEnv["torch_env ⭐<br/>Python 3.10<br/>PyTorch 2.6.0+cu124"]
    end

    subgraph 调用方式["调用方式"]
        Shell1["Git Bash<br/>需手动激活 conda"]
        Shell2["终端直接激活<br/>conda activate torch_env"]
    end

    Conda --> Base
    Conda --> TorchEnv
    Shell1 --> TorchEnv
    Shell2 --> TorchEnv
```

## CUDA 兼容性

```mermaid
graph TD
    subgraph Layer1["驱动支持上限"]
        L1["CUDA 13.0<br/>（NVIDIA 驱动 581.80 支持）"]
    end

    subgraph Layer2["PyTorch 编译版本"]
        L2["CUDA 12.4<br/>（torch 2.6.0+cu124）"]
    end

    subgraph Layer3["兼容性"]
        L3["✅ 向下兼容<br/>CUDA 12.4 程序可在<br/>CUDA 13.0 驱动上运行"]
    end

    L1 --> L2
    L2 --> L3
    L3 -->|说明| Note["驱动支持的 CUDA 版本 ≥<br/>PyTorch 编译的 CUDA 版本<br/>因此可以正常运行"]
```

## 环境变量与路径

```mermaid
flowchart TB
    subgraph Paths["关键路径"]
        CondaRoot["Miniconda 安装位置<br/>📁 C:/ProgramData/miniconda3/"]
        CondaExe["Conda 可执行文件<br/>📄 Scripts/conda.exe"]
        TorchEnvPython["torch_env Python<br/>📄 envs/torch_env/python.exe"]
    end

    subgraph Activation["激活 torch_env"]
        Step1["1️⃣ 打开终端"]
        Step2["2️⃣ conda activate torch_env"]
        Step3["3️⃣ 验证: python -c 'import torch; print(torch.cuda.is_available())'"]
    end

    CondaRoot -->|Scripts/| CondaExe
    CondaRoot -->|envs/| TorchEnvPython
    Step2 --> Step3
```

## 验证命令

| 检查项 | 命令 | 预期结果 |
|--------|------|----------|
| CUDA 驱动 | `nvidia-smi` | Driver 581.80, CUDA 13.0 |
| PyTorch 版本 | `python -c "import torch; print(torch.__version__)"` | `2.6.0+cu124` |
| CUDA 可用性 | `python -c "import torch; print(torch.cuda.is_available())"` | `True` |
| 显卡识别 | `python -c "import torch; print(torch.cuda.get_device_name(0))"` | `NVIDIA GeForce RTX 4060 Laptop GPU` |

> ⚠️ **注意**：运行 PyTorch 程序前，请确保已激活 `torch_env` 虚拟环境。
