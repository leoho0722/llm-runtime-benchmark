# LLM Runtime Benchmark

LLM Inference Runtime Benchmark using PyTorch, Ollama, Apple MLX, etc.

## Environment

Python 3.10.x

### PyTorch

```bash
python3.10 -m venv .venv-pytorch
source .venv-pytorch/bin/activate

pip install -U pip setuptools pip-autoremove
pip install -r pytorch/requirements-pytorch.txt
```

### Apple MLX

```bash
python3.10 -m venv .venv-mlx
source .venv-mlx/bin/activate

pip install -U pip setuptools pip-autoremove
pip install -r mlx/requirements-mlx.txt
```

### Ollama

```bash
python3.10 -m venv .venv-ollama
source .venv-ollama/bin/activate

pip install -U pip setuptools pip-autoremove
pip install -r ollama/requirements-ollama.txt
```

## Model Download

| Args          | Type  | Required     | Default                    | Description                        |
| ------------- | ----- | ------------ | -------------------------- | ---------------------------------- |
| `--repo_id`   | `str` | **Required** |                            | Path or Hugging Face Repository ID |
| `--token`     | `str` | **Optional** |                            | Hugging Face API Token             |
| `--cache_dir` | `str` | **Optional** | `~/.cache/huggingface/hub` | Cache directory for the model      |

```bash
# Activate the Virtual Environment (PyTorch or Apple MLX)
source .venv-pytorch/bin/activate
# or
source .venv-mlx/bin/activate

# Download the model from Hugging Face Hub
python hf_model_download.py --repo_id "mlx-community/gemma-2-9b-it-4bit"

# Download the model from Hugging Face Hub with custom cache directory
python hf_model_download.py --repo_id "mlx-community/gemma-2-9b-it-4bit" --cache_dir "/tmp/huggingface/hub"

# Download the model from Hugging Face Hub with custom hugging face token
python hf_model_download.py --repo_id "mlx-community/gemma-2-9b-it-4bit" --token "YOUR_HUGGING_FACE_API_TOKEN"
```

## Run

### PyTorch

| Args                  | Type   | Required     | Default | Description                              |
| --------------------- | ------ | ------------ | ------- | ---------------------------------------- |
| `-m`, `--model`       | `str`  | **Required** |         | Path or Hugging Face Model Repository ID |
| `--prompt`            | `str`  | **Required** |         | Prompt for the LLM Model                 |
| `--load_in_4bit`      | `bool` | **Optional** |         | Load the model in 4-bit                  |
| `--load_in_8bit`      | `bool` | **Optional** |         | Load the model in 8-bit                  |
| `--trust_remote_code` | `bool` | **Optional** |         | Trust remote code                        |
| `--max_tokens`        | `int`  | **Required** |         | Maximum number of new tokens to generate |

```bash
source .venv-pytorch/bin/activate

python pytorch/pytorch_benchmark.py -m "meta-llama/Llama-3.1-8B-Instruct" --prompt "What is the largest country in the world?" --load_in_4bit --trust_remote_code --max_new_tokens 512
```

### Apple MLX

| Args            | Type  | Required     | Default | Description                              |
| --------------- | ----- | ------------ | ------- | ---------------------------------------- |
| `-m`, `--model` | `str` | **Required** |         | Path or Hugging Face Model Repository ID |
| `--prompt`      | `str` | **Required** |         | Prompt for the LLM Model                 |
| `--max_tokens`  | `int` | **Required** |         | Maximum number of new tokens to generate |

```bash
source .venv-mlx/bin/activate

python mlx/mlx_benchmark.py -m "mlx-community/Llama-3.1-8B-Instruct-4bit" --prompt "What is the largest country in the world?" --max_tokens 512
```

### Ollama

| Args            | Type  | Required     | Default                  | Description                              |
| --------------- | ----- | ------------ | ------------------------ | ---------------------------------------- |
| `-m`, `--model` | `str` | **Required** |                          | Path or Hugging Face Model Repository ID |
| `--prompt`      | `str` | **Required** |                          | Prompt for the LLM Model                 |
| `--num_ctx`     | `int` | **Optional** | `2048`                   | Number of Context Length                 |
| `--num_predict` | `int` | **Optional** | `None`                   | Maximum number of new tokens to generate |
| `--ollama_host` | `str` | **Optional** | `http://localhost:11434` | Ollama Host                              |

```bash
source .venv-ollama/bin/activate

python ollama/ollama_benchmark.py -m "llama3.2:3b" --prompt "What is the largest country in the world?" --num_predict 100
```
