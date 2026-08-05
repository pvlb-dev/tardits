# Text And Rotating Dimensions In Token Space
 *or: how to make LMs bigger on the inside*

Welcome to **TARDITS**, a library exploring (tiny) Complex-Valued LM architectures. It's my take on the awesome [Let's build GPT](https://youtu.be/kCc8FmEb1nY?si=I0z2-jByJj_6_QIH) tutorial. After finishing my GPT-net and updating it (RoPE, swiGLU, RMSNorm...), I got the idea to write a complex transformer. Mainly because I asked myself if they *really* need an activation function... turns out **they don't 🤯**

After a few iterations, the $\mathbb{C}$-net got surprisingly stable. Of course, I had to try quaternions ($\mathbb{H}$) next. They were built using the Cayley-Dickson construction — so I pass 2 complex values through the transformer and let PyTorch do all the backprop work. Then the concept escalated a little, and the Octonion ($\mathbb{O}$) and Sedenion ($\mathbb{S}$) nets also somehow happened. Both work, but they take a really long time to train. Especially on CPU :D

There may be better uses for CVNNs than predicting text... but why not Shakespeare?

<br>


## 🙃 Architecture
<details>
<summary><b>Click to expand 🔻</b></summary>

<br>

<p align="center">
<img width="15%" src="https://github.com/user-attachments/assets/496ab290-72af-4087-b4fe-ff138fe22a37">
</p>
Figure: Complete dataflow inside the Transformer block, showing the transition from $\mathbb{R}$ to $\mathbb{Z}$ space, GyRoPE-infused attention, SAIL feed-forward, and residual routing.

### ⛵ SAIL (Self-Activating Interference Layer)
The main idea. In standard $\mathbb{R}$-nets, non-linear activation functions (like GELU or SwiGLU) in the feed-forward layer are strictly necessary. However, multiplying two complex numbers is inherently non-linear in itself. **SAIL** is essentially SwiGLU without the "Swi": It splits the input via a Linear Layer into two parts and multiplies them together. That alone is sufficient to get a fully trainable complex transformer. Combined with residual connections `z = z + SAIL(z)`, my guess is it could also be chaotic somehow (think Mandelbrot set).

### 🪢 GyRoPE - learnable RoPE, full circle
While ~~implementing~~ experimenting with RoPE in my $\mathbb{R}$-net, I found that a trainable RoPE initialized randomly from $-2\pi$ to $2\pi$ yielded results comparable to classical positive positional encodings. This proved to work nicely with the complex transformers. However the weights are few and they change *really* slowly during training.

* $\mathbb{C}$-RoPE rotates in the $2\mathcal{D}$ complex plane.
* $\mathbb{H}$-RoPE rotates around a unit 3-sphere ($S^3$).
* $\mathbb{O}$ and $\mathbb{S}$ rotate around their respective hyperspherical counterparts.

### 👀 Attention
How do you route attention in complex spaces? I experimented with various variants, but turns out: **simplicity wins**. Attention scores are computed via the Hermitian dot product ($Q K^\dagger$). By extracting only the **real component** before applying the causal mask and standard Softmax:

```python
# In C-Space:
wei = (q @ k.adjoint()) * (2 * self.head_size) ** -0.5
wei = wei.real
wei = F.softmax(wei.masked_fill(self.tril[:T, :T] == 0, float("-inf")), dim=-1)
# Weighted value aggregation remains hypercomplex:
out = wei.to(v.dtype) @ v
```

The real part perfectly captures the phase alignment and magnitude between queries and keys after multiplication, acting as a natural scalar affinity score. The resulting real weights are then used to scale the full complex/hypercomplex value vectors ($V$). No other variant yielded a better validation loss so far.

### 🏋️ Weight Initialization
Attention and SAIL weights are initialized around the unit circle/sphere and scaled by:

$$\sqrt{\frac{S}{d_{in} + d_{out}}}$$

(where $S$ is $\mathbb{C}=1.0$, $\mathbb{H}=0.5$, $\mathbb{O}=0.25$, $\mathbb{S}=0.125$)

### 🛌 Embedding & LM-Head
To map tokens into hypercomplex spaces, I used separate embedding tables - one for each real component (e.g., 4 tables for $\mathbb{H}$, 16 for $\mathbb{S}$) and initialized them with a scaled normal distribution (Var = 1.0):

```python
# Quaternion Embedding (4D)
z1 = torch.complex(emb_r1(x), emb_i1(x))
z2 = torch.complex(emb_r2(x), emb_i2(x))
---
 nn.init.normal_(emb.weight, std=(1 / math.sqrt(4)))
```

To project back to vocabulary logits, the `LMHead` simply flattens all complex components back to $\mathbb{R}$:

```python
features = torch.cat([z1.real, z1.imag, z2.real, z2.imag], dim=-1)
return self.lin(features)
```

### 🛡️ RMSNorm - Norming Amplitudes, Preserving Phases
 🧨 remove for exploding transformer noises (NaN) 💥
 
Standard normalization layers can easily distort geometric phase information in complex spaces.

The custom **RMSNorm** calculates the Root Mean Square solely based on vector magnitudes ($\Vert{}q\Vert{}^2 = \vert{}z_1\vert{}^2 + \vert{}z_2\vert{}^2$):

```python
# Calculate quaternion amplitude squared
ampsq = (z1 * z1.conj()).real + (z2 * z2.conj()).real
rms = torch.rsqrt((ampsq / 2.0).mean(dim=-1, keepdim=True) + self.eps)

return (z1 * rms) * self.weight, (z2 * rms) * self.weight
```

This scales feature amplitudes cleanly across layers while leaving phase angles completely intact.

<br>
<br>


</details>

## 🔥 Hot Curves (Benchmarks & Results)
<details>
<summary><b>Click to expand 🔻</b></summary>

<br>

### 🌊 SAIL Interference Patterns ($\mathbb{C}$)

#### 🎬 2D SAIL Interference Evolution
[![4 Layer SAIL](https://img.youtube.com/vi/nFiWQWMtX-Q/0.jpg)](https://www.youtube.com/watch?v=nFiWQWMtX-Q)
[![8 Layer SAIL](https://img.youtube.com/vi/OIvpBzp8Y8c/0.jpg)](https://youtube.com/shorts/OIvpBzp8Y8c?si=LPVmYf6Skw9IRELd)

#### 🎬 3D SAIL Phase-Amplitude Topography
[![3D 4 Layer SAIL](https://img.youtube.com/vi/r6ykSiY3ipY/0.jpg)](https://youtu.be/r6ykSiY3ipY?si=VosKe4IpfWTy1jbz)
[![3D 8 Layer SAIL](https://img.youtube.com/vi/7kjrDF7-UUg/0.jpg)](https://youtube.com/shorts/7kjrDF7-UUg?si=jcLONq29jUDC2mVX)

*Figure: Evolution of selective phase interference across all 8 Transformer layers over training steps. Deep layers (6 & 7) demonstrate extreme destructive phase cancellation alongside razor-sharp resonance peaks (> 10,000 amplitude). That's why the Post-Norm design makes a lot of sense here*

<br>

### 📉 Comparison Runs

shared parameters:

| batch_size | block_size | dropout | learning rate | vocab_size |
| :--------- | :--------: | :-----: | :-----------: | ---------- |
| 32         |    128     |   0.2   |    0.0005     | 65         |

#### $\mathbb{C}$ vs $\mathbb{R}$ / both ~180k params
<img width="1745" height="828" alt="Image" src="https://github.com/user-attachments/assets/0d506bfc-471a-4ffa-bf06-c882151ad470" />

| Model          |    Space     | $n_{embd}$ | Layers | Heads | Params | Train Loss | Val Loss  |
| :------------- | :----------: | :--------: | :----: | :---: | :----: | :--------: | :-------: |
| **ComplexGPT** | $\mathbb{C}$ |     54     |   4    |   6   | ~179k  | **1.472**  | **1.704** |
| **RealGPT**    | $\mathbb{R}$ |     60     |   4    |   6   | ~181k  |   1.526    |   1.759   |

#### $\mathbb{R}$ vs the complex rest

Number of parameters scaled. Note: the higher dimensional runs take *forever*. A "Sedenion" needs 256 multiplications instead of one.

<img width="1760" height="842" alt="Image" src="https://github.com/user-attachments/assets/78f152ca-ceab-435a-9b36-629031519223" />
*Figure: Training and validation loss trajectories across spaces ($\mathbb{R}, \mathbb{C}, \mathbb{H}, \mathbb{O}, \mathbb{S}$). Note how Complex-383k (green) consistently outperforms the larger Real-baseline (blue).*

| Model             |    Space     | $n_{embd}$ | Layers | Heads | Params | Train Loss | Val Loss  |
| :---------------- | :----------: | :--------: | :----: | :---: | :----: | :--------: | :-------: |
| **RealGPT**       | $\mathbb{R}$ |     96     |   5    |   6   | ~567k  |   1.367    |   1.645   |
| **ComplexGPT**    | $\mathbb{C}$ |     72     |   5    |   6   | ~384k  | **1.359**  |   1.633   |
| **QuaternionGPT** | $\mathbb{H}$ |     54     |   4    |   4   | ~356k  |   1.367    | **1.633** |
| **OctonionGPT**   | $\mathbb{O}$ |     32     |   4    |   4   | ~264k  |   1.398    |   1.668   |
| **SedenionGPT**   | $\mathbb{S}$ |     18     |   4    |   6   | ~184k  |   1.460    |   1.702   |

<br>

<details>
<summary><b>v0.2.0 Bonus - Centumduodevigintunions 🔻</b></summary>
<img width="1247" height="809" alt="image" src="https://github.com/user-attachments/assets/81aa403d-4203-48ef-8b90-ff45367b2f1c" />
It's ridiculous, pointless and I can't even spell it... but it runs (very slowly). The most interesting usecases may be found on level 1-3 ;)
</details>

<br><br>


</details>


## 🚀 Quickstart & Usage


### Install

```bash
pip install tardits
# or
uv add tardits
```

Optional - for [MLFlow](https://github.com/mlflow/mlflow) experiment tracking:

```bash
pip install tardits[mlflow]
```

### Quickstart

#### Example configuration (save as .yaml):
```yaml
device: "auto" # Choices: "auto", "cuda", "cpu", "mps"
seed: 1337

model:
  level: 1 # Complex
  block_size: 64
  n_head: 6
  n_embd: 42
  n_layer: 4
  dropout: 0.2

trainer:
  batch_size: 32
  max_iters: 5000
  eval_interval: 500
  learning_rate: 0.0005
  eval_iters: 200
  save_interval: 0
  checkpoint: "saved_model.pt"

mlflow:
  experiment: "" # add name to enable
  log_interval: 10
  deep_log_interval: 0
  system_metrics: false

data:
  input_path: "my_training_data.txt"
  vocab_save_path: "saved/charset.txt"
```
You can also take a config from [examples/configs] and the shakespeare.txt form [examples/data]


#### Start Training
```bash
ttrain -c config.yaml
```

#### Generate Text
```bash
tgen -c config.yaml -p "To be or not to be" -l 200

tgen -h
options:
  -h, --help            show this help message and exit
  --config, -c CONFIG   Path to YAML configuration file
  --checkpoint CHECKPOINT
                        Optional override for model checkpoint path
  --vocab VOCAB         Optional override for vocab path
  --prompt, -p PROMPT   Text prompt to start generation
  --length, -l LENGTH   Number of tokens to generate
  --temp, -t TEMP       Model sampling temperature
  --top_k, -k TOP_K     Top-k sampling threshold
  --follow, -f          Endless stream of tokens
  --delay DELAY         Delay between tokens in follow mode (seconds)
```


### v0.2.0 (Current Release)
- [x] **Dynamic Dimension Scaling:** Universal recursive Cayley-Dickson engine ($2^k$ dimensions, up to 128D Centumduodevigintunion)
- [x] **Package Refactoring:** Clean library structure for distribution
- [x] **CLI Tools:** Easy executions via `ttrain` and `tgen`
- [x] **PyPI Release:** Official package published on PyPI


## 🧭 Outlook v0.3.0 - Tesseract
- [x] **Logit Distillation Engine:** `DistillTrainer` supporting dynamic teacher-student logit alignment (e.g., SmolLM2)
- [x] **Synth Training Data:** script to generate customized synthetic training data
- [x] **Interactive TUI:** ~~Simple~~ amazing terminal-based chat interface (using `prompt-toolkit`)
- [ ] **Basic Inference API** + docker/compose
- [ ] **Tesseract Model Series:** Releasing small-footprint complex chat models (2D/4D) on HuggingFace
- [x] **Getting distracted** by SAIL-experiments for possible future releases 🤫🧪⛵
