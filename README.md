<div align="center">
<h1>Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs</h1>
Xiaoran Liu<sup>1,2*</sup>, Yuerong Song<sup>1,2*</sup>, Zhigeng Liu<sup>1,2</sup>, Zengfeng Huang<sup>1,2</sup>, 

Qipeng Guo<sup>2,3</sup>, Zhaoxiang Liu<sup>4</sup>, Shiguo Lian<sup>4</sup>, Ziwei He<sup>2,†</sup>, Xipeng Qiu<sup>1,2,†</sup>

<sup>1</sup> Fudan Univerisity, <sup>2</sup>Shanghai Innovation Institute, <sup>3</sup>Shanghai AI Lab, <sup>4</sup>China Unicom

[<a href="https://arxiv.org/abs/2512.07525">📝 Paper</a>] | [<a href="https://huggingface.co/papers/2512.07525">🔥 HF</a>] | [<a href="https://github.com/OpenMOSS/rope_pp">🚀 Code</a>] | [<a href="https://huggingface.co/collections/SII-xrliu/rope">🤗 Model</a>]
</div>

## Introduction

In this work, we propose ***RoPE++***, which re-injects the discarded imaginary component as a new group of attention heads computed in parallel with the real attentions. Particularly, we introduce ***RoPE++_EH*** that keeps equal attention head number while halving QKV parameters as well as KV cache, and ***RoPE++_EC*** that keeps equal cache size and doubles the number of attention heads. 

We first identify the loss of imaginary information in the complex form of RoPE and find it advantageous for capturing long-context dependencies. Compared with the real attention exhibiting stronger semantic locality, the imaginary attentions attend more to long-context information on average, promising gains on long-context tasks. Moreover, adding imaginary attention also exposes qk to a wider positional information range, implicitly improving length extrapolation.

Pre-training and evaluation at 376M and 776M sizes show that RoPE++_EH and RoPE++_EC outperform vanilla RoPE and other position embeddings on average across short- and long-context tasks. Further analysis reveals that the imaginary attentions play a dominant role in modeling long-context dependencies, confirming the effectiveness of introducing imaginary attention for improved long-context capability.

<p align="center">
<img src="./img/ropepp_main.png" width="750"/>
<p>

<p align="center">
<img src="./img/ropepp_equ.png" width="750"/>
<p>

<p align="center">
<img src="./img/ropepp_config.png" width="750"/>
<p>

## Installation

### Prepare Your OpenCompass

We run our downstream evaluation based on [OpenCompass](https://github.com/open-compass/opencompass).

```bash
git clone https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```

The necessary Python packages we use and their corresponding versions.

```
flash-attn==2.7.4.post1
torch==2.6.0
transformers==4.51.0
opencompass==0.4.2
```

### Prepare Your Model

Copy the folder `rope_pp/rope_pp/` to `opencompass/models/` and add the following line to the end of `opencompass/models/__init__.py`.

```python
from .rope_pp.rope_pp_wrapper import RoPEPPCausalLM
from .rope_pp.fope_wrapper import FoPECausalLM
from .rope_pp.alibi_wrapper import AlibiCausalLM
from .rope_pp.pythia_wrapper import PythiaCausalLM
from .rope_pp.mask_wrapper import MaskCausalLM
```

### Prepare Your Dataset

We pretrain our model with [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) and calculate validation loss with [OpenDataLab/Pile-CC](https://opendatalab.com/OpenDataLab/Pile-CC). 

Please download these datasets and copy the paths to the corresponding positions in our training Python scripts before training.

## Training

### Short-Context Training

1. Make sure you have download [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) and [OpenDataLab/Pile-CC](https://opendatalab.com/OpenDataLab/Pile-CC).

2. Make the following directory in `rope_pp/`.

```
checkpoints/
logs/
results/
wandb/
```

3. Take 776M RoPE++_EC as an example, execute the following command. Here, `imag2` stands for RoPE++_EC, and `imag1` stands for RoPE++_EH.

```bash
set -x
port=$(shuf -i25000-30000 -n1)
wait
export WANDB_RUN_GROUP="rope_pp-776m"

wait
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_rope_pp.py --config_abbr '776m' --imag --imag_mode 'imag2' --save_abbr 'rope_pp-776m-4k-imag2' > logs/rope_pp-776m-4k-imag2.log 2>&1
wait
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_rope_pp-decay.py --config_abbr '776m' --imag --imag_mode 'imag2' --save_abbr 'rope_pp-776m-4k-imag2' --load_ckpt 90000 --decay_step 10000 > logs/rope_pp-776m-4k-imag2-ckpt90000-decay.log 2>&1
```

### Long-Context Training

4. Take 776M RoPE++_EC as an example, execute the following command. 

```bash
set -x
port=$(shuf -i25000-30000 -n1)
wait
export WANDB_RUN_GROUP="rope_pp-776m"

wait # for NTK
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_rope_pp-lctx.py --config_abbr '776m' --imag --imag_mode 'imag2' --save_abbr 'rope_pp-776m-4k-imag2-ckpt90000-decay' --load_ckpt 10000 > logs/rope_pp-776m-4k-imag2-ckpt90000-decay-lctx.log 2>&1

wait  # for Linear PI
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_rope_pp-lctx-linear.py --config_abbr '776m' --factor 8  --imag --imag_mode 'imag2' --save_abbr 'rope_pp-776m-4k-imag2-ckpt90000-decay' --load_ckpt 10000 > logs/rope_pp-776m-4k-imag2-ckpt90000-decay-lctx-pi8.log 2>&1

wait  # for YaRN
deepspeed --master_port "$port" --include localhost:0,1,2,3,4,5,6,7 \
train_rope_pp-lctx-linear.py --config_abbr '776m' --yarn --factor 32 --imag --imag_mode 'imag2' --save_abbr 'rope_pp-776m-4k-imag2-ckpt90000-decay' --load_ckpt 10000 > logs/rope_pp-776m-4k-imag2-ckpt90000-decay-lctx-yarn32.log 2>&1
```

These training commands are detailed in `scripts/train-776m-rope_pp_ec.sh`. We also provide more bash scripts for other methods or other model scales.

> Note: please execute these in scripts under `rope_pp/` directory.

### Download Our Models

You can also download our checkpoints from Huggingface.

## Evaluation

Copy the folder `rope_pp/eval/` to your OpenCompass directory, and then you can try the following evaluations.

### Short-Context Evaluation

1. We evaluate short-context performance on tasks mainly in [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), including TruthfulQA, PIQA, HellaSwag, WinoGrande, ARC-e, GPQA, SocialIQA, OpenBookQA, and SuperGLUE. All models are tested within a 4k context length. 

2. Execute the following command.

```bash
python run.py eval/eval_rope_pp_short.py --dump-eval-details -r
wait
python run.py eval/eval_fope_short.py --dump-eval-details -r
wait
python run.py eval/eval_alibi_short.py --dump-eval-details -r
wait
python run.py eval/eval_pythia_short.py --dump-eval-details -r
```

### Long-Context Evaluation

1. We evaluate long-context performance at varying lengths on synthetic benchmarks, RULER, and BABILong.

2. Before evaluation, we need first to edit the prompt format of the RULER and BABILong to enable the base model to respond more effectively. In files like `ruler_cwe_gen.py` under the path `opencompass/configs/datasets/ruler/`, and files like `babilong_4k_gen.py` under the path `opencompass/configs/datasets/babilong/`, comment out the '\n' at the end of the prompt. The following is an example in `opencompass/configs/datasets/ruler/ruler_vt_gen.py`.

```python
vt_datasets = [
    {
        'abbr': 'ruler_vt',
        'type': RulerVtDataset,
        'num_chains': 1,
        'num_hops': 4,
        'reader_cfg': dict(input_columns=['prompt'], output_column='answer'),
        'infer_cfg': dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{prompt}'),
                        # dict(role='BOT', prompt='{answer}\n'),    # comment out this line
                    ]
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer),
        ),
        'eval_cfg': dict(
            evaluator=dict(type=RulerVtEvaluator),
        ),
    }
]
```

3. Execute the following command.

```bash
python run.py eval/eval_rope_pp_ruler.py --dump-eval-details -r
wait
python run.py eval/eval_rope_pp_babilong.py --dump-eval-details -r
```

### Perplexity (PPL) Evaluation

> We calculate the perplexity in `rope_pp/` directory instead of OpenCompass as follows.

1. We measure perplexity on WikiText and LAMBADA by executing the following command.

```bash
python test_ppl.py
```

### Noise Experiment

1. To verify how imaginary attention captures long-context dependencies and to contrast it with real attention in RoPE++, we design the noise experiment.

2. We add Gaussian noise with equal standard deviation to the imaginary and real attention components separately, and monitor the change in RoPE++ performance on RULER-4k.

3. Results can be acquired by executing the following command.

```bash
python run.py eval/eval_rope_pp_ruler_mask.py --dump-eval-details -r
```

## Results

<p align="center">
<img src="./img/ropepp_results.png" width="750"/>
<p>

<p align="center">
<img src="./img/ropepp_attn.png" width="750"/>
<p>

## Citation 

```
@article{liu2025beyond,
  title={Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs},
  author={Liu, Xiaoran and Song, Yuerong and Liu, Zhigeng and Huang, Zengfeng and Guo, Qipeng and Liu, Zhaoxiang and Lian, Shiguo and He, Ziwei and Qiu, Xipeng},
  journal={arXiv preprint arXiv:2512.07525},
  year={2025}
}
```
