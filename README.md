# Bridging Domains through Subspace-Aware Model Merging

This is the official repository for the paper:

> **[Bridging Domains through Subspace-Aware Model Merging](https://arxiv.org/abs/2603.05768)**<br>
> [Levy Chaves](https://scholar.google.com/citations?user=dDfbSWUAAAAJ), [Chao Zhou](https://scholar.google.com/citations?user=ttO5HpQAAAAJ), [Rebekka Burkholz](https://scholar.google.com/citations?user=vkWBb2wAAAAJ), [Eduardo Valle](https://scholar.google.com/citations?user=lxWPqWAAAAAJ), [Sandra Avila](https://scholar.google.com/citations?user=TbM0rGQAAAAJ)<br>
> CVPR 2026

**TL;DR**: Our paper studies model merging under domain shifts. Our method, **SCORE** (Subspace COnflict-Resolving mErging), resolves conflicts between task vectors using SVD-based subspace alignment.

<p align="center">
<img style="width:70%;" alt="thumbnail" src="assets/CVPR_Model_Merging_OOD.png">
</p>

> **Abstract**: Model merging integrates multiple task-specific models into a single consolidated one. Recent research has made progress in improving merging performance for in-distribution or multi-task scenarios, but domain generalization in model merging remains underexplored. We investigate how merging models fine-tuned on distinct domains affects generalization to unseen domains. Through an analysis of parameter competition in the task matrix using singular value decomposition, we show that merging models trained under different distribution shifts induces stronger conflicts between their subspaces compared to traditional multi-task settings. To mitigate this issue, we propose SCORE (Subspace COnflict-Resolving mErging), a method designed to alleviate such singular subspace conflicts. SCORE finds a shared orthogonal basis by computing the principal components of the concatenated leading singular vectors of all models. It then projects each task matrix into the shared basis, pruning off-diagonal components to remove conflicting singular directions. SCORE consistently outperforms, on average, existing model merging approaches in domain generalization settings across a variety of architectures and model scales, demonstrating its effectiveness and scalability.

## Overview

The core workflow in this repository is:

1. Fine-tune one model per domain (or per task ordering).
2. Convert fine-tuned checkpoints to task vectors relative to a shared pretrained checkpoint.
3. Merge task vectors with different methods (MagMax, TIES, SCORE variants, etc.).
4. Evaluate merged models with leave-one-domain-out protocol.

Core implementations live in:

- `src/merging/task_vectors.py` (task-vector arithmetic)
- `src/merging/score.py` (SCORE method)
- `src/merging/utils.py` (merge method dispatcher)
- `merge_domain_splitted_leave_one_out.py` (main merging/eval script)

## Installation

Create the conda environment from the provided file:

```bash
conda env create -f environment.yml
conda activate score_cvpr26
```

## Supported Models

- ViT-B-16
- ViT-B-32
- ViT-L-14

Pretrained checkpoint convention:

```text
checkpoints/{MODEL}/zeroshot.pt
```

## Supported Datasets

- ImageNetR
- DomainNet
- OfficeHome
- PACS
- FedISIC
- RetinaDomains
- NICOpp
- TerraIncognita

Dataset paths and per-dataset epoch defaults are configured in:

- `src/constants.py`

## Main Scripts

### Training

- `finetune_domain_splitted.py`: fine-tunes one model per domain .
- `finetune_8datasets.py`: 8-dataset training script used for separate experiments.

### Merging and Evaluation

- `merge_domain_splitted_leave_one_out.py`: leave-one-domain-out merge + eval pipeline.
- `merge_domain_splitted_DIL.py`: alternative DIL merge pipeline.
- `eval_domain_splitted_leave_one_out.py`: upper-bound/expert evaluation per left-out domain.
- `ensemble_domain_splitted_leave_one_out.py`: ensemble baseline.

### Example (Fine-tuning + Merging)

Example below runs one experiment on `ImageNetR` with `ViT-B-32`:

```bash
# 1) Fine-tune domain-specific checkpoints
python3 finetune_domain_splitted.py \
    --model ViT-B-16 \
    --dataset ImageNetR \
    --seed 5

# 2) Merge domain task vectors (leave-one-domain-out evaluation)
python3 merge_domain_splitted_leave_one_out.py \
    --model ViT-B-16 \
    --dataset ImageNetR \
    --merge-fn ours_v1 \
    --seed 5
```

Outputs are written under `checkpoints/` (fine-tuned encoders) and `results/merging/` (merge metrics JSON files).


## Merge Methods

CLI option `--merge-fn` supports (from `src/args.py`):

- `magmax`
- `avg`
- `ties`
- `isoc`
- `iso_cts`
- `tsv`
- `ours_v1`, `ours_v2`, `ours_v3`, `ours_v4` (SCORE variants. V1 is the one we used in the paper. Other variants are ablations. )
- `pcb`

## Output Paths

Checkpoints:

```text
checkpoints/{model}/domain_incremental/{dataset}/checkpoint_ep:{epochs}-lr:{lr}_{config_or_task}.pt
checkpoints/{model}/sequential_finetuning/domain_incremental/{dataset}/...
```

Merging results:

```text
results/merging/{model}/{dataset}/{merge_fn}/{seq-ft|ind-ft}/merging-{model}-{dataset}-DIL-{seq-ft|ind-ft}.json
```

## Repository Structure

```text
src/
    args.py
    constants.py
    eval.py
    heads.py
    modeling.py
    utils.py
    datasets/
    merging/
checkpoints/
results/
scripts/
```

## Credits

This project builds on:

- [task_vectors](https://github.com/mlfoundations/task_vectors)
- [ties-merging](https://github.com/prateeky2806/ties-merging)
- [TSV](https://github.com/AntoAndGar/task_singular_vectors)

## Citation

If you find this work useful, please consider citing it:

```bibtex
@inproceedings{chaves2026bridging,
    title     = {Bridging Domains through Subspace-Aware Model Merging},
    author    = {Chaves, Levy and Zhou, Chao and Burkholz, Rebekka and Valle, Eduardo and Avila, Sandra},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2026}
}
```
