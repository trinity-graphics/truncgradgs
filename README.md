# TruncGradGS: Improved 3D Gaussian Splatting via Truncated Gradient Updates

**Pacific Graphics 2026** — *The 34th Pacific Conference on Computer Graphics and Applications*

## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:trinity-graphics/truncgradgs.git --recursive --depth=1
```
or
```shell
# HTTPS
git clone https://github.com/trinity-graphics/truncgradgs.git --recursive --depth=1
```

## Setup

### Docker Setup

**Prerequisites:** setup docker with the [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

I provide a convenient docker image, published at `dubiouscactus/cudags:latest`.

You can build it with this command: `docker build --rm -t cudags .`.

To run it, **you need to mount the data volume in a specific path because the dataset comes
with symbolic links for my system**:
```python
docker run -it --runtime=nvidia \
    --gpus all \
    --shm-size=64gb \
    --mount type=bind,src=<path/to/datasets/root>,dst=/home/cactus/Data/Blender \
    --mount type=bind,src=.,dst=/home/ \
    cuda-gs:latest
```

I typically mount the repository to /home/ so I can experiment in my docker image, and
that way you have direct access to the output models.

### Local Setup
For any other setup, refer to the original 3DGS repo.

### Running

For our experiments, I provide shell scripts which are self explanatory.

#### Experiment 1
For static scenes, we want to evaluate our method against the baseline (3DGS) in various
settings: random init, COLMAP init, COLMAP dense init, etc.

To run XP1, you must:
1. Set whether this is the baseline or not with this env var: `IS_BASELINE=1`.
2. Set the model initialization with this env var: `MODEL_INIT=<init_type>`. Must be one
   of `true`, `random`, `random_ours`, `colmap`, `colmap_dense`. For our synthetic scenes, we
   use `random_ours`.
3. Call the script (`run_exp_1_ours.sh` or `run_exp_1_general.sh` depending on the
   benchmark):
```
./scripts/run_exp_1_ours.sh <dataset_root> <scene> <white_bg> <method_name> <exp_name>
```

This will loop over 8 frames and synchronize the experiment to wandb.ai. For the
experiment name, please use `<init>-<method>` such as `random-baseline` or `random-ours`, etc.

#### Experiment 2

For our method, you will need the motion masks which can be generated with our script.
To make things easier, you can run it standalone (no virtual environment required) with uv:
```shell
uv run scripts/export_motion_masks.py --help
```

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{morales2026truncgradgs,
  title        = {TruncGradGS: Improved 3D Gaussian Splatting via Truncated Gradient Updates},
  author       = {Morales, Th{\'e}o and Le-Pham, Nhat-Quynh and Atkins, Robin and Hua, Binh-Son},
  booktitle    = {Proceedings of Pacific Graphics 2026},
  year         = {2026},
  publisher    = {The Eurographics Association},
  eprint       = {2609.03534},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```

