# SNI-SLAM: Semantic Neural Implicit SLAM
Siting Zhu*, Guangming Wang*, Hermann Blum, Jiuming Liu, Liang Song, Marc Pollefeys, Hesheng Wang
<div align="center">
  <h3>CVPR 2024 [<a href="https://arxiv.org/pdf/2311.11016.pdf">Paper</a>] [<a href="https://drive.google.com/file/d/1oRKoly8cxple0Z3CcgbBvC_8wYQhOtR3/view?usp=drive_link">Suppl</a>]</h3>
</div>
<div align="center">
  <h3>T-PAMI 2025 [<a href="https://ieeexplore.ieee.org/document/11260914">Paper</a>] [<a href="https://irmvlab.github.io/sni-slam-plus.github.io/">Project Page</a>]</h3>
</div>

## Demo

<p align="center">
  <a href="">
    <img src="./demo/sem_mapping.gif" alt="Logo" width="80%">
  </a>
</p>

## Installation

This repo uses [pixi](https://pixi.sh) for dependency management (Python 3.14, PyTorch, CUDA 12).

**Requirements:** Linux, CUDA-capable GPU (driver ≥ 12.2).

### 1. Install pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc   # or open a new terminal
```

### 2. Install conda dependencies

```bash
CONDA_OVERRIDE_CUDA=12.2 pixi install
```

The `CONDA_OVERRIDE_CUDA` override is required because pixi checks for a CUDA driver virtual package that may not be visible on login/compute nodes. Set the value to match your installed driver (`nvidia-smi` → top-right corner).

### 3. Install tiny-cuda-nn

[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) is a CUDA extension that must be compiled from source. It cannot be resolved through pixi, so install it once with:

```bash
CONDA_OVERRIDE_CUDA=12.2 pixi run install-tcnn
```

This task sets all required env vars automatically. It auto-detects your GPU's compute capability at compile time, so it works on any NVIDIA GPU.

### 4. Verify the installation

```bash
pixi run python -c "import tinycudann as tcnn, torch; print('tcnn OK')"
```

> **Note:** The original repo used conda + `environment.yaml` targeting Python 3.7 and CUDA 11.3. This version has been updated to use pixi with a modern stack (Python 3.14, CUDA 12+). The `environment.yaml` is kept for reference but is not used.

## Run
### Replica
1. Download the data with semantic annotations in [google drive](https://drive.google.com/drive/u/0/folders/1BCu8bCGKG9HmnLFbyx7DIHI0slgkeo4h) and save the data into the `./data/replica` folder. We only provide a subset of Replica dataset. For all Replica data generation, please refer to directory `data_generation`.
2. Download the pretrained segmentation network in [google drive](https://drive.google.com/drive/u/0/folders/1BCu8bCGKG9HmnLFbyx7DIHI0slgkeo4h) and save it into the `./seg` folder (unzip `seg/facebookresearch_dinov2_main.zip`).

Run SNI-SLAM:
```bash
CONDA_OVERRIDE_CUDA=12.2 pixi run python run.py configs/Replica/room1.yaml
```
The mesh for evaluation is saved as `$OUTPUT_FOLDER/mesh/final_mesh_eval_rec_culled.ply`

### Quick test with fewer frames

To test the pipeline without running all frames, add `n_img` to your config:
```yaml
# in configs/Replica/room1.yaml
data:
  input_folder: data/replica/room_1/
  output: output/Replica/room1/test
  n_img: 200  # remove this line for the full run
```

You can also set a coarser meshing resolution to speed up mesh generation:
```yaml
meshing:
  resolution: 0.05  # default is 0.01 — larger = faster but lower quality
```

### Mesh-only mode (skip re-running tracking/mapping)

If tracking and mapping already completed but mesh generation failed, you can reload the last checkpoint and regenerate the mesh without reprocessing all frames:

```bash
pixi run python -W ignore run.py configs/Replica/room1.yaml --mesh_only
```

This loads the latest checkpoint from `$OUTPUT_FOLDER/ckpts/`, reconstructs the keyframe data from disk, and runs meshing directly.


## Evaluation

### Average Trajectory Error
To evaluate the average trajectory error. Run the command below with the corresponding config file:
```bash
# An example for room1 of Replica
pixi run python src/tools/eval_ate.py configs/Replica/room1.yaml
```
### Reconstruction Metrics
We follow [code](https://github.com/JingwenWang95/neural_slam_eval) for reconstruction evaluation.

## Visualizing SNI-SLAM Results
For visualizing the results, we recommend to set `mesh_freq: 40` in [configs/SNI-SLAM.yaml](configs/SNI-SLAM.yaml) and run SNI-SLAM from scratch.

After SNI-SLAM is trained, run the following command for visualization.

```bash
pixi run python visualizer.py configs/Replica/room1.yaml --top_view --save_rendering
```
The result of the visualization will be saved at `output/Replica/room1/vis.mp4`. The green trajectory indicates the ground truth trajectory, and the red one is the trajectory of SNI-SLAM.

### Visualizer Command line arguments
- `--output $OUTPUT_FOLDER` output folder (overwrite the output folder in the config file)
- `--top_view` set the camera to top view. Otherwise, the camera is set to the first frame of the sequence
- `--save_rendering` save rendering video to `vis.mp4` in the output folder
- `--no_gt_traj` do not show ground truth trajectory

## Citing
If you find our code or paper useful, please consider citing:
```BibTeX
@inproceedings{zhu2024sni,
  title={Sni-slam: Semantic neural implicit slam},
  author={Zhu, Siting and Wang, Guangming and Blum, Hermann and Liu, Jiuming and Song, Liang and Pollefeys, Marc and Wang, Hesheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21167--21177},
  year={2024}
}
@ARTICLE{zhu2025sni,
  author={Zhu, Siting and Wang, Guangming and Blum, Hermann and Wang, Zhong and Zhang, Ganlin and Cremers, Daniel and Pollefeys, Marc and Wang, Hesheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={SNI-SLAM++: Tightly-Coupled Semantic Neural Implicit SLAM}, 
  year={2026},
  volume={48},
  number={3},
  pages={3399-3416}
}
```
