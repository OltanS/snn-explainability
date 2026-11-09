# Attribution for spiking neural networks

Code for *Continuous Attribution on Discrete Spikes: Revealing the Explainability
Paradox in Spiking Neural Networks*, Neuromorphic Computing and Engineering, 2026
([DOI](https://doi.org/10.1088/2634-4386/aeae68)).

Implements class activation maps for SNNs built with SpikingJelly, including
Grad-CAM, AblationCAM, SAM and SmoothGrad with Gaussian and Poisson noise.

## Layout

```
train_model_ddp.py   train a model
compare_cams.py      compare attribution methods
pointing_game.py     pointing game evaluation
models/              network definitions
src/cams/            attribution methods
src/utils/           data, training and CLI helpers
notebooks/           analysis notebooks
```

## Usage

```bash
pip install -r requirements.txt

torchrun --nproc_per_node=gpu train_model_ddp.py \
    --model_name spiking_vgg11_bn --dataset_name dvs_gesture
torchrun --nproc_per_node=1 compare_cams.py \
    --model_name spiking_vgg11_bn --dataset_name dvs_gesture
```

All scripts share the arguments in `src/utils/args_utils.py`.

## License

MIT
