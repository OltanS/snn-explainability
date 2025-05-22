### SCORECAM

1. Get scorecam working by forking and cloning from [repo](https://github.com/jacobgil/pytorch-grad-cam).
2. Apply scorecam, compare outputs to "cumulative membrane potential" for class in next timestep, as its version of "logits"

### TODO

- Get "cumulative membrane potential over time" from forward function, and its deltas DONE
- Get scorecam working
- Save/load the model to enable inference time cam-ing. DONE
- Script for training the model
- Beging working with the model during inference to ensure all is as intended, one pic at a time.

### FUTURE

- Turn classification video stream into events, using https://github.com/SensorsINI/v2e. 1 frame can be 1 timestep.
