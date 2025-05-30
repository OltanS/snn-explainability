### SCORECAM
Apply scorecam, compare outputs to "cumulative membrane potential" for class in next timestep, as its version of "logits"

### TODO
- Unpack imagenet and ucf
- Figure out why vgg isn't working, fix it.

Ideas:
- Check to see if VGG works without BPTT

### PAST
- Get "cumulative membrane potential over time" from forward function, and its deltas DONE
- Get scorecam working DONE
- Save/load the model to enable inference time cam-ing. DONE
- Script for training the model DONE
- Begin working with the model during inference to ensure all is as intended, one pic at a time. DONE

- Add BPTT to basic_snn, see if it breaks it. -> Seems to work OK, but not helpful. Perhaps BPTT not great for truly neuromorphic data, and many BPTT's cause issues?
- Pool before, instead of after -> Seems to work OK