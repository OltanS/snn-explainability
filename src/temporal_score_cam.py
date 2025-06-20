import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_snn import BaseSNN

class TemporalScoreCAM:
    """
    Goal: 
    - Score is determined by the change of membrane potential between timesteps,
    We can compute this in one go per channel by passing the masked input through the network
    and then computing how much it increases the membrane potential.
    """
    def __init__(self, snn_model : BaseSNN, threshold=1.0):
        self.threshold = threshold
        self.snn_model = snn_model

    def get_cam_weights(self, input_tensor, activations):
        '''
        input tensor -> input into the network, unaltered, b, t, c, h, w
        activations -> activations of the target layer, b, t, c', h', w'

        '''
        input_channels = input_tensor.shape[2]  # Input channels
        with torch.no_grad():
            batch_size, timesteps, num_channels, _, _ = activations.shape # b, t, c_out, h', w'
            upsampled_activations = self.upsample_activations(activations, input_tensor, normalize=True) # b, t, c_out, h, w
            
            # time-chanel masked inputs
            CHANNEL_BATCH_SIZE = 16
            masked_inputs = []
            for c_batch_idx in range(0, num_channels, CHANNEL_BATCH_SIZE):
                c_end_idx = min(c_batch_idx + CHANNEL_BATCH_SIZE, num_channels)
                batch_channels = list(range(c_batch_idx, c_end_idx))

                for c in batch_channels:
                    mask = upsampled_activations[:, :, c:c+1].repeat(1, 1, input_channels, 1, 1)
                    masked_inputs.append(input_tensor * mask) # b, t, c, h, w
        
            stacked_masked_inputs = torch.stack(masked_inputs, dim=0) # c_out, B, t, c, h, w 
            # acquire baseline per-timestep outputs as well as memories
            baseline_outputs, mem_histories = self.snn_model(input_tensor)  # (BATCH, TIME, NUM_CLASSES) for out_vals
            # problem! membrane states assume the b, c', h', w' don't change.
            # But we want to push B * c_out, t, c, h, w through, which would not work.
            # We could broadcast the memory into B * c_out, t, c, h, w, but for now
            # likely best to just loop...
            all_scores = []
            for t in range(timesteps):
                channel_scores = []
                # restore the membrane states for the current timestep
                for c in range(stacked_masked_inputs.shape[0]):
                    self.snn_model.restore_membrane_states(mem_histories, t)
                    one_channel_masked_batched = stacked_masked_inputs[c]
                    # these are the output for all the c_out channel masked items in the batch that correspond to the timestep
                    # they are of shape b, num_classes
                    outputs = self.snn_model._forward_one_step(one_channel_masked_batched[:, t], t) # b, num_classes
                    baseline = baseline_outputs[:, t] # b, num_classes
                    scores = outputs - baseline
                    channel_scores.append(scores) 
                channel_scores = torch.stack(channel_scores, dim=1) # b, num_channels, num_classes
                all_scores.append(channel_scores)

            all_scores = torch.stack(all_scores, dim=1) # b, t, num_channels, num_classes
            return all_scores  #(BATCH, TIME, NUM_CHANNELS, NUM_CLASSES)

    def generate_cam_(self, input_tensor, target_layer):
        '''
        Generate the CAM for the specified targets
        
        input_tensor -> input into the network, b, t, c, h, w
        target_layer -> the layer we are interested in
        
        returns CAM of shape (batch_size, timesteps, height, width)
        '''
        with torch.no_grad():
            activations = self.extract_activations(input_tensor, target_layer) # (B, T, C', H', W')
            weights = self.get_cam_weights(input_tensor, activations) # (B, T, C', NUM_CLASSES) 
            
            # Get dimensions
            batch_size, timesteps, num_channels, h, w = activations.shape
            num_classes = weights.shape[3]

            weights = weights.unsqueeze(-1).unsqueeze(-1) # (B, T, C', NUM_CLASSES, 1, 1)
            activations_expanded = activations.unsqueeze(3) # (B, T, C', 1, H', W')

            cams = torch.zeros(batch_size, timesteps, num_classes, h, w, device=input_tensor.device)

            weighted_activations = weights * activations_expanded # (B, T, C', NUM_CLASSES, H', W')
            cams = weighted_activations.sum(dim=2) # (B, T, NUM_CLASSES, H', W')

            # Ensure all values are non-negative, ReLU-like
            cams = torch.clamp(cams, min=0)

            # per class normalization across temporal dimension, to show most important spatiotemporal features
            for c in range(num_classes):
                class_cams = cams[:, :, c]
                reshaped = class_cams.view(batch_size, timesteps, -1) # (BATCH, TIME, H'*W')
                mins = reshaped.min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0] # (B, 1, 1)
                maxs = reshaped.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0] # (B, 1, 1)
                mins = mins.view(batch_size, 1, 1, 1)  # (B, 1, 1, 1)
                maxs = maxs.view(batch_size, 1, 1, 1)  # (B, 1, 1, 1)
                cams[:, :, c] = (class_cams - mins) / (maxs - mins + 1e-8) # (B, T, H', W')
            
            return cams # (B, T, NUM_CLASSES, H', W')
    
    def rescale_cams(self, cams, input_tensor):
        B, T, Cc, Hf, Wf = cams.shape
        H0, W0 = input_tensor.shape[-2:]
        # reshape to (B*T, Cc, H', W') for interpolation
        cams = cams.view(B * T, Cc, Hf, Wf)
        # bilinear upsample to original H0×W0
        cams = F.interpolate(cams, size=(H0, W0), mode='bilinear', align_corners=False)
        # back to (B, T, Cc, H0, W0)
        cams = cams.view(B, T, Cc, H0, W0)
        return cams

    def generate_cam(self, input_tensor, target_layer):
        cams = self.generate_cam_(input_tensor, target_layer)
        cams = self.rescale_cams(cams, input_tensor)
        return cams # (B, T, NUM_CLASSES, H0, W0)
        
    def extract_activations(self, input_tensor, target_layer):
        '''
        input tensor -> input into the network, unaltered, b, t, c, h, w
        target_layer -> the layer we are interested in (last conv layer)
        '''
        activations = []

        def hook_fn(module, input, output):
            activations.append(output)

        # register hook to the target layer
        handle = target_layer.register_forward_hook(hook_fn)
        # forward pass through the model
        self.snn_model(input_tensor)
        # remove the hook
        handle.remove()
        return torch.stack(activations, dim=1) # (BATCH, TIME, NUM_CHANNELS, HEIGHT, WIDTH)

    def upsample_activations(self, activations, target_tensor, normalize=True):
        with torch.no_grad():
            batch_size, timesteps, num_channels, _, _ = activations.shape
            # bilinear upsample to the spatial size of target_tensor
            upsample = nn.UpsamplingBilinear2d(size=target_tensor.shape[-2:])
            upsampled_activations = []
            for t in range(timesteps):
                act_t = activations[:, t]           # (B, C, h', w')
                ups_t = upsample(act_t)             # (B, C, H, W)
                if normalize:
                    # per‐channel min/max over spatial dims
                    flat = ups_t.view(batch_size, num_channels, -1)
                    mins = flat.min(2, keepdim=True)[0].view(batch_size, num_channels, 1, 1)
                    maxs = flat.max(2, keepdim=True)[0].view(batch_size, num_channels, 1, 1)
                    ups_t = (ups_t - mins) / (maxs - mins + 1e-8)
                # append whether normalized or not
                upsampled_activations.append(ups_t)
            # stack back into (B, T, C, H, W)
            return torch.stack(upsampled_activations, dim=1)

    def find_most_important_cam_frames(self, cams, class_idx):
        '''
        Returns the indices cams for the class sorted by their importance.
        '''
        b, t, num_classes, h, w = cams.shape
        class_cams = cams[:, :, class_idx]  # (B, T, H, W)
        # flatten to (B, T, H*W)
        flat_class_cams = class_cams.view(b, t, -1)  # (B, T, H*W)
        importance = flat_class_cams.sum(dim=2)  # (B, T)

        return importance.argsort(dim=1, descending=True)  # (B, T), indices of most important frames per batch

# unit tests
if __name__ == "__main__":
    from models.basicsnn import BasicSNN
    import numpy as np
    import traceback
    
    def test_temporal_score_cam():
        # Test parameters
        batch_size = 3
        timesteps = 5
        in_channels = 2
        height, width = 128, 128
        num_classes = 11
        
        # Create a small BasicSNN model for testing
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BasicSNN(num_classes=num_classes, num_channels=in_channels, time_steps=timesteps, beta=0.95, enable_bntt=False).to(device)
        
        # Find the target convolutional layer (assuming it's the last conv layer)
        target_layer = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                target_layer = module
        
        if target_layer is None:
            print("No convolutional layer found in the model")
            return False
        
        # Create sample input tensor
        input_tensor = torch.rand(batch_size, timesteps, in_channels, height, width, device=device)
        
        # Initialize TemporalScoreCAM
        score_cam = TemporalScoreCAM(model)
        
        # Test extract_activations
        def test_extract_activations():
            activations = score_cam.extract_activations(input_tensor, target_layer)
            print(f"Activations shape: {activations.shape}")
            # Check if activations have correct shape (batch_size, timesteps, channels, h', w')
            assert len(activations.shape) == 5
            assert activations.shape[0] == batch_size
            assert activations.shape[1] == timesteps
            return True
        
        # Test get_cam_weights
        def test_get_cam_weights():
            activations = score_cam.extract_activations(input_tensor, target_layer)
            weights = score_cam.get_cam_weights(input_tensor, activations)
            print(f"Weights shape: {weights.shape}")
            # Weights should be (batch_size, num_channels, timesteps, num_classes)
            b, t, cc, hc, wc = activations.shape
            assert len(weights.shape) == 4
            assert weights.shape[0] == batch_size
            assert weights.shape[1] == timesteps
            assert weights.shape[2] == cc
            assert weights.shape[3] == num_classes
            return True
        
        # Test generate_cam
        def test_generate_cam_():
            cams = score_cam.generate_cam_(input_tensor, target_layer)
            print(f"CAM shape: {cams.shape}")
            b, t, cc, hc, wc = score_cam.extract_activations(input_tensor, target_layer).shape
            # CAMs should be (batch_size, timesteps, num_classes, height', width')
            assert len(cams.shape) == 5
            assert cams.shape[0] == batch_size
            assert cams.shape[1] == timesteps
            assert cams.shape[2] == num_classes
            assert cams.shape[3] == hc
            assert cams.shape[4] == wc
            # Check if CAMs are non-negative
            assert torch.all(cams >= 0)
            # Values should be between 0 and 1 after normalization
            assert torch.all((cams >= 0) & (cams <= 1))
            return True
        
        def test_rescale_cams():
            cams = score_cam.generate_cam_(input_tensor, target_layer)
            rescaled_cams = score_cam.rescale_cams(cams, input_tensor)
            print(f"Rescaled CAM shape: {rescaled_cams.shape}")
            # Rescaled CAMs should have the same shape as input_tensor
            assert rescaled_cams.shape == (batch_size, timesteps, num_classes, height, width)
            return True

        def test_generate_cam():
            cams = score_cam.generate_cam(input_tensor, target_layer)
            print(f"Generated CAM shape: {cams.shape}")
            # Generated CAMs should have the same shape as rescaled_cams
            assert cams.shape == (batch_size, timesteps, num_classes, height, width)
            # Check if CAMs are non-negative
            assert torch.all(cams >= 0)
            # Values should be between 0 and 1 after normalization
            assert torch.all((cams >= 0) & (cams <= 1))
            return True

        def test_generate_cam_extensive():
            # disect generate_cams, and examine each part with prints, make sure the logic is solid.
            # lets redefine dimensions for trivial printing
            batch_size = 1
            timesteps = 2
            in_channels = 1
            height, width = 4, 4
            num_classes = 3
            model = BasicSNN(num_classes=num_classes, num_channels=in_channels, time_steps=timesteps, beta=0.95, enable_bntt=False).to(device)
            score_cam = TemporalScoreCAM(model)
            activations = score_cam.extract_activations(input_tensor, target_layer)
            weights = score_cam.get_cam_weights(input_tensor, activations) # (B, T, C', NUM_CLASSES) 
            # TODO
        # Run all tests
        tests = [
            ("extract_activations", test_extract_activations),
            ("get_cam_weights", test_get_cam_weights),
            ("generate_cam", test_generate_cam_),
            ("rescale_cams", test_rescale_cams),
            ("generate_cam", test_generate_cam)
        ]
        
        all_passed = True
        print(f"Running units tests with batch size {batch_size}, timesteps {timesteps}, in_channels {in_channels}, height {height}, width {width}, num_classes {num_classes}...")
        for test_name, test_func in tests:
            print(f"Running test: {test_name}...")
            try:
                if test_func():
                    print(f"✅ {test_name} passed")
                else:
                    print(f"❌ {test_name} failed")
                    all_passed = False
            except Exception as e:
                print(f"❌ {test_name} failed with error:")
                traceback.print_exc()  # Print the full traceback
                all_passed = False
        
        if all_passed:
            print("All tests passed! 🎉")
        else:
            print("Some tests failed. Check the output above.")
    
    # Run all tests
    test_temporal_score_cam()