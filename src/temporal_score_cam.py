import torch
import torch.nn as nn

from src.snn_utils import compute_raw_mem, compute_delta_raw_mem

class TemporalScoreCAM:
    """
    Goal: 
    - Score is determined by the change of membrane potential between timesteps,
    We can compute this in one go per channel by passing the masked input through the network
    and then computing how much it increases the membrane potential.
    """
    def __init__(self, snn_model, threshold=1.0):
        self.threshold = threshold
        self.snn_model = snn_model

    def get_cam_weights(self, input_tensor, target_layer):
        '''
        input tensor -> input into the network, unaltered, b, t, c, h, w
        target_layer -> the layer we are interested in (last conv layer)
        targets -> the target class (e.g. 0, 1, 2, etc.)

        returns the cumulative membrane potential change for each channel for all classes for all timesteps.
        '''
        with torch.no_grad():
            upsample = nn.UpsamplingBilinear2d(size=input_tensor.shape[-2:]) # last two dimensions are height and width
            activations = self.extract_activations(input_tensor, target_layer)
            batch_size, timesteps, num_channels, _, _ = activations.shape # something like , B, T, 64, 14, 14 whatever
            input_channels = input_tensor.shape[2]  # Input channels

            upsampled_activations = []
            # don't upsample the time dimension
            for t in range(timesteps):
                act_t = activations[:, t] 
                ups_t = upsample(act_t)

                # normalize the activations
                mins = ups_t.view(batch_size, num_channels, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
                maxs = ups_t.view(batch_size, num_channels, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
                normalized = (ups_t - mins) / (maxs - mins + 1e-8)
                upsampled_activations.append(normalized)
            
            upsampled_activations = torch.stack(upsampled_activations, dim=1) # b, t, c, h, w

            weights = []

            CHANNEL_BATCH_SIZE = 16
            for c_batch_idx in range(0, num_channels, CHANNEL_BATCH_SIZE):
                c_end_idx = min(c_batch_idx + CHANNEL_BATCH_SIZE, num_channels)
                batch_channels = list(range(c_batch_idx, c_end_idx))

                masked_inputs = []
                for c in batch_channels:
                    mask = upsampled_activations[:, :, c:c+1].repeat(1, 1, input_channels, 1, 1)
                    masked_inputs.append(input_tensor * mask)
                
                stacked_masked_inputs = torch.cat(masked_inputs, dim=0)
                deltas = self.compute_channel_deltas(stacked_masked_inputs) # (BATCH, TIME, NUM_CLASSES)
                channel_deltas = deltas.reshape(len(batch_channels), batch_size, timesteps, -1) # (C_BATCH, BATCH, TIME, NUM_CLASSES)
                channel_deltas = channel_deltas.permute(1, 2, 0, 3) # (BATCH, TIME, C_BATCH, NUM_CLASSES)
                weights.append(channel_deltas)

            weights = torch.cat(weights, dim=2) # (BATCH, TIME, NUM_CHANNELS, NUM_CLASSES)
            return weights

    def generate_cam(self, input_tensor, target_layer):
        '''
        Generate the CAM for the specified targets
        
        input_tensor -> input into the network, b, t, c, h, w
        target_layer -> the layer we are interested in
        targets -> the target class indices (optional)
        
        returns CAM of shape (batch_size, timesteps, height, width)
        '''
        with torch.no_grad():
            activations = self.extract_activations(input_tensor, target_layer) # (B, T, C', H', W')
            weights = self.get_cam_weights(input_tensor, target_layer) # (B, T, C', NUM_CLASSES) 
            
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
            
            return cams
        
    def compute_channel_deltas(self, masked_input):
        '''
        computer the delta in cumulative membrane potential between timesteps for all classification targets
        masked_input -> input tensor with the scorecam channel mask applied
        needs to be of shape (B, T, C, H, W) so batching is possible.
        '''
        out_spikes, out_mem = self.snn_model(masked_input)
        raw_mem = compute_raw_mem(out_spikes, out_mem, self.threshold) # (BATCH, TIME, NUM_CLASSES)
        # changes in cumulative membrane potential for all classes across all timesteps
        delta_raw_mem = compute_delta_raw_mem(raw_mem) # (BATCH, TIME, NUM_CLASSES)
        return delta_raw_mem
        
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
        model = BasicSNN(num_classes=num_classes).to(device)
        
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
        
        # Test compute_channel_deltas
        def test_compute_channel_deltas():
            # Create a small batch of masked inputs
            masked_input = torch.rand(4, timesteps, in_channels, height, width, device=device)
            deltas = score_cam.compute_channel_deltas(masked_input)
            print(f"Channel deltas shape: {deltas.shape}")
            # Check shape: (batch_size, timesteps, num_classes)
            assert deltas.shape == (4, timesteps, num_classes)
            return True
        
        # Test get_cam_weights
        def test_get_cam_weights():
            weights = score_cam.get_cam_weights(input_tensor, target_layer)
            print(f"Weights shape: {weights.shape}")
            # Weights should be (batch_size, num_channels, timesteps, num_classes)
            b, t, cc, hc, wc = score_cam.extract_activations(input_tensor, target_layer).shape
            assert len(weights.shape) == 4
            assert weights.shape[0] == batch_size
            assert weights.shape[1] == timesteps
            assert weights.shape[2] == cc
            assert weights.shape[3] == num_classes
            return True
        
        # Test generate_cam
        def test_generate_cam():
            cams = score_cam.generate_cam(input_tensor, target_layer)
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
        
        # Run all tests
        tests = [
            ("extract_activations", test_extract_activations),
            ("compute_channel_deltas", test_compute_channel_deltas),
            ("get_cam_weights", test_get_cam_weights),
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