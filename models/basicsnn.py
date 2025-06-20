import torch
import torch.nn as nn

import snntorch as snn

from models.base_snn import BaseSNN

class BasicSNN(BaseSNN):
    '''Simple convolutional SNN.'''
    def __init__(self, num_classes, num_channels, time_steps, beta, enable_bntt):
        super(BasicSNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1   = snn.BatchNormTT2d(32, time_steps) if enable_bntt else nn.ModuleList([nn.Identity() for _ in range(time_steps)])
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = snn.BatchNormTT2d(64, time_steps) if enable_bntt else nn.ModuleList([nn.Identity() for _ in range(time_steps)])
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = snn.BatchNormTT2d(64, time_steps) if enable_bntt else nn.ModuleList([nn.Identity() for _ in range(time_steps)])
        self.lif3 = snn.Leaky(beta=beta, init_hidden=True)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(64 * 4 * 4, num_classes)
        self.pool = nn.AvgPool2d(2)
        
    def _forward_one_step(self, xt, t):
        B, C, H, W = xt.shape # batch size,  number of channels(polarities), height, width
        c1 = self.conv1(xt)
        c1 = self.bn1[t](c1)
        s1 = self.lif1(c1)
        s1 = self.pool(s1)
        c2 = self.conv2(s1)
        c2 = self.bn2[t](c2)
        s2 = self.lif2(c2)
        s2 = self.pool(s2)
        c3 = self.conv3(s2)
        c3 = self.bn3[t](c3)
        s3 = self.lif3(c3)
        g2 = self.adaptive_pool(s3).view(B, -1)
        f  = self.fc1(g2)
        return f


if __name__ == "__main__":
    num_classes = 11
    #model = BasicSNN(num_classes=num_classes, num_channels=2, time_steps=5, beta=0.9, enable_bntt=True)
    batch_size, timesteps, channels, h, w = 8, 5, 2, 128, 128
    
    def test_create_mem_histories():
        model = BasicSNN(num_classes=num_classes, num_channels=channels, time_steps=timesteps, beta=0.9, enable_bntt=True)
        mem_histories = model.create_mem_histories()
        # Check that keys correspond to Leaky layers
        leaky_layer_names = [name for name, module in model.named_modules() if module.__class__.__name__ == "Leaky"]
        assert set(mem_histories.keys()) == set(leaky_layer_names)
        # Check that all values are empty lists
        for v in mem_histories.values():
            assert isinstance(v, list)
            assert len(v) == 0
        return True

    def test_reset_memories():
        model = BasicSNN(num_classes=num_classes, num_channels=channels, time_steps=timesteps, beta=0.9, enable_bntt=True)
        # Forward pass to initialize memories
        x = torch.randn(batch_size, timesteps, channels, h, w)
        model(x)
        # Manually set memories to nonzero
        for name, module in model.named_modules():
            if module.__class__.__name__ == "Leaky":
                module.mem = torch.full_like(module.mem, 42.0)
                #print(f"Shape of {name}.mem: {module.mem.shape}") # b, c, h, w
        # Check all Leaky mems are 42.0
        for name, module in model.named_modules():
            if module.__class__.__name__ == "Leaky":
                assert torch.all(module.mem == 42.0)
        # Reset
        model.reset_memories()
        # Check all Leaky mems are zero
        for name, module in model.named_modules():
            if module.__class__.__name__ == "Leaky":
                assert torch.all(module.mem == 0)
        return True
    
    def test_collect_memories():
        model = BasicSNN(num_classes=num_classes, num_channels=channels, time_steps=timesteps, beta=0.9, enable_bntt=True)
        x = torch.randn(batch_size, timesteps, channels, h, w)
        mem_histories = model.create_mem_histories()
        model(x)
        model.collect_memories(mem_histories)
        # Check that each history has one entry and is a tensor
        for v in mem_histories.values():
            assert len(v) == 1
            assert isinstance(v[0], torch.Tensor)
        return True

    def test_format_mem_histories():
        model = BasicSNN(num_classes=num_classes, num_channels=channels, time_steps=timesteps, beta=0.9, enable_bntt=True)
        x = torch.randn(batch_size, timesteps, channels, h, w)
        mem_histories = model.create_mem_histories()
        model(x)
        # Collect for two timesteps
        for _ in range(timesteps):
            model.collect_memories(mem_histories)
        model.format_mem_histories(mem_histories)
        for v in mem_histories.values():
            assert isinstance(v, torch.Tensor)
            print(f"Shape of membrane history tensor: {v.shape}")
            assert v.shape[0] == batch_size
            assert v.shape[1] == timesteps
        return True
    
    def test_restore_membrane_states():
        model = BasicSNN(num_classes=num_classes, num_channels=channels, time_steps=timesteps, beta=0.9, enable_bntt=True)
        x = torch.randn(batch_size, timesteps, channels, h, w)
        mem_histories = model.create_mem_histories()
        model(x)
        old_value = 24.0
        # Change all mems to a known value
        for name, module in model.named_modules():
            if module.__class__.__name__ == "Leaky":
                module.mem = torch.full_like(module.mem, old_value)
        # Collect for two timesteps
        for _ in range(2):
            model.collect_memories(mem_histories)
        model.format_mem_histories(mem_histories)
        # Change all mems to a known value
        for name, module in model.named_modules():
            if module.__class__.__name__ == "Leaky":
                module.mem = torch.full_like(module.mem, 42.0)
        # Restore to timestep 0, batch 0
        model.restore_membrane_states(mem_histories, timestep=0)
        # Check that mem[0] matches mem_histories at timestep 0
        for name, module in model.named_modules():
            if module.__class__.__name__ == "Leaky":
                expected = mem_histories[name][0]
                actual = module.mem[0]
                #print(f"Restored membrane state for {name}: expected {expected}, actual {module.mem[0]}")
                assert torch.allclose(actual, expected)
        return True
    
    def test_restore_membrane_states_extensive():
        model = BasicSNN(num_classes=num_classes, num_channels=channels, time_steps=timesteps, beta=0.9, enable_bntt=True)
        x = torch.randn(batch_size, timesteps, channels, h, w)
        baseline_outputs, mem_histories = model(x)
        
        # Store membrane states at t=2 - get from mem_histories, not current module.mem
        target_timestep = 2
        expected_states = {}
        for name, module in model.named_modules():
            if isinstance(module, snn.Leaky) and name in mem_histories:
                # Get the state at timestep 2 from the histories
                expected_states[name] = mem_histories[name][:, target_timestep].clone()
        
        # Do some other forward passes to change states
        dummy_input = torch.randn_like(x)
        model(dummy_input)
        
        # Restore and check
        model.restore_membrane_states(mem_histories, target_timestep)
        for name, module in model.named_modules():
            if isinstance(module, snn.Leaky) and name in expected_states:
                # print(f"Shape of module mem {module.mem.shape} Expected shape {expected_states[name].shape}")
                # print(module.mem[0][0])
                # print("---------------------------------------")
                # print(expected_states[name][0][0])
                assert torch.allclose(module.mem, expected_states[name])
        
        return True

    def test_forward_and_mem_histories():
        model = BasicSNN(num_classes=num_classes, num_channels=channels, time_steps=timesteps, beta=0.9, enable_bntt=True)
        x = torch.randn(batch_size, timesteps, channels, h, w)
        outs, mem_histories = model(x)

        # Check output shape: (batch, time, num_classes)
        assert outs.shape == (batch_size, timesteps, num_classes)

        # Check mem_histories keys correspond to Leaky layers
        leaky_layer_names = [name for name, module in model.named_modules() if module.__class__.__name__ == "Leaky"]
        assert set(mem_histories.keys()) == set(leaky_layer_names)

        # Check each mem_history is a tensor of shape (T, B, ...)
        for v in mem_histories.values():
            assert isinstance(v, torch.Tensor)
            assert v.shape[0] == batch_size
            assert v.shape[1] == timesteps


        return True
    
    def test__forward_without_reset():
        model = BasicSNN(num_classes=num_classes, num_channels=channels, time_steps=timesteps, beta=0.9, enable_bntt=True)
        x = torch.randn(batch_size, timesteps, channels, h, w)
        # initial forward to initialize memories
        throwaway_outs, throwaway_mem_histories = model.forward(x)
        for name, module in model.named_modules():
            if module.__class__.__name__ == "Leaky":
                module.mem = torch.full_like(module.mem, 50.0)

        first_outs, mem_histories = model._forward_without_reset(x)
        model.reset_memories()
        second_outs, second_mem_histories = model._forward_without_reset(x)
        # should be different after reset

        assert not torch.allclose(first_outs, second_outs)
        model.reset_memories()
        third_outs, third_mem_histories = model._forward_without_reset(x)
        # should be the same after reset
        assert torch.allclose(second_outs, third_outs)

        outs = first_outs
        # Output shape should be (batch, time, num_classes)
        assert outs.shape == (batch_size, timesteps, num_classes)
        # Mem histories keys should match Leaky layers
        leaky_layer_names = [name for name, module in model.named_modules() if module.__class__.__name__ == "Leaky"]
        assert set(mem_histories.keys()) == set(leaky_layer_names)
        return True

    def test_forward_from_state():
        model = BasicSNN(num_classes=num_classes, num_channels=channels, time_steps=timesteps, beta=0.9, enable_bntt=True)
        x = torch.randn(batch_size, timesteps, channels, h, w)
        print(f"Input shape: {x.shape}")
        # Run once to get mem_histories
        initial_outs, mem_histories = model(x)
        # Pick a random timestep and batch index
        timestep = 2
        # create version of x of inputs after timestep
        x = x[:, timestep + 1:]  # shape (1, timesteps-timestep, channels, h, w)
        # Run forward_from_state
        print(f"Input shape for forward_from_state: {x.shape} for restore to timestep {timestep} of total {timesteps}")
        outs2, mem_histories2 = model.forward_from_state(x, timestep, mem_histories)
        # Output shape should be (batch, time, num_classes)
        assert outs2.shape == (batch_size, timesteps-timestep - 1, num_classes)
        # get outs from timestep onwards
        relevant_outs = initial_outs[:, timestep + 1:]
        assert relevant_outs.shape == outs2.shape
        # Check that the outputs match
        assert torch.allclose(relevant_outs, outs2)
        return True
    
    tests = {
        "create_mem_histories": test_create_mem_histories,
        "reset_memories": test_reset_memories,
        "collect_memories": test_collect_memories,
        "format_mem_histories": test_format_mem_histories,
        "restore_membrane_states": test_restore_membrane_states,
        "restore_membrane_states_extensive" : test_restore_membrane_states_extensive,
        "forward_and_mem_histories": test_forward_and_mem_histories,
        "_forward_without_reset": test__forward_without_reset,
        "forward_from_state": test_forward_from_state,
    }

    all_passed = True
    print(f"Running units tests with batch size {batch_size}, timesteps {timesteps}, in_channels {channels}, height {h}, width {w}, num_classes {num_classes}...")
    with torch.no_grad():
        for test_name, test_func in tests.items():
            print(f"Running test: {test_name}...")
            try:
                if test_func():
                    print(f"✅ {test_name} passed")
                else:
                    print(f"❌ {test_name} failed")
                    all_passed = False
            except Exception as e:
                import traceback
                print(f"❌ {test_name} failed with error:")
                traceback.print_exc()  # Print the full traceback
                all_passed = False
    
    if all_passed:
        print("All tests passed! 🎉")
    else:
        print("Some tests failed. Check the output above.")