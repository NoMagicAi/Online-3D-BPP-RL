# test_module_2.py
import torch
import envs  # Import to register the environment
from acktr.model import Policy
from acktr.envs import make_vec_envs

# A simple class to replace SimpleNamespace for compatibility
class MockArgs:
    pass

def run_all_model_tests(batch_size, device):
    """
    A helper function to run a suite of tests for a given batch size and device.
    """
    print(f"\n--- Running Tests for batch_size={batch_size} on device='{device}' ---")

    # 1. Create the environment
    mock_args = MockArgs()
    mock_args.enable_rotation = True
    mock_args.box_size_set = [(2, 2, 2), (5, 5, 5)]
    mock_args.container_size = (10, 10, 10)
    mock_args.data_type = 'rs'
    
    envs = make_vec_envs(
        env_name='Bpp-v0', seed=1, num_processes=batch_size, gamma=None,
        log_dir=None, device=device, allow_early_resets=False, args=mock_args
    )

    # 2. Instantiate the Policy and move to the correct device
    policy = Policy(envs.observation_space.shape, envs.action_space)
    policy.to(device)
    print("[PASS] Policy object instantiated successfully.")
    
    # 3. Prepare dummy inputs
    obs = envs.reset()
    rnn_hxs = torch.zeros(batch_size, policy.recurrent_hidden_state_size).to(device)
    masks = torch.zeros(batch_size, 1).to(device)

    # 4. Test stochastic act() method
    value, action, log_probs, _ = policy.act(obs, rnn_hxs, masks)
    assert action.shape == (batch_size, 3), f"act() returned wrong action shape: {action.shape}"
    assert log_probs.shape == (batch_size, 1), f"act() returned wrong log_probs shape: {log_probs.shape}"
    print("[PASS] Stochastic act() executes and returns correct shapes.")

    # 5. Test deterministic act() method
    value, action, log_probs, _ = policy.act(obs, rnn_hxs, masks, deterministic=True)
    assert action.shape == (batch_size, 3), f"Deterministic act() returned wrong action shape: {action.shape}"
    print("[PASS] Deterministic act() executes and returns correct shapes.")

    # 6. Test evaluate_actions() method
    sample_action = torch.tensor([envs.action_space.sample() for _ in range(batch_size)]).to(device)
    value, log_probs, entropy, _ = policy.evaluate_actions(obs, rnn_hxs, masks, sample_action)
    assert log_probs.shape == (batch_size,), f"evaluate_actions returned wrong log_probs shape: {log_probs.shape}"
    assert isinstance(entropy.item(), float), "evaluate_actions entropy is not a float"
    print("[PASS] evaluate_actions() executes and returns correct shapes.")

    # 7. Test get_value() method
    value = policy.get_value(obs, rnn_hxs, masks)
    assert value.shape == (batch_size, 1), f"get_value() returned wrong shape: {value.shape}"
    print("[PASS] get_value() executes and returns correct shapes.")

    # 8. Test gradient flow
    assert all(p.requires_grad for p in policy.parameters()), "Not all policy parameters require gradients."
    # Dummy loss from critic and actor outputs
    value.mean().backward()
    log_probs.mean().backward()
    assert any(p.grad is not None for p in policy.parameters()), "No gradients were computed after backward pass."
    policy.zero_grad() # Clean up gradients for the next test run
    print("[PASS] Gradients are computed successfully.")
    print(f"--- Finished Tests for batch_size={batch_size} on device='{device}' ---")
    envs.close() # Add this line for graceful shutdown


if __name__ == '__main__':
    # Run all tests on CPU with a single environment
    run_all_model_tests(batch_size=1, device='cpu')
    
    # Re-run all tests on CPU with multiple environments
    run_all_model_tests(batch_size=4, device='cpu')

    # If a CUDA-enabled GPU is available, run tests there as well
    if torch.cuda.is_available():
        run_all_model_tests(batch_size=4, device='cuda')
    else:
        print("\n--- CUDA not available, skipping GPU tests ---")

    print("\n--- All Module 2 Tests Completed ---")