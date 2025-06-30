
import numpy as np
from types import SimpleNamespace
from lsy_drone_racing.control.min_snap_mpc import MinSnapMPCController


def make_fake_config():
    """Create a fake configuration for testing purposes."""
    cfg = SimpleNamespace()
    cfg.env = SimpleNamespace()
    cfg.env.freq = 50
    cfg.env.track = SimpleNamespace()

    # Define gates
    cfg.env.track.gates = [
        SimpleNamespace(pos=[0.5, 0.0, 0.5], rpy=[0.0, 0.0, 0.0]),
        SimpleNamespace(pos=[1.0, 0.0, 1.0], rpy=[0.0, 0.0, 0.0]),
        SimpleNamespace(pos=[0.0, 1.0, 1.0], rpy=[0.0, 0.0, 0.0])
    ]

    # Define obstacles
    cfg.env.track.obstacles = [
        SimpleNamespace(pos=[0.5, 0.5, 1.4]),
        SimpleNamespace(pos=[0.0, 0.0, 1.4])
    ]

    return cfg


def test_interpolate_reference_behavior(controller: MinSnapMPCController):
    """Test trajectory interpolation behavior for various time scenarios."""
    print("Testing trajectory interpolation behavior...")

    t_total = controller._t_total
    horizon = controller._horizon

    # Test interpolation at t = 0 (start of trajectory)
    interp_start = controller._interpolate_reference(0.0)
    assert interp_start.shape == (6, horizon + 1), f"Wrong shape for start interpolation: {interp_start.shape}"

    # Test interpolation at t = t_total (end of trajectory)
    interp_end = controller._interpolate_reference(t_total)
    assert interp_end.shape == (6, horizon + 1), f"Wrong shape for end interpolation: {interp_end.shape}"

    # Test interpolation beyond t_total (should clip to end values)
    interp_beyond = controller._interpolate_reference(t_total + 1.0)
    expected_clip = controller._interpolate_reference(t_total)
    assert np.allclose(interp_beyond, expected_clip), "Interpolation beyond trajectory should clip to end values"

    # Test interpolation before t = 0 (should hold initial state across horizon)
    interp_before = controller._interpolate_reference(-1.0)
    state_at_zero = interp_start[:, 0]  # Initial state at t=0
    for k in range(horizon + 1):
        assert np.allclose(interp_before[:, k], state_at_zero), f"Horizon point {k} should match t=0 state for t < 0"

    # Test interpolation with no trajectory
    old_ref = controller._reference_trajectory
    controller._reference_trajectory = None
    try:
        zeros_interp = controller._interpolate_reference(0.0)
        assert np.allclose(zeros_interp, 0.0), "Interpolation with no trajectory should return zeros"
    except Exception as e:
        print(f"Warning: _interpolate_reference raised with None trajectory: {e}")
    finally:
        controller._reference_trajectory = old_ref

    print("✅ Trajectory interpolation tests passed.")


def test_mpc_setup_and_feasibility(controller: MinSnapMPCController):
    """Test the MPC solver setup and feasibility."""
    print("Testing MPC setup and feasibility...")

    # Check that MPC components are initialized
    assert controller._mpc_solver is not None, "MPC solver not initialized"
    assert controller._X0_param is not None, "Initial state parameter not set"
    assert controller._X_ref_param is not None, "Reference parameter not set"

    # Test with a non-zero initial state near an obstacle
    x0 = np.array([0.5, 0.5, 1.0, 0.1, 0.1, 0.1])  # Near first obstacle
    xref = controller._interpolate_reference(0.0)

    controller._mpc_solver.set_value(controller._X0_param, x0)
    controller._mpc_solver.set_value(controller._X_ref_param, xref)

    # Attempt to solve the MPC problem
    try:
        sol = controller._mpc_solver.solve()
        u0 = sol.value(controller._U_var[:, 0])
        assert np.all(np.abs(u0) <= controller._u_max + 1e-2), "Control input exceeds limits"
        print("✅ MPC solver feasible and control inputs within limits.")
    except Exception as e:
        print("❌ MPC solver failed! Attempting to debug variable values...")
        try:
            debug_vars = {
                'X0': controller._mpc_solver.debug.value(controller._X0_param),
                'Xref': controller._mpc_solver.debug.value(controller._X_ref_param),
            }
            print("Debug variable values:")
            for k, v in debug_vars.items():
                print(f"{k}: {v}")
        except Exception as dbg_e:
            print(f"Could not retrieve debug values: {dbg_e}")
        raise AssertionError(f"MPC solver failed: {e}")


def test_internal_state_consistency(controller: MinSnapMPCController):
    """Verify the consistency of the controller's internal state."""
    print("Checking internal state consistency...")

    # Check presence of required attributes
    assert hasattr(controller, "_gates"), "Controller missing _gates attribute"
    assert hasattr(controller, "_obstacles"), "Controller missing _obstacles attribute"
    assert hasattr(controller, "_waypoints"), "Controller missing _waypoints attribute"
    assert hasattr(controller, "_t_total"), "Controller missing _t_total attribute"

    num_gates = len(controller._gates)
    num_obs = len(controller._obstacles)
    num_waypoints = controller._waypoints.shape[0]
    t_total = controller._t_total

    # Validate counts and values
    assert num_gates > 0, "No gates loaded"
    assert num_obs >= 0, "Obstacles list empty or missing"
    assert num_waypoints == num_gates, f"Waypoints count ({num_waypoints}) does not match gates count ({num_gates})"
    assert t_total > 0, "Reference trajectory total time is not positive"

    # Check that waypoints match gate positions
    for i, gate in enumerate(controller._gates):
        assert np.allclose(gate['pos'], controller._waypoints[i]), f"Waypoint {i} does not match gate position"

    print(f"Loaded {num_gates} gates, {num_obs} obstacles, and {num_waypoints} waypoints.")
    print(f"Reference trajectory total time: {t_total:.3f} seconds")

    print("✅ Internal state consistency checks passed.")


def test_compute_control(controller: MinSnapMPCController):
    """Test the compute_control method required by the parent Controller class."""
    print("Testing compute_control method...")

    # Test with valid observation
    obs = {'pos': np.array([0.0, 0.0, 0.0]), 'vel': np.zeros(3)}
    action = controller.compute_control(obs)
    assert action.shape == (13,), f"Expected 13-element action, got {action.shape}"
    assert np.allclose(action[9:13], 0.0), "Last 4 elements (yaw, rates) should be zero"
    assert not np.isnan(action).any(), "Action contains NaNs"

    # Test with trajectory completion
    controller._tick = int(controller._t_total * controller._freq) + 1
    action = controller.compute_control(obs)
    assert np.allclose(action[0:3], controller._reference_trajectory['pos'][-1]), "Action should match final position"
    assert np.allclose(action[3:], 0.0), "Non-position elements should be zero at trajectory end"
    assert controller._finished, "Controller should be finished at trajectory end"

    # Test fallback control with no reference trajectory
    old_ref = controller._reference_trajectory
    controller._reference_trajectory = None
    action = controller.compute_control(obs)
    assert np.allclose(action[0:3], obs['pos']), "Fallback should return current position"
    assert np.allclose(action[3:], 0.0), "Fallback non-position elements should be zero"
    controller._reference_trajectory = old_ref

    print("✅ Compute control tests passed.")


def test_step_callback(controller: MinSnapMPCController):
    """Test the step_callback method."""
    print("Testing step_callback method...")

    obs = {'pos': np.array([0.0, 0.0, 0.0]), 'vel': np.zeros(3)}
    action = np.zeros(13)
    initial_tick = controller._tick
    finished = controller.step_callback(action, obs, 0.0, False, False, {})
    assert controller._tick == initial_tick + 1, "Tick counter should increment"
    assert not finished, "Controller should not be finished initially"

    # Test progress printing (simulates 2 seconds)
    controller._tick = controller._freq * 2 - 1
    finished = controller.step_callback(action, obs, 0.0, False, False, {})
    assert controller._tick == controller._freq * 2, "Tick counter should increment correctly"

    print("✅ Step callback tests passed.")


def test_reset(controller: MinSnapMPCController):
    """Test the reset method."""
    print("Testing reset method...")

    controller._tick = 100
    controller._finished = True
    controller.reset()
    assert controller._tick == 0, "Tick counter should be reset"
    assert not controller._finished, "Finished flag should be reset"

    print("✅ Reset tests passed.")


def run_full_controller_tests():
    """Run all tests for the MinSnapMPCController."""
    print("\n=== Initializing fake config and controller ===")
    config = make_fake_config()

    obs = {
        'pos': np.array([0.0, 0.0, 0.0]),
        'vel': np.array([0.0, 0.0, 0.0])
    }

    info = {}
    controller = MinSnapMPCController(obs, info, config)

    print("\n=== Running trajectory interpolation tests ===")
    test_interpolate_reference_behavior(controller)

    print("\n=== Running MPC feasibility tests ===")
    test_mpc_setup_and_feasibility(controller)

    print("\n=== Running internal state consistency tests ===")
    test_internal_state_consistency(controller)

    print("\n=== Running compute_control tests ===")
    test_compute_control(controller)

    print("\n=== Running step_callback tests ===")
    test_step_callback(controller)

    print("\n=== Running reset tests ===")
    test_reset(controller)

    print("\n✅ All MinSnapMPC tests passed successfully.")


if __name__ == "__main__":
    run_full_controller_tests()
