#!/usr/bin/env python3
"""
Comprehensive test suite for MinSnapMPCController.
Tests all functionality including MPC optimization, trajectory generation, and control logic.
"""

import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
import time
import sys
import traceback


def create_test_config():
    """Create a test configuration with gates and obstacles."""
    config = SimpleNamespace()
    config.env = SimpleNamespace()
    config.env.freq = 50
    config.env.track = SimpleNamespace()

    # Define test gates - simple path
    config.env.track.gates = [
        SimpleNamespace(pos=[0.45, -0.5, 0.56], rpy=[0.0, 0.0, 2.35]),
        SimpleNamespace(pos=[1.0, -1.05, 1.11], rpy=[0.0, 0.0, -0.78]),
        SimpleNamespace(pos=[0.0, 1.0, 0.56], rpy=[0.0, 0.0, 0.0]),
        SimpleNamespace(pos=[-0.5, 0.0, 1.11], rpy=[0.0, 0.0, 3.14])
    ]

    # Define test obstacles
    config.env.track.obstacles = [
        SimpleNamespace(pos=[1.0, 0.0, 1.4]),
        SimpleNamespace(pos=[0.5, -1.0, 1.4]),
        SimpleNamespace(pos=[0.0, 1.5, 1.4]),
        SimpleNamespace(pos=[-0.5, 0.5, 1.4])
    ]

    return config


def test_controller_initialization():
    """Test controller initialization and setup."""
    print("=" * 60)
    print("TESTING CONTROLLER INITIALIZATION")
    print("=" * 60)
    
    config = create_test_config()
    obs = {
        'pos': np.array([0.0, 0.0, 0.5]),
        'vel': np.array([0.0, 0.0, 0.0])
    }
    info = {}
    
    try:
        # Import your controller class here
        # from lsy_drone_racing.control.min_snap_mpc import MinSnapMPCController
        # For testing, we'll assume it's already imported
        
        controller = MinSnapMPCController(obs, info, config)
        
        # Test basic attributes
        assert hasattr(controller, '_gates'), "Missing _gates attribute"
        assert hasattr(controller, '_obstacles'), "Missing _obstacles attribute"
        assert hasattr(controller, '_waypoints'), "Missing _waypoints attribute"
        assert hasattr(controller, '_reference_trajectory'), "Missing _reference_trajectory attribute"
        
        print(f"✓ Controller initialized successfully")
        print(f"  - Gates loaded: {len(controller._gates)}")
        print(f"  - Obstacles loaded: {len(controller._obstacles)}")
        print(f"  - Waypoints generated: {controller._waypoints.shape[0]}")
        print(f"  - Trajectory total time: {controller._t_total:.2f}s")
        
        return controller
        
    except Exception as e:
        print(f"✗ Controller initialization failed: {e}")
        traceback.print_exc()
        return None


def test_trajectory_generation(controller):
    """Test trajectory generation functionality."""
    print("\n" + "=" * 60)
    print("TESTING TRAJECTORY GENERATION")
    print("=" * 60)
    
    try:
        # Check trajectory exists
        assert controller._reference_trajectory is not None, "No reference trajectory generated"
        
        traj = controller._reference_trajectory
        
        # Check trajectory structure
        required_keys = ['t', 'pos', 'vel', 'acc']
        for key in required_keys:
            assert key in traj, f"Missing trajectory key: {key}"
        
        # Check trajectory dimensions
        n_points = len(traj['t'])
        assert traj['pos'].shape == (n_points, 3), f"Position shape mismatch: {traj['pos'].shape}"
        assert traj['vel'].shape == (n_points, 3), f"Velocity shape mismatch: {traj['vel'].shape}"
        assert traj['acc'].shape == (n_points, 3), f"Acceleration shape mismatch: {traj['acc'].shape}"
        
        # Check trajectory values are finite
        assert np.all(np.isfinite(traj['pos'])), "Position contains non-finite values"
        assert np.all(np.isfinite(traj['vel'])), "Velocity contains non-finite values"
        assert np.all(np.isfinite(traj['acc'])), "Acceleration contains non-finite values"
        
        # Check trajectory starts at initial position
        initial_pos_error = np.linalg.norm(traj['pos'][0] - np.array([0.0, 0.0, 0.5]))
        assert initial_pos_error < 0.1, f"Trajectory doesn't start at initial position, error: {initial_pos_error}"
        
        # Check trajectory passes through waypoints (approximately)
        waypoint_errors = []
        for i, waypoint in enumerate(controller._waypoints):
            # Find closest point in trajectory to waypoint
            distances = np.linalg.norm(traj['pos'] - waypoint, axis=1)
            min_distance = np.min(distances)
            waypoint_errors.append(min_distance)
            
        max_waypoint_error = np.max(waypoint_errors)
        assert max_waypoint_error < 0.5, f"Trajectory deviates too much from waypoints: {max_waypoint_error}"
        
        print(f"✓ Trajectory generation successful")
        print(f"  - Trajectory points: {n_points}")
        print(f"  - Max waypoint error: {max_waypoint_error:.3f}m")
        print(f"  - Max velocity: {np.max(np.linalg.norm(traj['vel'], axis=1)):.2f} m/s")
        print(f"  - Max acceleration: {np.max(np.linalg.norm(traj['acc'], axis=1)):.2f} m/s²")
        
        return True
        
    except Exception as e:
        print(f"✗ Trajectory generation test failed: {e}")
        traceback.print_exc()
        return False


def test_interpolate_reference(controller):
    """Test reference trajectory interpolation."""
    print("\n" + "=" * 60)
    print("TESTING REFERENCE INTERPOLATION")
    print("=" * 60)
    
    try:
        # Test interpolation at different times
        test_times = [0.0, controller._t_total * 0.25, controller._t_total * 0.5, 
                     controller._t_total * 0.75, controller._t_total, controller._t_total + 1.0]
        
        for t in test_times:
            ref_horizon = controller._interpolate_reference(t)
            
            # Check dimensions
            expected_shape = (6, controller._horizon + 1)
            assert ref_horizon.shape == expected_shape, f"Wrong interpolation shape at t={t}: {ref_horizon.shape}"
            
            # Check values are finite
            assert np.all(np.isfinite(ref_horizon)), f"Non-finite values in interpolation at t={t}"
            
        # Test with no trajectory
        old_traj = controller._reference_trajectory
        controller._reference_trajectory = None
        zero_ref = controller._interpolate_reference(0.0)
        assert np.allclose(zero_ref, 0.0), "Should return zeros when no trajectory"
        controller._reference_trajectory = old_traj
        
        print(f"✓ Reference interpolation working correctly")
        print(f"  - Tested at {len(test_times)} different time points")
        print(f"  - Horizon length: {controller._horizon}")
        
        return True
        
    except Exception as e:
        print(f"✗ Reference interpolation test failed: {e}")
        traceback.print_exc()
        return False


def test_mpc_setup(controller):
    """Test MPC optimization setup."""
    print("\n" + "=" * 60)
    print("TESTING MPC SETUP")
    print("=" * 60)
    
    try:
        # Setup MPC (should be called automatically in compute_control)
        controller._setup_mpc()
        
        # Check MPC components exist
        assert controller._mpc_solver is not None, "MPC solver not created"
        assert controller._X0_param is not None, "Initial state parameter not created"
        assert controller._X_ref_param is not None, "Reference parameter not created"
        assert controller._X_var is not None, "State variables not created"
        assert controller._U_var is not None, "Control variables not created"
        
        print(f"✓ MPC setup successful")
        print(f"  - Horizon: {controller._horizon}")
        print(f"  - State dimension: 6 (pos + vel)")
        print(f"  - Control dimension: 3 (acceleration)")
        print(f"  - Obstacles: {len(controller._obstacles)}")
        
        return True
        
    except Exception as e:
        print(f"✗ MPC setup test failed: {e}")
        traceback.print_exc()
        return False


def test_mpc_solving(controller):
    """Test MPC optimization solving."""
    print("\n" + "=" * 60)
    print("TESTING MPC SOLVING")
    print("=" * 60)
    
    try:
        # Test multiple initial states
        test_states = [
            np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),  # Start position
            np.array([0.5, 0.5, 1.0, 0.1, 0.1, 0.0]),  # Mid trajectory
            np.array([1.0, 1.0, 1.1, 0.0, 0.2, 0.0])   # Near waypoint
        ]
        
        solve_times = []
        
        for i, state in enumerate(test_states):
            start_time = time.time()
            
            # Get reference trajectory
            current_time = i * 0.5  # Different times for each test
            ref_traj = controller._interpolate_reference(current_time)
            
            # Set parameters and solve
            controller._mpc_solver.set_value(controller._X0_param, state)
            controller._mpc_solver.set_value(controller._X_ref_param, ref_traj)
            
            sol = controller._mpc_solver.solve()
            solve_time = time.time() - start_time
            solve_times.append(solve_time)
            
            # Check solution quality
            u_opt = sol.value(controller._U_var[:, 0])
            x_opt = sol.value(controller._X_var)
            
            # Check control limits
            assert np.all(np.abs(u_opt) <= controller._u_max + 1e-3), f"Control limits violated: {u_opt}"
            
            # Check solution is finite
            assert np.all(np.isfinite(u_opt)), f"Non-finite control solution: {u_opt}"
            assert np.all(np.isfinite(x_opt)), f"Non-finite state solution"
            
        avg_solve_time = np.mean(solve_times)
        max_solve_time = np.max(solve_times)
        
        print(f"✓ MPC solving successful")
        print(f"  - Test cases solved: {len(test_states)}")
        print(f"  - Average solve time: {avg_solve_time*1000:.1f}ms")
        print(f"  - Maximum solve time: {max_solve_time*1000:.1f}ms")
        print(f"  - All solutions within control limits")
        
        return True
        
    except Exception as e:
        print(f"✗ MPC solving test failed: {e}")
        traceback.print_exc()
        return False


def test_compute_control(controller):
    """Test the main compute_control method."""
    print("\n" + "=" * 60)
    print("TESTING COMPUTE_CONTROL")
    print("=" * 60)
    
    try:
        # Test at different points in trajectory
        test_observations = [
            {'pos': np.array([0.0, 0.0, 0.5]), 'vel': np.array([0.0, 0.0, 0.0])},
            {'pos': np.array([0.5, 0.3, 0.8]), 'vel': np.array([0.2, 0.1, 0.0])},
            {'pos': np.array([1.2, 0.8, 1.1]), 'vel': np.array([0.1, 0.3, 0.0])}
        ]
        
        control_times = []
        
        for i, obs in enumerate(test_observations):
            controller._tick = i * 25  # Simulate different time steps
            
            start_time = time.time()
            action = controller.compute_control(obs)
            control_time = time.time() - start_time
            control_times.append(control_time)
            
            # Check action structure
            assert action.shape == (13,), f"Wrong action shape: {action.shape}"
            assert np.all(np.isfinite(action)), f"Non-finite action values: {action}"
            
            # Check action components
            pos_cmd = action[0:3]
            vel_cmd = action[3:6]
            acc_cmd = action[6:9]
            yaw_rates = action[9:13]
            
            # Check acceleration limits
            assert np.all(np.abs(acc_cmd) <= controller._u_max + 0.1), f"Acceleration command too high: {acc_cmd}"
            
            # Check yaw and rates are zero (as expected)
            assert np.allclose(yaw_rates, 0.0, atol=1e-3), f"Yaw/rates should be zero: {yaw_rates}"
            
        # Test trajectory completion
        controller._tick = int(controller._t_total * controller._freq) + 10
        final_obs = {'pos': np.array([0.0, 1.5, 0.8]), 'vel': np.array([0.0, 0.0, 0.0])}
        final_action = controller.compute_control(final_obs)
        assert controller._finished, "Controller should be finished at end of trajectory"
        
        avg_control_time = np.mean(control_times)
        max_control_time = np.max(control_times)
        
        print(f"✓ Compute control successful")
        print(f"  - Test cases: {len(test_observations)}")
        print(f"  - Average control time: {avg_control_time*1000:.1f}ms")
        print(f"  - Maximum control time: {max_control_time*1000:.1f}ms")
        print(f"  - Trajectory completion detected correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Compute control test failed: {e}")
        traceback.print_exc()
        return False


def test_fallback_control(controller):
    """Test fallback control mechanism."""
    print("\n" + "=" * 60)
    print("TESTING FALLBACK CONTROL")
    print("=" * 60)
    
    try:
        # Test fallback with valid trajectory
        current_pos = np.array([0.5, 0.5, 1.0])
        current_vel = np.array([0.1, 0.1, 0.0])
        current_time = 1.0
        
        fallback_action = controller._fallback_control(current_time, current_pos, current_vel)
        
        # Check fallback action structure
        assert fallback_action.shape == (13,), f"Wrong fallback action shape: {fallback_action.shape}"
        assert np.all(np.isfinite(fallback_action)), f"Non-finite fallback values: {fallback_action}"
        
        # Test fallback with no trajectory
        old_traj = controller._reference_trajectory
        controller._reference_trajectory = None
        
        no_traj_action = controller._fallback_control(current_time, current_pos, current_vel)
        assert np.allclose(no_traj_action[0:3], current_pos), "Should return current position when no trajectory"
        assert np.allclose(no_traj_action[3:], 0.0), "Other components should be zero"
        
        controller._reference_trajectory = old_traj
        
        print(f"✓ Fallback control working correctly")
        print(f"  - Handles valid trajectory case")
        print(f"  - Handles missing trajectory case")
        
        return True
        
    except Exception as e:
        print(f"✗ Fallback control test failed: {e}")
        traceback.print_exc()
        return False


def test_step_callback_and_reset(controller):
    """Test step callback and reset functionality."""
    print("\n" + "=" * 60)
    print("TESTING STEP CALLBACK AND RESET")
    print("=" * 60)
    
    try:
        # Test step callback
        initial_tick = controller._tick
        action = np.zeros(13)
        obs = {'pos': np.array([0.0, 0.0, 0.5]), 'vel': np.array([0.0, 0.0, 0.0])}
        
        finished = controller.step_callback(action, obs, 0.0, False, False, {})
        assert controller._tick == initial_tick + 1, "Tick should increment"
        assert not finished or controller._finished, "Finished status should be consistent"
        
        # Test reset
        controller._tick = 100
        controller._finished = True
        controller.reset()
        
        assert controller._tick == 0, "Tick should reset to 0"
        assert not controller._finished, "Finished flag should reset"
        
        print(f"✓ Step callback and reset working correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Step callback and reset test failed: {e}")
        traceback.print_exc()
        return False


def test_trajectory_info(controller):
    """Test trajectory information retrieval."""
    print("\n" + "=" * 60)
    print("TESTING TRAJECTORY INFO")
    print("=" * 60)
    
    try:
        info = controller.get_trajectory_info()
        
        # Check info structure
        expected_keys = ['total_time', 'current_time', 'progress', 'num_waypoints', 
                        'num_gates', 'num_obstacles', 'max_velocity', 'max_acceleration', 'finished']
        
        for key in expected_keys:
            assert key in info, f"Missing info key: {key}"
        
        # Check info values
        assert info['total_time'] > 0, "Total time should be positive"
        assert 0 <= info['progress'] <= 1, f"Progress should be 0-1: {info['progress']}"
        assert info['num_gates'] == len(controller._gates), "Gate count mismatch"
        assert info['num_obstacles'] == len(controller._obstacles), "Obstacle count mismatch"
        assert info['max_velocity'] > 0, "Max velocity should be positive"
        
        print(f"✓ Trajectory info working correctly")
        print(f"  - Total time: {info['total_time']:.2f}s")
        print(f"  - Progress: {info['progress']*100:.1f}%")
        print(f"  - Max velocity: {info['max_velocity']:.2f} m/s")
        print(f"  - Max acceleration: {info['max_acceleration']:.2f} m/s²")
        
        return True
        
    except Exception as e:
        print(f"✗ Trajectory info test failed: {e}")
        traceback.print_exc()
        return False


def plot_trajectory_visualization(controller):
    """Create visualization of the generated trajectory."""
    print("\n" + "=" * 60)
    print("CREATING TRAJECTORY VISUALIZATION")
    print("=" * 60)
    
    try:
        if controller._reference_trajectory is None:
            print("No trajectory to plot")
            return False
        
        traj = controller._reference_trajectory
        
        fig = plt.figure(figsize=(15, 10))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(traj['pos'][:, 0], traj['pos'][:, 1], traj['pos'][:, 2], 'b-', linewidth=2, label='Trajectory')
        
        # Plot waypoints
        waypoints = controller._waypoints
        ax1.scatter(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], 
                   c='red', s=100, marker='o', label='Waypoints')
        
        # Plot obstacles
        for i, obs in enumerate(controller._obstacles):
            pos = obs['pos']
            radius = obs['radius']
            # Simple sphere representation
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = pos[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = pos[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = pos[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            ax1.plot_surface(x, y, z, alpha=0.3, color='red')
        
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_zlabel('Z [m]')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        
        # Velocity profile
        ax2 = fig.add_subplot(2, 2, 2)
        vel_magnitude = np.linalg.norm(traj['vel'], axis=1)
        ax2.plot(traj['t'], vel_magnitude, 'g-', linewidth=2)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Velocity [m/s]')
        ax2.set_title('Velocity Profile')
        ax2.grid(True)
        
        # Acceleration profile
        ax3 = fig.add_subplot(2, 2, 3)
        acc_magnitude = np.linalg.norm(traj['acc'], axis=1)
        ax3.plot(traj['t'], acc_magnitude, 'r-', linewidth=2)
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Acceleration [m/s²]')
        ax3.set_title('Acceleration Profile')
        ax3.grid(True)
        
        # XY trajectory
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.plot(traj['pos'][:, 0], traj['pos'][:, 1], 'b-', linewidth=2, label='Trajectory')
        ax4.scatter(waypoints[:, 0], waypoints[:, 1], c='red', s=100, marker='o', label='Waypoints')
        
        # Plot obstacle projections
        for obs in controller._obstacles :
            pos = obs['pos']
            radius = obs['radius']
            circle = plt.Circle((pos[0], pos[1]), radius, color='red', alpha=0.3)
            ax4.add_patch(circle)
        
        ax4.set_xlabel('X [m]')
        ax4.set_ylabel('Y [m]')
        ax4.set_title('XY Trajectory View')
        ax4.legend()
        ax4.grid(True)
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig('trajectory_test_results.png', dpi=150, bbox_inches='tight')
        print(f"✓ Trajectory visualization saved as 'trajectory_test_results.png'")
        
        return True
        
    except Exception as e:
        print(f"✗ Trajectory visualization failed: {e}")
        traceback.print_exc()
        return False


def run_comprehensive_test():
    """Run all tests for the MinSnapMPCController."""
    print("STARTING COMPREHENSIVE MINSNAP MPC CONTROLLER TEST")
    print("=" * 80)
    
    test_results = {}
    controller = None
    
    # Test initialization
    controller = test_controller_initialization()
    test_results['initialization'] = controller is not None
    
    if controller is None:
        print("\n" + "=" * 80)
        print("CRITICAL ERROR: Controller initialization failed. Cannot continue tests.")
        print("=" * 80)
        return False
    
    # Run all tests
    test_functions = [
        ('trajectory_generation', test_trajectory_generation),
        ('interpolate_reference', test_interpolate_reference),
        ('mpc_setup', test_mpc_setup),
        ('mpc_solving', test_mpc_solving),
        ('compute_control', test_compute_control),
        ('fallback_control', test_fallback_control),
        ('step_callback_reset', test_step_callback_and_reset),
        ('trajectory_info', test_trajectory_info),
        ('visualization', plot_trajectory_visualization)
    ]
    
    for test_name, test_func in test_functions:
        test_results[test_name] = test_func(controller)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:25} : {status}")
    
    print("-" * 40)
    print(f"Total tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your MinSnapMPC controller is working correctly.")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    # Make sure to import your controller before running
    try:
        from lsy_drone_racing.control.min_snap_mpc import MinSnapMPCController
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    except ImportError as e:
        print(f"Error importing MinSnapMPCController: {e}")
        print("Please make sure the controller module is properly installed and accessible.")
        sys.exit(1)