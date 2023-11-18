import numpy as np
from leap3d.scanning.scan_results import ScanResults
from leap3d.config import DATA_DIR

def test_laser_position_and_velocity():
    # Create a ScanResults object with a test file
    case_filename = DATA_DIR / "case_0000.npz"
    scan_results = ScanResults(case_filename)

    for timestep in range(scan_results.total_timesteps - 1):
        # Get the laser position and velocity at timestep t
        x_t0, y_t0, _, _, vx_t0, vy_t0, _ = scan_results.get_laser_data_at_timestep(timestep)
        x_t1, y_t1 = scan_results.get_laser_coordinates_at_timestep(timestep + 1)

        assert ((not np.equal(x_t0, x_t1) and vx_t0 != 0) or (vx_t0 == 0))
        assert ((not np.equal(y_t0, y_t1) and vy_t0 != 0) or (vy_t0 == 0))
