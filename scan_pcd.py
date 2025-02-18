import os
import platform

if platform.system() == "Linux":
    os.environ["WEBOTS_HOME"] = "/usr/local/webots"

from controller import Supervisor, GPS, Compass, Lidar

from src.utils import warp_robot
from src.transformations import create_tf_matrix

from tqdm import tqdm
import numpy as np
import math
import pickle

import open3d as o3d

def main() -> None:
    supervisor: Supervisor = Supervisor()

    timestep: int = int(supervisor.getBasicTimeStep())  # in ms

    lidar: Lidar = supervisor.getDevice('lidar')
    lidar.enable(timestep)
    lidar.enablePointCloud()

    gps: GPS = supervisor.getDevice('gps')
    gps.enable(timestep)
    supervisor.step()
    gps_readings: [float] = gps.getValues()
    robot_position: (float, float) = (gps_readings[0], gps_readings[1])

    compass: Compass = supervisor.getDevice('compass')
    compass.enable(timestep)
    compass_readings: [float] = compass.getValues()
    robot_orientation: float = math.atan2(compass_readings[0], compass_readings[1])

    warp_robot(supervisor, "EPUCK", robot_position)

    data = []

    y = np.arange(0.1, 10.2, 0.2)
    x = np.repeat(np.arange(0.1, 10.2, 0.2), len(y))

    # Calculate the necessary number of repeats for full_y
    num_repeats = (len(x) // len(y)) + 1

    # Create full_y by repeating the pattern enough times and then slicing to the correct length
    full_y = np.tile(np.concatenate((y, y[::-1])), num_repeats)[:len(x)]

    # Now full_y has the same length as x
    moves = [(x[i], full_y[i]) for i in range(len(x))]

    print(f"number of moves: {len(moves)}")

    map_name = "test_map_3"

    filename = f"point_clouds/{map_name}.npy"

    with open(f'worlds/custom_maps/{map_name}_mask.pkl', 'rb') as f:
        mask = pickle.load(f)

    num_points_not_scanned = 0
    for new_position in tqdm(moves):
        x, y = int(new_position[0]*1000), int(new_position[1]*1000)

        if x>0 and x<len(mask) and y>0 and y<len(mask) and mask[y][x]==0:
            supervisor.step()

            warp_robot(supervisor, "EPUCK", new_position)
            gps_readings = gps.getValues()
            robot_position = (gps_readings[0], gps_readings[1])
            compass_readings = compass.getValues()
            robot_orientation = math.atan2(compass_readings[0], compass_readings[1])

            pcd = lidar.getPointCloud()

            robot_tf: np.ndarray = create_tf_matrix((robot_position[0], robot_position[1], 0.0), robot_orientation)
            data_tmp = np.array([[point.x, point.y, 0, 1] for point in pcd if math.isfinite(point.x) and math.isfinite(point.y) and math.isfinite(point.z)])
            z_tmp = np.array([point.z for point in pcd if math.isfinite(point.x) and math.isfinite(point.y) and math.isfinite(point.z)])

            data_tmp = data_tmp.T

            tf_data = robot_tf @ data_tmp
            tf_data = tf_data.T
            tf_data = tf_data[:, :3]
            tf_data[:, 2] = z_tmp

            data += list(tf_data)
        else:
            num_points_not_scanned += 1
    print(f"Did not scan in {num_points_not_scanned}  locations")
    data_arr = np.array(data)

    np.save(filename, data_arr)

    data_arr = data_arr[data_arr[:, -1] >= 0]
    print(f"Got {data_arr.shape[0]} points")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data_arr)
    o3d.visualization.draw_plotly([point_cloud])


if __name__ == '__main__':
    main()
