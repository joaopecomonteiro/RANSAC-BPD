# RANSAC-Based Plane Detection
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository contains the official PyTorch implementation of RANSAC-BPD

This project was done with [@pinthoz](https://github.com/pinthoz) and [@70m9](https://github.com/70m9)

To run the RANSAC you need to have a point cloud. We have created a way to generate a map and get the point cloud in the Webots simulator.

## Map generation 

To generate a random map you need run the `generate_map.py` script in the root directory:
```
python generate_map.py
```

## Point cloud scanning
After you have the map you will need to scan the point cloud with the help of the Webots simulator. The robot need to have its name changed to `EPUCK` and its `supervisor` setting turned to `True`.
You can scan the point cloud by running the `scan_pcd.py` script in the root directory:
```
python scan_pcd.py
```

## Plane detection
After you have your point cloud you can perform the plane detection by running the `main.py` script in the root directory:
```
python main.py
```



