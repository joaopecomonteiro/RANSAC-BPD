import random
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

class Plane:
    """
    Implementation of planar RANSAC.

    Based on pyRANSAC-3D's implementation (https://github.com/leomariga/pyRANSAC-3D)
    
    """

    def __init__(self):
        self.inliers = []
        self.equation = []

    def fit(self, pts, thresh=0.05, minPoints=100, maxIteration=1000, orientation=None, random_seed=None):
        """
        Find the best equation for a plane.

        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param thresh: Threshold distance from the plane which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.
        :param random_seed: Random seed for reproducibility.
        :returns:
        - `self.equation`:  Parameters of the plane using Ax+By+Cy+D `np.array (1, 4)`
        - `self.inliers`: points from the dataset considered inliers
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        dbscan = DBSCAN(eps=0.2, min_samples=1)
        n_points = pts.shape[0]
        best_eq = []
        best_inliers = []

        for it in tqdm(range(maxIteration)):
            if orientation is None:
                # Samples 3 random points
                id_samples = random.sample(range(0, n_points), 3)
                pt_samples = pts[id_samples]
                
            elif orientation == "vertical":
                # Samples 2 random points and project the first one vertically
                id_samples = random.sample(range(0, n_points), 2)
                pt_samples = pts[id_samples]
                pt_proj = np.array([pt_samples[0, 0], pt_samples[0, 1], pt_samples[0, 2] + 0.7])
                pt_samples = np.vstack((pt_samples, pt_proj))
                
            elif orientation == "horizontal":
                # Samples 2 random points and project the first one horizontally
                id_samples = random.sample(range(0, n_points), 2)
                pt_samples = pts[id_samples]
                pt_proj = np.array([pt_samples[0, 0] + 0.7, pt_samples[0, 1], pt_samples[0, 2]])
                pt_samples = np.vstack((pt_samples, pt_proj))

            # We have to find the plane equation described by those 3 points
            # We find first 2 vectors that are part of this plane
            # A = pt2 - pt1
            # B = pt3 - pt1
            vecA = pt_samples[1, :] - pt_samples[0, :]
            vecB = pt_samples[2, :] - pt_samples[0, :]

            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC = np.cross(vecA, vecB)

            # Check if vecC is a zero vector
            if np.linalg.norm(vecC) == 0:
                continue  # Skip this iteration if the points are collinear

            # Normalize vecC
            vecC = vecC / np.linalg.norm(vecC)

            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[2]*z = -k
            # We have to use a point to find k
            k = -np.sum(np.multiply(vecC, pt_samples[1, :]))
            plane_eq = [vecC[0], vecC[1], vecC[2], k]

            # Distance from a point to a plane
            # https://mathworld.wolfram.com/Point-PlaneDistance.html
            # Distance from a point to a plane
            dist_pt = (
                              plane_eq[0] * pts[:, 0] + plane_eq[1] * pts[:, 1] + plane_eq[2] * pts[:, 2] + plane_eq[3]
                      ) / np.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

            # Select indexes where distance is smaller than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            if len(pt_id_inliers) > len(best_inliers) and len(pt_id_inliers) > minPoints:
                best_eq = plane_eq
                best_inliers = pt_id_inliers

            self.inliers = best_inliers
            self.equation = best_eq

        inliers_plane = pts[self.inliers, :]
        inliers_indices = self.inliers
        if len(inliers_plane) > 0:
            clusters = dbscan.fit_predict(inliers_plane)
            unique_clusters = set(clusters)

            biggest_cluster_size = -1
            biggest_cluster_points = None
            biggest_cluster_indices = None

            for cluster_label in unique_clusters:
                if cluster_label == -1:
                    continue  # Skip noise points labeled as -1
                
                cluster_indices = inliers_indices[clusters == cluster_label]
                cluster_points = inliers_plane[clusters == cluster_label]

                if len(cluster_points) > biggest_cluster_size:
                    biggest_cluster_size = len(cluster_points)
                    biggest_cluster_points = cluster_points
                    biggest_cluster_indices = cluster_indices

            # Now biggest_cluster_indices contains the indices of the points in the largest cluster
            inliers_plane_indices = biggest_cluster_indices
        else:
            inliers_plane_indices = []
        
        self.inliers_indices = inliers_plane_indices
        
        return self.equation, self.inliers_indices
