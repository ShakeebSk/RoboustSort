# RoboustSort

RobustSort is a lightweight object tracking algorithm that matches bounding boxes using area, position, and aspect ratio similarity. It supports re-identification of lost objects, making it ideal for real-time video tracking in low-resource environments without deep learning or Kalman filters.

# Features
Multi-feature Matching: Uses box area, aspect ratio, and centroid position to match objects
Object Re-identification: Remembers lost objects and can re-identify them when they reappear
Trajectory Tracking: Optional trajectory visualization to track object movement paths
No Deep Learning for Tracking: Only uses geometric features for the actual tracking algorithm
Minimal Dependencies: Core algorithm only requires NumPy
Configurable Parameters: Easily tune the algorithm for your specific use case

# How It's Better Than SORT
RobustSort improves upon the original SORT (Simple Online Realtime Tracking) algorithm in several ways:

No Kalman Filter: Operates without complex motion prediction, making it more robust to erratic movements
Multi-feature Matching: Uses not just position but also area and aspect ratio for more robust tracking
Re-identification: Can remember and re-identify objects that temporarily disappear
Configurable Weights: Easily adjust which features matter more for your specific use case
Trajectory Visualization: Built-in support for visualizing object paths
