import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import morphology
import networkx as nx
import re
from PIL import Image
from DIP import process_fundus_image, skeleton_to_graph
from OD_Detection import detect_optic_disc
from Turtle_DFS import find_best_starting_vessel

# Define the RetinalVesselAnalyzer class to analyse the retinal images
class RetinalVesselAnalyzer:
    # Constructor to initialize the class
    def __init__(self, original_image_path, segmented_image_path=None):
        self.original_image_path = original_image_path  # Path to the original image
        self.segmented_image_path = segmented_image_path  # Path to the segmented image
        self.image_id = os.path.splitext(os.path.basename(original_image_path))[0]  # Extract filename and remove the file extension

        # Initialize containers for results
        self.optic_disc_center = None
        self.fundus_boundary = None
        self.fundus_contour_points = None
        self.polar_endpoints = []
        self.polar_bifurcations = []
        self.polar_vessel_paths = []
        self.turning_angles = []
        self.branch_lengths = []
        self.vessel_graph = None
        self.skeleton_image = None
        self.binary_image = None
        self.clean_image = None

        # Load original image
        self.original_image = cv2.imread(self.original_image_path)
        if self.original_image is None:
            raise ValueError(f"Could not read image at {self.original_image_path}")

        # Load segmentation mask if available
        self.segmented_image = None
        if self.segmented_image_path and os.path.exists(self.segmented_image_path):
            pil_image = Image.open(self.segmented_image_path)
            # Convert PIL image to numpy array
            self.segmented_image = np.array(pil_image.convert('L'))  # Convert to grayscale
            # Threshold to ensure binary (0 or 255) values
            _, self.segmented_image = cv2.threshold(self.segmented_image, 127, 255, cv2.THRESH_BINARY)

        print(f"Loaded image {self.image_id} - " +
              f"Original: {os.path.basename(original_image_path)}" +
              (
                  f", Segmented: {os.path.basename(segmented_image_path)}" if segmented_image_path else ", No segmentation mask"))


    # Method to detect the fundus boundary
    def detect_fundus_boundary(self):
        # Convert to grayscale if not already
        if len(self.original_image.shape) == 3:
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.original_image.copy()

        # Get image dimensions
        h, w = gray.shape

        # Apply Median blur to remove salt and pepper noise and preserve edges
        # It is a non-linear filter that replaces each pixel with the median of the pixels in a neighborhood
        blurred = cv2.medianBlur(gray, 11)

        # Calculates the mean (Average Brightness) and standard deviation (Spread of the brightness values) of the pixel intensities
        mean_intensity = np.mean(blurred)
        std_intensity = np.std(blurred)

        # Create a binary mask with pixels (Fundus) significantly different from background
        # Threshold will help remove anything outside the bell curve of the intensity distribution
        # Greater than threshold -> 255 or else 0
        threshold = mean_intensity - std_intensity
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

        # Clean up the binary mask
        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)  # Removes the small white regions
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # Fills the small holes in the foreground

        # Find contours in the binary mask
        # Contour is a boundary of the connected white pixels
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)

            # The area of the largest contour is compared to the area of the image
            # This helps to determine if the contour is the fundus or the entire image
            contour_area = cv2.contourArea(largest_contour)
            image_area = h * w

            # Covers the circle with minimum area
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            self.fundus_boundary = (int(x), int(y), int(radius))

            self.fundus_contour_points = largest_contour

            # Check if the contour covers 90% of the image area
            if contour_area > 0.9 * image_area:
                # This calculates the distance of each foreground pixel to the nearest background pixel
                dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

                # Find the point with maximum distance (Center of largest inscribed circle and its radius)
                _, max_dist, _, max_loc = cv2.minMaxLoc(dist_transform)  # The maximum distance is the radius of the largest inscribed circle
                # The center of the largest inscribed circle is the point with maximum distance
                center_x, center_y = max_loc

                # Create a circular mask with the center and radius
                self.fundus_boundary = (int(center_x), int(center_y), int(max_dist))
                print(
                    f"Image {self.image_id}: Fundus boundary detected at {self.fundus_boundary}")
                return self.fundus_boundary
            else:
                # Fit a circle to the contour
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)

                self.fundus_boundary = (int(x), int(y), int(radius))
                print(f"Image {self.image_id}: Fundus boundary detected at {self.fundus_boundary}")
                return self.fundus_boundary
        return self.fundus_boundary

    def detect_optic_disc(self):
        # Detect optic Disc center
        self.optic_disc_center = detect_optic_disc(self.original_image_path)
        print(f"Optic disc detected at {self.optic_disc_center}")

        # Process vessels if we haven't already
        if not hasattr(self, 'binary_image') or self.binary_image is None:
            self.process_vessels()

        # Find the best starting vessel point to refine optic disc location
        optic_radius = 55  # Standard optic disc radius
        if self.binary_image is not None:
            best_point = find_best_starting_vessel(
                self.binary_image,
                self.optic_disc_center,
                optic_radius
            )

            # Convert (row, col) to (x, y) format
            refined_center = (best_point[1], best_point[0])

            # Only update if the refinement is within a reasonable distance of original
            od_x, od_y = self.optic_disc_center
            new_x, new_y = refined_center
            shift_distance = np.sqrt((od_x - new_x) ** 2 + (od_y - new_y) ** 2)

            if shift_distance < 200:  # Limit drastic shifts in position
                self.optic_disc_center = refined_center
            else:
                pass

        return self.optic_disc_center

    def to_polar(self, point):
        # Ensure optic disc is detected
        if self.optic_disc_center is None:
            self.detect_optic_disc()

        # Convert from (row, col) to (x, y) for standard polar calculations
        y, x = point

        # The optic disc center is in (x, y) format
        od_x, od_y = self.optic_disc_center

        # Calculate polar coordinates
        dx = x - od_x
        dy = y - od_y

        # Radius: Distance from optic disc
        radius = np.sqrt(dx ** 2 + dy ** 2)

        # Angle: 0 degrees is to the right, increases counter-clockwise
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        # Normalize angle to [0, 360)
        angle_deg = (angle_deg + 360) % 360

        return radius, angle_deg

    def from_polar(self, radius, angle_deg):
        # Ensure optic disc is detected
        if self.optic_disc_center is None:
            self.detect_optic_disc()

        # Convert angle to radians
        angle_rad = np.radians(angle_deg)

        # Calculate Cartesian coordinates
        od_x, od_y = self.optic_disc_center
        x = od_x + radius * np.cos(angle_rad)
        y = od_y + radius * np.sin(angle_rad)

        return int(y), int(x)

    def process_vessels(self):
        # Check if we already have a segmentation mask loaded
        if self.segmented_image is not None:
            # If we have a loaded segmentation mask, use it directly
            segmented = self.segmented_image

            # Threshold if necessary (in case it's not fully binary)
            _, binary_image = cv2.threshold(segmented, 127, 1, cv2.THRESH_BINARY)
            clean_image = binary_image.copy()

            # Skeletonize
            skeleton_image = morphology.skeletonize(binary_image > 0)
            skeleton_image = skeleton_image.astype(np.uint8)

            print(f"  Processed vessels using provided segmentation mask")

        elif self.segmented_image_path and os.path.exists(self.segmented_image_path):
            # If we have a segmentation mask path, read it
            segmented_image = cv2.imread(self.segmented_image_path, cv2.IMREAD_GRAYSCALE)

            if segmented_image is not None:
                # Threshold
                _, binary_image = cv2.threshold(segmented_image, 127, 1, cv2.THRESH_BINARY)
                clean_image = binary_image.copy()

                # Skeletonize
                skeleton_image = morphology.skeletonize(binary_image > 0)
                skeleton_image = skeleton_image.astype(np.uint8)

                print(f"  Processed vessels using segmentation mask from {self.segmented_image_path}")
            else:
                # Fallback to processing from original
                print(f"  Could not read segmentation mask, falling back to vessel detection")
                results = process_fundus_image(self.original_image_path)
                # Use the results from the fundus image processing
                binary_image = results.get('binary_image')
                clean_image = results.get('clean_image', binary_image)
                skeleton_image = results.get('skeleton_image', clean_image)
        else:
            # Process original image if no segmentation mask
            print(f"  No segmentation mask provided, detecting vessels from original image")
            results = process_fundus_image(self.original_image_path)
            binary_image = results.get('binary_image')
            clean_image = results.get('clean_image', binary_image)
            skeleton_image = results.get('skeleton_image', clean_image)

        # Store the results
        self.clean_image = clean_image
        self.binary_image = binary_image
        self.skeleton_image = skeleton_image

        # Create graph representation
        self.vessel_graph = skeleton_to_graph(skeleton_image)
        print(f"  Created vessel graph with {len(self.vessel_graph.nodes)} nodes")

        return self.vessel_graph, skeleton_image

    def extract_endpoints_and_bifurcations(self):
        if not hasattr(self, 'vessel_graph') or self.vessel_graph is None:
            self.process_vessels()

        if self.optic_disc_center is None:
            self.detect_optic_disc()

        # Initialize lists to store results
        self.polar_endpoints = []
        self.polar_bifurcations = []

        # Count for debugging
        endpoint_count = 0
        bifurcation_count = 0

        # Iterate through each node in the graph
        for node in self.vessel_graph.nodes:
            # Get degree of the node (number of connections)
            degree = self.vessel_graph.degree(node)

            # Endpoints have exactly one connection
            if degree == 1:
                polar_coords = self.to_polar(node)
                # Store both polar and original Cartesian coordinates
                self.polar_endpoints.append((*polar_coords, node[1], node[0]))  # (radius, angle, x, y)
                endpoint_count += 1

            # Bifurcation points have three or more connections
            elif degree >= 3:
                polar_coords = self.to_polar(node)
                # Store both polar and original Cartesian coordinates
                self.polar_bifurcations.append((*polar_coords, node[1], node[0]))  # (radius, angle, x, y)
                bifurcation_count += 1

        print(f"  Extracted {endpoint_count} endpoints and {bifurcation_count} bifurcations")
        return self.polar_endpoints, self.polar_bifurcations

    def find_vessel_paths_polar(self):
        if not hasattr(self, 'vessel_graph') or self.vessel_graph is None:
            self.process_vessels()

        if not self.polar_endpoints:
            self.extract_endpoints_and_bifurcations()

        # Convert optic disc center to row, col format
        od_x, od_y = self.optic_disc_center
        od_node = (od_y, od_x)  # Convert to (row, col) format

        # Find the closest node in the graph to the optic disc center
        if od_node not in self.vessel_graph.nodes:
            od_node = min(
                self.vessel_graph.nodes,
                key=lambda n: (n[0] - od_y) ** 2 + (n[1] - od_x) ** 2  # Minimizing the Euclidean distance squared
            )
            print(f"  Using closest vessel node to optic disc: {od_node}")

        # For each endpoint, Find the shortest path from the optic disc
        self.polar_vessel_paths = []
        paths_found = 0
        paths_failed = 0

        for endpoint_data in self.polar_endpoints:
            # Extract Cartesian coordinates from the endpoint data
            _, _, x, y = endpoint_data
            endpoint_node = (y, x)  # Convert back to (row, col)

            try:
                # Find the shortest path from optic disc to endpoint
                path = nx.shortest_path(self.vessel_graph, od_node, endpoint_node)

                # Convert path to polar coordinates
                polar_path = []
                for node in path:
                    radius, angle = self.to_polar(node)
                    polar_path.append((radius, angle, node[1], node[0]))  # (radius, angle, x, y)

                self.polar_vessel_paths.append(polar_path)
                paths_found += 1

            except nx.NetworkXNoPath:
                # No path found between optic disc and this endpoint
                paths_failed += 1
                continue

        print(f"  Found {paths_found} vessel paths from optic disc, {paths_failed} paths failed")
        return self.polar_vessel_paths

    def analyze_turning_angles(self):
        if not self.polar_vessel_paths:
            self.find_vessel_paths_polar()

        self.turning_angles = []

        # Track counts for debugging
        total_angles_checked = 0
        significant_angles_found = 0

        # For each path
        for path_index, path in enumerate(self.polar_vessel_paths):
            if len(path) < 3:  # Need at least 3 points to detect turning
                continue

            # Analyze segments for turning angles
            path_angles = []
            path_segments_checked = 0

            # First, extract all turning angles for this path, even non-significant ones
            for i in range(1, len(path) - 1):
                path_segments_checked += 1
                total_angles_checked += 1

                # Get three consecutive points (Polar Coordinate Format)
                prev_point = path[i - 1][:2]  # (radius, angle)
                curr_point = path[i][:2]  # (radius, angle)
                next_point = path[i + 1][:2]  # (radius, angle)

                # Get the angles in degrees
                prev_angle = prev_point[1]
                curr_angle = curr_point[1]
                next_angle = next_point[1]

                # Calculate the turning angle (handle wrap-around at 360 degrees)
                # First segment angle = direction from prev to curr
                first_segment_angle = (curr_angle - prev_angle + 360) % 360
                if first_segment_angle > 180:
                    first_segment_angle -= 360  # Convert to [-180, 180] range

                # Second segment angle = direction from curr to next
                second_segment_angle = (next_angle - curr_angle + 360) % 360
                if second_segment_angle > 180:
                    second_segment_angle -= 360  # Convert to [-180, 180] range

                # Turning angle = change in direction between segments
                turning_angle = second_segment_angle - first_segment_angle

                # Normalize to [-180, 180]
                if turning_angle > 180:
                    turning_angle -= 360
                elif turning_angle < -180:
                    turning_angle += 360

                # Store all turning angles for this path
                path_angles.append((
                    i,
                    curr_point[0],  # radius
                    curr_point[1],  # angle
                    turning_angle
                ))

                if abs(turning_angle) > 0:
                    # Store path index, segment index, radius, angle, turning angle
                    self.turning_angles.append((
                        path_index,
                        i,
                        curr_point[0],  # radius
                        curr_point[1],  # angle
                        turning_angle
                    ))
                    significant_angles_found += 1
        return self.turning_angles

    def analyze_branch_lengths(self):
        if not self.polar_vessel_paths:
            self.find_vessel_paths_polar()

        self.branch_lengths = []

        for path_idx, path in enumerate(self.polar_vessel_paths):
            if len(path) < 2:  # Need at least 2 points to calculate length
                continue

            # Calculate total path length
            total_length = 0
            segment_lengths = []

            for i in range(1, len(path)):
                # Get consecutive points in Cartesian coordinates
                prev_x, prev_y = path[i - 1][2:4]
                curr_x, curr_y = path[i][2:4]

                # Calculate Euclidean distance and sum the lengths
                segment_length = np.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
                total_length += segment_length
                segment_lengths.append(segment_length)

            # Store path index, start radius, end radius, total length
            self.branch_lengths.append((
                path_idx,  # path index
                path[0][0],  # start radius
                path[-1][0],  # end radius
                total_length  # total length
            ))

        print(f"  Analyzed {len(self.branch_lengths)} branch lengths")
        return self.branch_lengths

    def analyze_bifurcation_angles(self):
        if not hasattr(self, 'vessel_graph') or self.vessel_graph is None:
            self.process_vessels()

        if not self.polar_bifurcations:
            self.extract_endpoints_and_bifurcations()

        bifurcation_angles = []
        bifurcation_count = 0

        for bif_data in self.polar_bifurcations:
            _, _, x, y = bif_data
            bif_node = (y, x)  # Convert to (row, col)

            # Get neighbors
            neighbors = list(self.vessel_graph.neighbors(bif_node))

            # Need at least 3 neighbors for a bifurcation
            if len(neighbors) >= 3:
                # Convert neighbors to polar coordinates relative to bifurcation point
                neighbor_angles = []

                for neighbor in neighbors:
                    # Calculate vector from bifurcation to neighbor
                    ny, nx = neighbor
                    dx = nx - x
                    dy = ny - y

                    # Calculate angle
                    angle_rad = np.arctan2(dy, dx)
                    angle_deg = np.degrees(angle_rad)

                    # Normalize to [0, 360)
                    angle_deg = (angle_deg + 360) % 360

                    neighbor_angles.append(angle_deg)

                # Sort angles
                neighbor_angles.sort()

                # Calculate angles between adjacent branches
                branch_angles = []
                for i in range(len(neighbor_angles)):
                    next_i = (i + 1) % len(neighbor_angles)
                    angle_diff = (neighbor_angles[next_i] - neighbor_angles[i] + 360) % 360
                    branch_angles.append(angle_diff)

                # Store bifurcation data: radius, angle, branch angles
                bifurcation_angles.append((
                    bif_data[0],  # radius
                    bif_data[1],  # angle
                    branch_angles  # list of angles between branches
                ))
                bifurcation_count += 1

        print(f"  Analyzed angles for {bifurcation_count} bifurcation points")
        return bifurcation_angles

    def analyze_all(self):
        # Detect fundus boundary
        self.detect_fundus_boundary()

        # Detect optic disc
        self.detect_optic_disc()

        # Process vessel image
        self.process_vessels()

        # Extract features in polar coordinates
        self.extract_endpoints_and_bifurcations()
        self.find_vessel_paths_polar()
        self.analyze_turning_angles()
        self.analyze_branch_lengths()
        bifurcation_angles = self.analyze_bifurcation_angles()

        # Calculate additional statistics
        results = {
            'image_id': self.image_id,
            'original_image_path': self.original_image_path,
            'segmented_image_path': self.segmented_image_path,
            'fundus_boundary': self.fundus_boundary,
            'optic_disc_center': self.optic_disc_center,
            'num_endpoints': len(self.polar_endpoints),
            'num_bifurcations': len(self.polar_bifurcations),
            'num_paths': len(self.polar_vessel_paths),
            'num_turning_points': len(self.turning_angles),
            'polar_endpoints': self.polar_endpoints,
            'polar_bifurcations': self.polar_bifurcations,
            'turning_angles': self.turning_angles,
            'branch_lengths': self.branch_lengths,
            'bifurcation_angles': bifurcation_angles
        }

        # Compute statistics on turning angles if available
        if self.turning_angles:
            angles = [angle for _, _, _, _, angle in self.turning_angles]
            results['mean_turning_angle'] = np.mean(angles)
            results['median_turning_angle'] = np.median(angles)
            results['std_turning_angle'] = np.std(angles)
            results['min_turning_angle'] = np.min(angles)
            results['max_turning_angle'] = np.max(angles)

        # Compute statistics on branch lengths
        if self.branch_lengths:
            lengths = [length for _, _, _, length in self.branch_lengths]
            results['mean_branch_length'] = np.mean(lengths)
            results['median_branch_length'] = np.median(lengths)
            results['std_branch_length'] = np.std(lengths)
            results['min_branch_length'] = np.min(lengths)
            results['max_branch_length'] = np.max(lengths)

        return results

    def plot_polar_features(self, show_plots=True):
        fig = plt.figure(figsize=(16, 10))

        # Plot 1: Original image with fundus boundary and optic disc
        ax1 = fig.add_subplot(221)

        if len(self.original_image.shape) == 3:
            ax1.imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        else:
            ax1.imshow(self.original_image, cmap='gray')

        # Draw fundus boundary
        if self.fundus_boundary:
            fundus_x, fundus_y, fundus_radius = self.fundus_boundary
            fundus_circle = plt.Circle((fundus_x, fundus_y), fundus_radius,
                                       color='red', fill=False, linewidth=2)
            ax1.add_patch(fundus_circle)

        # Draw optic disc
        if self.optic_disc_center:
            od_x, od_y = self.optic_disc_center
            od_circle = plt.Circle((od_x, od_y), 55,
                                   color='green', fill=False, linewidth=2)
            ax1.add_patch(od_circle)
            ax1.plot(od_x, od_y, 'go', markersize=10)

        ax1.set_title(f'Original Image: {self.image_id}')

        # Plot 2: Segmentation mask with endpoints and bifurcations
        ax2 = fig.add_subplot(222)

        # Display the binary image or segmentation if available
        if self.segmented_image is not None:
            ax2.imshow(self.segmented_image, cmap='gray')
        elif self.clean_image is not None:
            ax2.imshow(self.clean_image, cmap='gray')
        else:
            ax2.text(0.5, 0.5, 'No segmentation available',
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes)

        # Plot endpoints and bifurcations on the segmentation
        if self.polar_endpoints:
            endpoints_x = [x for _, _, x, _ in self.polar_endpoints]
            endpoints_y = [y for _, _, _, y in self.polar_endpoints]
            ax2.scatter(endpoints_x, endpoints_y, color='blue', s=15, marker='o', label='Endpoints')

        if self.polar_bifurcations:
            bifurcations_x = [x for _, _, x, _ in self.polar_bifurcations]
            bifurcations_y = [y for _, _, _, y in self.polar_bifurcations]
            ax2.scatter(bifurcations_x, bifurcations_y, color='red', s=15, marker='x', label='Bifurcations')

        # Draw optic disc on segmentation as well
        if self.optic_disc_center:
            od_x, od_y = self.optic_disc_center
            od_circle = plt.Circle((od_x, od_y), 55,
                                   color='green', fill=False, linewidth=2)
            ax2.add_patch(od_circle)

        ax2.set_title('Segmentation with Features')
        ax2.legend()

        # Plot 3: Turning Angles
        ax3 = fig.add_subplot(223)

        if self.turning_angles:
            angles = [angle for _, _, _, _, angle in self.turning_angles]
            ax3.hist(angles, bins=36, alpha=0.7, color='purple', label='Turning Angles')
            ax3.set_xlabel('Angle (degrees)')
            ax3.set_ylabel('Frequency')

            # Add turning angle statistics
            mean_angle = np.mean(angles)
            median_angle = np.median(angles)
            std_angle = np.std(angles)

            stats_text = f"Angles:\nMean: {mean_angle:.1f}°\nMedian: {median_angle:.1f}°\nStd: {std_angle:.1f}°"
            ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax3.set_title('Distribution of Turning Angles')
        ax3.legend()

        # Plot 4: Branch Lengths
        ax4 = fig.add_subplot(224)

        if self.branch_lengths:
            lengths = [length for _, _, _, length in self.branch_lengths]
            # Normalize lengths to [0-180] range for better visualization
            max_length = max(lengths) if lengths else 1
            norm_lengths = [l * 180 / max_length for l in lengths]
            ax4.hist(norm_lengths, bins=36, alpha=0.7, color='green', label='Branch Lengths (normalized)')
            ax4.set_xlabel('Normalized Length')
            ax4.set_ylabel('Frequency')

            # Add branch length statistics
            mean_length = np.mean(lengths)
            median_length = np.median(lengths)
            std_length = np.std(lengths)

            stats_text = f"Lengths:\nMean: {mean_length:.1f}\nMedian: {median_length:.1f}\nStd: {std_length:.1f}"
            ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax4.set_title('Distribution of Branch Lengths')
        ax4.legend()

        plt.tight_layout()

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        return fig


def find_matching_segmentation(orig_path, seg_dirs):
    if not seg_dirs:
        return None

    # For STARE dataset
    if 'stare-images' in orig_path:
        # Get the filename
        orig_filename = os.path.basename(orig_path)
        base_name = os.path.splitext(orig_filename)[0]

        # Find the labels-vk_STARE directory
        for seg_dir in seg_dirs:
            if 'labels-vk_STARE' in seg_dir:
                # Create the path to the segmentation mask
                seg_path = os.path.join(seg_dir, f"{base_name}.vk.ppm")
                if os.path.exists(seg_path):
                    return seg_path

    # For training dataset
    elif 'training/images' in orig_path:
        # Get the filename
        orig_filename = os.path.basename(orig_path)

        # Extract the number part
        match = re.search(r'(\d+)_training', orig_filename)
        if match:
            number = match.group(1)

            # Find the 1st_manual directory
            for seg_dir in seg_dirs:
                if '1st_manual' in seg_dir:
                    # Create the path to the segmentation mask
                    seg_path = os.path.join(seg_dir, f"{number}_manual1.gif")
                    if os.path.exists(seg_path):
                        return seg_path

    return None

def process_image_directories(orig_dirs, seg_dirs=None, sample_limit=None):
    # Get all original image paths
    orig_images = []
    for directory in orig_dirs:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.ppm', '.gif')):
                    orig_images.append(os.path.join(directory, filename))
        else:
            print(f"Warning: Directory {directory} does not exist")

    # Limit samples if specified
    if sample_limit and sample_limit > 0:
        orig_images = orig_images[:sample_limit]

    print(f"Found {len(orig_images)} original images to process")

    # Process each image
    all_results = []

    for i, orig_path in enumerate(orig_images):
        print(f"\nProcessing image {i + 1}/{len(orig_images)}: {orig_path}")

        # Find corresponding segmented image if available
        seg_path = find_matching_segmentation(orig_path, seg_dirs)

        if seg_path:
            print(f"Found matching segmentation: {seg_path}")
        else:
            print(f"No matching segmentation found")

        try:
            # Analyze the image
            analyzer = RetinalVesselAnalyzer(orig_path, seg_path)
            results = analyzer.analyze_all()

            # Show visualization
            analyzer.plot_polar_features(show_plots=True)

            # Add to results
            all_results.append(results)

        except Exception as e:
            print(f"Error processing {orig_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate results
    aggregated = aggregate_results(all_results)

    return aggregated, all_results

def aggregate_results(results_list):
    if not results_list:
        return {}

    # Extract basic statistics
    num_images = len(results_list)

    # Initialize all aggregation lists
    all_fields = {
        'fundus_radius': [],
        'endpoints': [],
        'num_endpoints': [],
        'bifurcations': [],
        'num_bifurcations': [],
        'turning_angles': [],
        'branch_lengths': []
    }

    # Collect data from all images
    for r in results_list:
        if 'fundus_boundary' in r and r['fundus_boundary']:
            all_fields['fundus_radius'].append(r['fundus_boundary'][2])  # Radius is the third element

        if 'polar_endpoints' in r:
            all_fields['endpoints'].extend(r['polar_endpoints'])
            all_fields['num_endpoints'].append(len(r['polar_endpoints']))

        if 'polar_bifurcations' in r:
            all_fields['bifurcations'].extend(r['polar_bifurcations'])
            all_fields['num_bifurcations'].append(len(r['polar_bifurcations']))

        if 'turning_angles' in r and r['turning_angles']:
            all_fields['turning_angles'].extend([angle for _, _, _, _, angle in r['turning_angles']])

        if 'branch_lengths' in r and r['branch_lengths']:
            all_fields['branch_lengths'].extend([length for _, _, _, length in r['branch_lengths']])

    # Compute aggregated statistics
    aggregated = {
        'num_images': num_images,
        'mean_endpoints_per_image': np.mean(all_fields['num_endpoints']) if all_fields['num_endpoints'] else 0,
        'std_endpoints_per_image': np.std(all_fields['num_endpoints']) if all_fields['num_endpoints'] else 0,
        'mean_bifurcations_per_image': np.mean(all_fields['num_bifurcations']) if all_fields[
            'num_bifurcations'] else 0,
        'std_bifurcations_per_image': np.std(all_fields['num_bifurcations']) if all_fields[
            'num_bifurcations'] else 0,
        'mean_fundus_radius': np.mean(all_fields['fundus_radius']) if all_fields['fundus_radius'] else 0,
        'std_fundus_radius': np.std(all_fields['fundus_radius']) if all_fields['fundus_radius'] else 0,
    }

    # Add turning angle statistics if available
    if all_fields['turning_angles']:
        aggregated.update({
            'total_turning_points': len(all_fields['turning_angles']),
            'mean_turning_angle': np.mean(all_fields['turning_angles']),
            'median_turning_angle': np.median(all_fields['turning_angles']),
            'std_turning_angle': np.std(all_fields['turning_angles']),
            'min_turning_angle': np.min(all_fields['turning_angles']),
            'max_turning_angle': np.max(all_fields['turning_angles'])
        })

    # Add branch length statistics if available
    if all_fields['branch_lengths']:
        aggregated.update({
            'total_branches': len(all_fields['branch_lengths']),
            'mean_branch_length': np.mean(all_fields['branch_lengths']),
            'median_branch_length': np.median(all_fields['branch_lengths']),
            'std_branch_length': np.std(all_fields['branch_lengths']),
            'min_branch_length': np.min(all_fields['branch_lengths']),
            'max_branch_length': np.max(all_fields['branch_lengths'])
        })

    # Store all endpoints and bifurcations for synthetic model generation
    aggregated['all_endpoints'] = all_fields['endpoints']
    aggregated['all_bifurcations'] = all_fields['bifurcations']

    return aggregated

def plot_distributions(aggregated):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram for turning angles
    if 'total_turning_points' in aggregated and aggregated['total_turning_points'] > 0:
        turning_angles = np.random.normal(
            loc=aggregated['mean_turning_angle'],
            scale=aggregated['std_turning_angle'],
            size=aggregated['total_turning_points']
        )
        axs[0].hist(turning_angles, bins=30, color='purple', alpha=0.7)
        axs[0].set_title('Distribution of Turning Angles')
        axs[0].set_xlabel('Angle (degrees)')
        axs[0].set_ylabel('Frequency')

    # Histogram for branch lengths
    if 'total_branches' in aggregated and aggregated['total_branches'] > 0:
        branch_lengths = np.random.normal(
            loc=aggregated['mean_branch_length'],
            scale=aggregated['std_branch_length'],
            size=aggregated['total_branches']
        )
        axs[1].hist(branch_lengths, bins=30, color='green', alpha=0.7)
        axs[1].set_title('Distribution of Branch Lengths')
        axs[1].set_xlabel('Length (pixels)')
        axs[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Define directories containing images
    original_dirs = ['stare-images', 'training/images']
    segmented_dirs = ['labels-vk_STARE', 'training/1st_manual']

    # Process a small sample for testing (set to None to process all)
    sample_limit = 2  # Process just 2 images for demonstration

    # Run the analysis
    aggregated_results, all_results = process_image_directories(
        original_dirs, segmented_dirs, sample_limit
    )

    # Plot distributions of turning angles and branch lengths
    plot_distributions(aggregated_results)

    # Display aggregated results
    print("\nAggregated Results:")
    for key, value in aggregated_results.items():
        if isinstance(value, (list, tuple)):
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}")