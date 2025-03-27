import cv2
import numpy as np
import matplotlib.pyplot as plt
import turtle
import networkx as nx
from DIP import (process_fundus_image, skeleton_to_graph)
from OD_Detection import detect_optic_disc


def find_best_starting_vessel(binary_image, optic_disc_location, radius):
    """
    Find the thickest vessel on either the left or right side of the optic disc.

    Args:
        binary_image: Binary image of the vessel segmentation
        optic_disc_location: (x, y) location of the optic disc center
        radius: Radius to search around the optic disc

    Returns:
        Coordinates of the best starting point (row, col)
    """
    # Extract dimensions and create coordinates
    h, w = binary_image.shape
    y_grid, x_grid = np.ogrid[:h, :w]

    # Convert optic_disc_location from (x, y) to (col, row)
    center_col, center_row = optic_disc_location

    # Create a circular mask for inside the optic disc
    dist_from_center = np.sqrt((x_grid - center_col) ** 2 + (y_grid - center_row) ** 2)
    inside_disc_mask = dist_from_center < radius  # True for pixels inside the circle
    left_side_mask = x_grid < center_col  # True for pixels on the left side
    right_side_mask = x_grid > center_col  # True for pixels on the right side
    left_side_disc_mask = inside_disc_mask & left_side_mask  # True for pixels inside disc on left side
    right_side_disc_mask = inside_disc_mask & right_side_mask  # True for pixels inside disc on right side

    # Apply masks to the binary image to get vessels on each side
    left_vessels = binary_image & left_side_disc_mask
    right_vessels = binary_image & right_side_disc_mask

    # Check each side for the thickest vessel
    best_score = -1
    best_point = None

    # Function to evaluate each side
    def evaluate_side(vessel_mask, side_name):
        nonlocal best_score, best_point  # Nonlocal allows modifying outer scope variables -> best_score, best_point

        # If no vessels on this side, return immediately
        if np.sum(vessel_mask) == 0:
            return

        # Get all the vessel pixel coordinates
        vessel_points = np.where(vessel_mask > 0)

        for i in range(len(vessel_points[0])):
            row, col = vessel_points[0][i], vessel_points[1][i]  # Get the row and column of each vessel pixel

            # Takes a 5x5 window centered at row,col in the binary image
            r_min, r_max = max(0, row - 2), min(h - 1, row + 3)
            c_min, c_max = max(0, col - 2), min(w - 1, col + 3)
            local_window = binary_image[r_min:r_max, c_min:c_max]
            # Simple thickness estimation using local sum in a 5x5 window
            thickness_score = np.sum(local_window)

            # Distance to Initial optic disc center
            distance = np.sqrt((row - center_row) ** 2 + (col - center_col) ** 2)

            # Prefer thicker vessels that are closer to center
            # We want higher thickness and lower distance
            combined_score = thickness_score - (distance * 0.1)  # Penalizes further distance points

            if combined_score > best_score:
                best_score = combined_score
                best_point = (int(row), int(col))

    # Evaluate both sides
    evaluate_side(left_vessels, "left")
    evaluate_side(right_vessels, "right")

    # If no suitable point found, use optic disc center
    if best_point is None:
        print("No suitable vessel starting point found. Using optic disc center.")
        best_point = (int(center_row), int(center_col))
    else:
        print(f"Selected vessel starting point at: {best_point} with score: {best_score}")

    return best_point


def find_multiple_vessel_points_bfs(binary_image, main_starting_point, optic_disc_location, radius, min_thickness):
    """
    Use BFS to find multiple vessel points inside the optic disc circle.

    Args:
        binary_image: Binary image with vessels
        main_starting_point: The best starting point found earlier (row, col)
        optic_disc_location: (x, y) coordinates of the optic disc center
        radius: Radius of the optic disc
        min_thickness: Minimum vessel thickness to consider

    Returns:
        List of vessel points inside the optic disc
    """
    # Extract dimensions and create coordinates
    h, w = binary_image.shape

    # Convert optic_disc_location from (x, y) to (col, row)
    center_col, center_row = optic_disc_location

    # Create a circular mask for inside the optic disc
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_grid - center_col) ** 2 + (y_grid - center_row) ** 2)
    inside_disc_mask = dist_from_center < radius

    # Get vessel pixels inside optic disc
    vessels_in_disc = binary_image & inside_disc_mask

    # List to store vessel points with their thickness scores
    vessel_points = []

    # Get coordinates of all vessel pixels in the circle
    points = np.where(vessels_in_disc > 0)

    # For each vessel pixel, calculate thickness score
    for i in range(len(points[0])):
        row, col = points[0][i], points[1][i]

        # Simple thickness estimation using local sum in a 5x5 window
        r_min, r_max = max(0, row - 2), min(h - 1, row + 3)
        c_min, c_max = max(0, col - 2), min(w - 1, col + 3)
        local_window = binary_image[r_min:r_max, c_min:c_max]
        thickness_score = np.sum(local_window)

        # Only consider points with thickness above threshold
        if thickness_score >= min_thickness:
            # Calculate distance to optic disc center
            distance = np.sqrt((row - center_row) ** 2 + (col - center_col) ** 2)

            # Calculate boundary proximity to favor points near the disc boundary
            boundary_proximity = abs(distance - radius)

            # Combined score favoring thicker vessels that are near boundary
            combined_score = thickness_score - (boundary_proximity * 0.5)

            vessel_points.append((int(row), int(col), combined_score))

    # Sort by score (higher first)
    vessel_points.sort(key=lambda x: x[2], reverse=True)

    # Extract the top points, removing duplicates that are too close to each other
    min_distance_between_points = 10
    selected_points = []

    for point in vessel_points:
        row, col, _ = point

        # Check if this point is far enough from all previously selected points
        is_far_enough = True
        for selected_point in selected_points:
            s_row, s_col = selected_point
            distance = np.sqrt((row - s_row) ** 2 + (col - s_col) ** 2)
            if distance < min_distance_between_points:
                is_far_enough = False
                break

        if is_far_enough:
            selected_points.append((row, col))

            # Limit to a reasonable number of starting points
            if len(selected_points) >= 5:
                break

    # Always include the main starting point if it's not already selected
    if main_starting_point not in selected_points and len(selected_points) > 0:
        selected_points.insert(0, main_starting_point)
    elif len(selected_points) == 0:
        selected_points.append(main_starting_point)

    print(f"Selected {len(selected_points)} vessel starting points using BFS")
    return selected_points


def find_branch_endpoints(G, start_node):
    """
    Find all branch endpoints reachable from the start node.

    Args:
        G: NetworkX graph of the vessel skeleton
        start_node: Starting node (row, col) in the graph

    Returns:
        List of endpoint nodes (nodes with degree 1)
    """
    # Find the closest node in the graph if start_node isn't exactly in the graph
    if start_node not in G:
        start_node = min(G.nodes(), key=lambda n: (n[0] - start_node[0]) ** 2 + (n[1] - start_node[1]) ** 2)
        print(f"Adjusted starting point to closest node in graph: {start_node}")

    # Get all nodes connected to the start node
    reachable_nodes = nx.node_connected_component(G, start_node)
    subgraph = G.subgraph(reachable_nodes)

    # Find all endpoints (nodes with degree 1)
    endpoints = [node for node in subgraph.nodes() if subgraph.degree(node) == 1]

    # Remove start_node if it's also an endpoint
    if start_node in endpoints:
        endpoints.remove(start_node)

    print(f"Found {len(endpoints)} branch endpoints from starting point {start_node}")
    return endpoints


def find_path_to_endpoint(G, start_node, endpoint):
    """
    Find the path from start_node to endpoint in the graph.

    Args:
        G: NetworkX graph of the vessel skeleton
        start_node: Starting node (row, col)
        endpoint: Target endpoint node (row, col)

    Returns:
        List of nodes representing the path
    """
    try:
        path = nx.shortest_path(G, start_node, endpoint)
        return path
    except nx.NetworkXNoPath:
        return []


def draw_vessels_from_optic_disc(G, image_shape, starting_points, scale=1.0):
    """
    Draw vessel paths from optic disc starting points to all endpoints.

    Args:
        G: NetworkX graph of vessel skeleton
        image_shape: (height, width) of the image
        starting_points: List of starting points inside optic disc
        scale: Scale factor for visualization
    """
    h, w = image_shape

    # Convert starting_points to tuples to ensure they're hashable
    starting_points = [tuple(point) for point in starting_points]

    # Filter starting points that aren't in the graph
    valid_starting_points = []
    for point in starting_points:
        if point in G.nodes():
            valid_starting_points.append(point)
        else:
            # Try to find the closest node in the graph
            closest_point = min(
                G.nodes(),
                key=lambda n: (n[0] - point[0]) ** 2 + (n[1] - point[1]) ** 2
            )
            valid_starting_points.append(closest_point)
            print(f"Adjusted starting point from {point} to closest node: {closest_point}")

    # If no valid points, exit
    if not valid_starting_points:
        print("No valid starting points found in the graph.")
        return

    # Turtle setup
    screen = turtle.Screen()
    screen.setup(width=w * scale, height=h * scale)
    screen.setworldcoordinates(-w / 2 * scale, -h / 2 * scale, w / 2 * scale, h / 2 * scale)
    screen.title("Fundus Vessel Visualization - DFS from Optic Disc")

    # Create a turtle with customized appearance
    t = turtle.Turtle()
    t.hideturtle()
    t.speed(0)  # Fastest speed
    t.pensize(2)  # Thicker line for better visibility
    turtle.tracer(0, 0)  # Disable animation for faster drawing

    # Colors
    colors = ["red"]

    # Keep track of all visited paths to avoid redrawing
    drawn_paths = set()

    # Helper to move the turtle to a node
    def move_to_node(node):
        r, c = node
        x = (c - w / 2) * scale
        y = (h / 2 - r) * scale
        t.goto(x, y)

    # Process each starting point
    for i, start_point in enumerate(valid_starting_points):
        print(f"Drawing vessel tree from starting point {i + 1}: {start_point}")

        # Set color for this vessel tree
        t.pencolor(colors[i % len(colors)])

        # Find all branch endpoints from this starting point
        endpoints = find_branch_endpoints(G, start_point)

        # Draw paths to all endpoints using DFS approach
        for endpoint in endpoints:
            # Find path from start to endpoint
            path = find_path_to_endpoint(G, start_point, endpoint)

            if not path:
                continue

            # Skip if we've already drawn this path
            path_key = tuple(sorted([start_point, endpoint]))
            if path_key in drawn_paths:
                continue

            # Mark this path as drawn
            drawn_paths.add(path_key)

            # Draw this path from starting point to endpoint
            t.penup()
            move_to_node(start_point)
            t.pendown()

            for node in path[1:]:  # Skip the starting point which we've already moved to
                move_to_node(node)

        # Update display after each starting point
        turtle.update()

    print("Drawing complete. Close the turtle window to continue.")
    turtle.mainloop()


def visualize_optic_disc_and_starting_points(original_image, binary_image,
                                             optic_disc_location, optic_disc_radius,
                                             main_starting_point, all_starting_points):
    """
    Visualize the optic disc and starting points

    Args:
        original_image: Original fundus image
        binary_image: Binary image with vessels
        optic_disc_location: (x, y) coordinates of the optic disc center
        optic_disc_radius: Radius of the optic disc
        main_starting_point: The best starting point found (row, col)
        all_starting_points: List of all vessel starting points
    """
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot original image with optic disc circle and starting points
    if len(original_image.shape) == 3:
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(original_image, cmap='gray')

    # Draw circle around the optic disc
    optic_disc_circle = plt.Circle((optic_disc_location[0], optic_disc_location[1]),
                                   optic_disc_radius, color='r', fill=False, linewidth=2)
    axes[0].add_patch(optic_disc_circle)

    # Mark the optic disc center
    axes[0].scatter(optic_disc_location[0], optic_disc_location[1],
                    color='red', s=100, marker='x', label='Optic Disc Center')

    # Mark main starting point with a star
    axes[0].scatter(main_starting_point[1], main_starting_point[0],
                    color='yellow', s=150, marker='*', label='Main Starting Point')

    # Mark all other starting points with smaller dots
    for point in all_starting_points:
        if point != main_starting_point:  # Skip the main point which already has a star
            axes[0].scatter(point[1], point[0],
                            color='blue', s=20, marker='o')

    axes[0].set_title('Original Image')
    axes[0].legend()

    # Plot binary vessel mask with optic disc and starting points
    axes[1].imshow(binary_image, cmap='gray')

    # Draw circle around the optic disc
    optic_disc_circle2 = plt.Circle((optic_disc_location[0], optic_disc_location[1]),
                                    optic_disc_radius, color='r', fill=False, linewidth=2)
    axes[1].add_patch(optic_disc_circle2)

    # Mark the optic disc center
    axes[1].scatter(optic_disc_location[0], optic_disc_location[1],
                    color='red', s=100, marker='x', label='Optic Disc Center')

    # Mark main starting point with a star
    axes[1].scatter(main_starting_point[1], main_starting_point[0],
                    color='yellow', s=150, marker='*', label='Main Starting Point')

    # Mark all other starting points with smaller dots
    for point in all_starting_points:
        if point != main_starting_point:
            axes[1].scatter(point[1], point[0],
                            color='blue', s=20, marker='o')

    axes[1].set_title('Binary Vessel Mask')

    plt.tight_layout()
    plt.show()


def analyze_fundus_image(image_path):
    """
    Main function to analyze a fundus image

    Args:
        image_path: Path to the fundus image
    """
    print(f"Processing image: {image_path}")

    # Step 1: Process the fundus image for vessel segmentation
    print("Segmenting vessels...")
    results = process_fundus_image(image_path)

    binary_image = results.get('binary_image')
    skeleton_image = results.get('skeleton_image',
                                 binary_image)  # Use binary_image as fallback if skeleton not available

    original_image = cv2.imread(image_path)

    # Step 2: Detect the optic disc
    print("Detecting optic disc...")
    optic_disc_location = detect_optic_disc(image_path)

    if optic_disc_location is None:
        print("Failed to detect optic disc. Using image center as fallback.")
        h, w = original_image.shape[:2]
        optic_disc_location = (w // 2, h // 2)

    print(f"Optic disc detected at: {optic_disc_location}")

    # Fixed radius for optic disc
    optic_disc_radius = 55
    min_thickness = 1

    # Step 3: Find the best vessel starting point within the optic disc
    print("Finding optimal vessel starting point...")
    main_starting_point = find_best_starting_vessel(
        binary_image,
        optic_disc_location,
        radius=optic_disc_radius
    )

    # Step 4: Find multiple vessel starting points within the optic disc using BFS
    print("Finding multiple vessel starting points using BFS...")
    all_starting_points = find_multiple_vessel_points_bfs(
        binary_image,
        main_starting_point,
        optic_disc_location,
        radius=optic_disc_radius,
        min_thickness=min_thickness
    )

    # Step 5: Visualize optic disc and starting points
    visualize_optic_disc_and_starting_points(
        original_image,
        binary_image,
        optic_disc_location,
        optic_disc_radius,
        main_starting_point,
        all_starting_points
    )

    # Step 6: Create a graph representation of the skeleton
    print("Creating graph representation...")
    G = skeleton_to_graph(skeleton_image)

    # Step 7: Draw vasculature pattern from optic disc to endpoints
    print("Drawing vasculature pattern from optic disc to endpoints using DFS...")
    print("Close the visualization window when done.")
    draw_vessels_from_optic_disc(G, original_image.shape[:2], all_starting_points, scale=1.0)


if __name__ == "__main__":
    image_path = "test/images/05_test.tif"
    analyze_fundus_image(image_path)