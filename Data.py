import turtle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import networkx as nx
from Turtle_DFS import (
    process_fundus_image,
    detect_optic_disc,
    find_best_starting_vessel,
    find_multiple_vessel_points_bfs,
    find_branch_endpoints,
    find_path_to_endpoint,
    skeleton_to_graph
)


def extract_vessel_angles_from_optic_disc(G, image_shape, starting_points, scale=1.0):
    """
    Draw vessel paths from optic disc and extract turning angles at each turning point.
    This records the relative angle change (how much the turtle turns) at each point.

    Args:
        G: NetworkX graph of vessel skeleton
        image_shape: (height, width) of the image
        starting_points: List of starting points inside optic disc
        scale: Scale factor for visualization

    Returns:
        List of turning angles at turning points
    """
    h, w = image_shape
    turning_angles = []  # To store the turning angles

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
        return turning_angles

    # Turtle setup
    screen = turtle.Screen()
    screen.setup(width=w * scale, height=h * scale)
    screen.setworldcoordinates(-w / 2 * scale, -h / 2 * scale, w / 2 * scale, h / 2 * scale)
    screen.title("Fundus Vessel Angle Extraction")

    # Create a turtle with customized appearance
    t = turtle.Turtle()
    t.hideturtle()
    t.speed(0)  # Fastest speed
    t.pensize(2)  # Thicker line for better visibility
    turtle.tracer(0, 0)  # Disable animation for faster drawing

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
        t.pencolor("red")

        # Find all branch endpoints from this starting point
        endpoints = find_branch_endpoints(G, start_point)

        # Draw paths to all endpoints using DFS approach
        for endpoint in endpoints:
            # Find path from start to endpoint
            path = find_path_to_endpoint(G, start_point, endpoint)

            if not path or len(path) < 2:
                continue

            # Skip if we've already drawn this path
            path_key = tuple(sorted([start_point, endpoint]))
            if path_key in drawn_paths:
                continue

            # Mark this path as drawn
            drawn_paths.add(path_key)

            # Draw this path from starting point to endpoint
            t.penup()
            move_to_node(path[0])

            # Always set initial heading to 0 for each path
            t.setheading(0)

            # Calculate the initial direction based on first segment
            if len(path) > 1:
                first_pos = t.position()
                # Calculate position of next node
                next_row, next_col = path[1]
                next_x = (next_col - w / 2) * scale
                next_y = (h / 2 - next_row) * scale
                # Calculate direction to next node
                dx = next_x - first_pos[0]
                dy = next_y - first_pos[1]
                initial_heading = np.degrees(np.arctan2(dy, dx))
                # Set the heading
                t.setheading(initial_heading)

            t.pendown()

            # Remember the previous heading
            prev_heading = t.heading()

            # Draw the path and record turning angles at turning points
            for j in range(1, len(path)):
                # Move to the next node in the path
                move_to_node(path[j])

                # If there's another node ahead, calculate new heading
                if j < len(path) - 1:
                    # Get current position after move
                    new_pos = t.position()
                    # Calculate position of the next node
                    next_row, next_col = path[j + 1]
                    next_x = (next_col - w / 2) * scale
                    next_y = (h / 2 - next_row) * scale

                    # Calculate new heading to next node
                    dx = next_x - new_pos[0]
                    dy = next_y - new_pos[1]
                    new_heading = np.degrees(np.arctan2(dy, dx))

                    # Calculate the turning angle (how much the turtle needs to turn)
                    # Using the smallest angle between -180 and +180 degrees
                    turning_angle = (new_heading - prev_heading + 180) % 360 - 180

                    # Check if this is a significant turn
                    if abs(turning_angle) > 5:  # 5 degree threshold for considering it a turn
                        # Update heading
                        t.setheading(new_heading)
                        # Record the turning angle
                        turning_angles.append(turning_angle)
                        # Update previous heading
                        prev_heading = new_heading

        # Update display after each starting point
        turtle.update()

    print("Drawing complete. Close the turtle window to continue with analysis.")
    screen.exitonclick()

    return turning_angles


def plot_angle_frequency(angles):
    """
    Plot a histogram and continuous graph of turning angles against frequency.

    Args:
        angles: List of turning angles at turning points (-180 to +180 degrees)
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Histogram
    bins = np.linspace(-180, 180, 37)  # 36 bins of 10 degrees each
    hist, bin_edges = np.histogram(angles, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    ax1.bar(bin_centers, hist, width=10, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('Histogram of Vessel Turning Angles')
    ax1.set_xlabel('Turning Angle (degrees)')
    ax1.set_ylabel('Frequency')
    ax1.set_xticks(np.arange(-180, 181, 30))
    ax1.grid(True, alpha=0.3)

    # Sort angles for continuous plot
    sorted_angles = sorted(angles)

    # Create frequency data (y-axis)
    y = np.arange(1, len(sorted_angles) + 1)

    # Continuous plot
    ax2.plot(sorted_angles, y, 'r-', linewidth=2)
    ax2.set_title('Continuous Plot of Vessel Turning Angles')
    ax2.set_xlabel('Turning Angle (degrees)')
    ax2.set_ylabel('Cumulative Frequency')
    ax2.set_xlim(-180, 180)
    ax2.set_xticks(np.arange(-180, 181, 30))
    ax2.grid(True)

    # Add additional statistics
    mean_angle = np.mean(angles)
    median_angle = np.median(angles)
    std_angle = np.std(angles)

    stats_text = f"Mean: {mean_angle:.1f}°\nMedian: {median_angle:.1f}°\nStd Dev: {std_angle:.1f}°"
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('vessel_turning_angles.png')  # Save the figure
    plt.show()


def analyze_vessel_angles(image_path):
    """
    Analyze a fundus image and extract vessel turning angles.

    Args:
        image_path: Path to the fundus image
    """
    # Step 1: Process the fundus image for vessel segmentation
    print(f"Processing image: {image_path}")
    results = process_fundus_image(image_path)

    binary_image = results.get('binary_image')
    skeleton_image = results.get('skeleton_image', binary_image)

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

    # Step 4: Find multiple vessel starting points within the optic disc
    print("Finding multiple vessel starting points...")
    all_starting_points = find_multiple_vessel_points_bfs(
        binary_image,
        main_starting_point,
        optic_disc_location,
        radius=optic_disc_radius,
        min_thickness=min_thickness
    )

    # Step 5: Create a graph representation of the skeleton
    print("Creating graph representation...")
    G = skeleton_to_graph(skeleton_image)

    # Step 6: Draw vessels and extract turning angles
    print("Drawing vessels and extracting turning angles...")
    turning_angles = extract_vessel_angles_from_optic_disc(G, original_image.shape[:2], all_starting_points)

    # Step 7: Plot turning angle frequency
    print("Plotting turning angle frequency...")
    plot_angle_frequency(turning_angles)

    # Save turning angles to file for further analysis
    np.savetxt('vessel_turning_angles.txt', turning_angles)

    # Display informative statistics
    print(f"Extracted {len(turning_angles)} turning angles from vessel paths")
    if turning_angles:
        print(f"Turning angle statistics:")
        print(f"  Minimum: {min(turning_angles):.1f}°")
        print(f"  Maximum: {max(turning_angles):.1f}°")
        print(f"  Mean: {np.mean(turning_angles):.1f}°")
        print(f"  Median: {np.median(turning_angles):.1f}°")
        print(f"  Standard deviation: {np.std(turning_angles):.1f}°")

        # Count turns in each direction
        left_turns = sum(1 for angle in turning_angles if angle < 0)
        right_turns = sum(1 for angle in turning_angles if angle > 0)
        straight = sum(1 for angle in turning_angles if angle == 0)

        print(f"Turn direction distribution:")
        print(f"  Left turns (<0°): {left_turns} ({left_turns / len(turning_angles) * 100:.1f}%)")
        print(f"  Right turns (>0°): {right_turns} ({right_turns / len(turning_angles) * 100:.1f}%)")
        print(f"  Straight (0°): {straight} ({straight / len(turning_angles) * 100:.1f}%)")

    return turning_angles


if __name__ == "__main__":
    # Change this to the path of your fundus image
    image_path = "test/images/05_test.tif"
    analyze_vessel_angles(image_path)