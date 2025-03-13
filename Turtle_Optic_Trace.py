import cv2
import numpy as np
import matplotlib.pyplot as plt
import turtle
from DIP import (process_fundus_image, skeleton_to_graph)
from OD_Detection import detect_optic_disc


def find_best_starting_vessel(binary_image, optic_disc_location, radius):

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


def find_multiple_vessel_points(binary_image, best_starting_point, radius, min_thickness):

    # Extract dimensions and create coordinates
    h, w = binary_image.shape
    y_grid, x_grid = np.ogrid[:h, :w]

    # Unpack the starting point (in row, col format)
    center_row, center_col = best_starting_point

    # Create a circular mask around the best starting point
    dist_from_center = np.sqrt((x_grid - center_col) ** 2 + (y_grid - center_row) ** 2)
    inside_circle_mask = dist_from_center < radius

    # Apply mask to the binary image
    vessels_in_circle = binary_image & inside_circle_mask

    # List to store vessel points with their thickness scores
    vessel_points = []

    # Get coordinates of all vessel pixels in the circle
    points = np.where(vessels_in_circle > 0)

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
            vessel_points.append((int(row), int(col), thickness_score))

    # Sort by thickness score (higher first)
    vessel_points.sort(key=lambda x: x[2], reverse=True)

    # Extract the top points, removing duplicates that are too close to each other
    # We'll use a minimum distance between points to filter out duplicates
    min_distance_between_points = 15
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

    # Always include the original best starting point
    if best_starting_point not in selected_points and len(selected_points) > 0:
        selected_points.insert(0, best_starting_point)
    elif len(selected_points) == 0:
        selected_points.append(best_starting_point)

    print(f"Selected {len(selected_points)} vessel starting points")
    return selected_points


def draw_skeleton_multi_start(G, image_shape, starting_points, scale=1.0):
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
    screen.title("Fundus Vessel Visualization")

    # Create a turtle with customized appearance
    t = turtle.Turtle()
    t.hideturtle()
    t.speed(0)
    t.pensize(2)  # Thicker line for better visibility
    t.pencolor("red")  # Red color for better visibility
    turtle.tracer(0, 0)  # disable animation for faster drawing

    # Keep track of all visited nodes across all starting points
    all_visited = set()

    # Helper to move the turtle to a pixel node (row, col)
    def move_to_node(node):
        (r, c) = node
        x = (c - w / 2) * scale
        y = (h / 2 - r) * scale
        t.goto(x, y)

    # Process each starting point
    for i, start_point in enumerate(valid_starting_points):
        # Skip if this point was already visited in a previous traversal
        if start_point in all_visited:
            continue

        # Use a different color for each starting point
        colors = ["red", "blue", "green", "purple", "orange"]
        t.pencolor(colors[i % len(colors)])

        # Set for visited nodes in this traversal
        visited = set()
        visited.add(start_point)
        all_visited.add(start_point)

        # Iterative DFS using a stack
        stack = [start_point]

        # Move turtle to the starting node
        t.penup()
        move_to_node(start_point)
        t.pendown()

        while stack:
            u = stack[-1]

            # Get any unvisited neighbor (considering global visited set)
            unvisited_neighbors = [v for v in G[u] if v not in all_visited]

            if unvisited_neighbors:
                v = unvisited_neighbors[0]
                visited.add(v)
                all_visited.add(v)
                # Move forward to the neighbor
                move_to_node(v)
                stack.append(v)
            else:
                # No unvisited neighbors, so backtrack
                stack.pop()
                if stack:
                    # Move turtle back to the new top of the stack
                    t.penup()
                    move_to_node(stack[-1])
                    t.pendown()

        # Update the screen after each starting point is processed
        turtle.update()

    turtle.mainloop()


def visualize_optic_disc_and_starting_points(original_image, binary_image,
                                             main_starting_point, all_starting_points):
    search_radius = 55

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot original image with optic disc circle and starting points
    if len(original_image.shape) == 3:
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    else:
        axes[0].imshow(original_image, cmap='gray')

    # Draw circle around the main starting point
    search_circle = plt.Circle((main_starting_point[1], main_starting_point[0]),
                               search_radius, color='r', fill=False, linewidth=2)
    axes[0].add_patch(search_circle)

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

    # Draw circle around the main starting point
    search_circle2 = plt.Circle((main_starting_point[1], main_starting_point[0]),
                                search_radius, color='r', fill=False, linewidth=2)
    axes[1].add_patch(search_circle2)

    # Mark main starting point with a star
    axes[1].scatter(main_starting_point[1], main_starting_point[0],
                    color='yellow', s=150, marker='*', label='Main Starting Point')

    # Mark all other starting points with smaller dots
    for point in all_starting_points:
        if point != main_starting_point:  # Skip the main point which already has a star
            axes[1].scatter(point[1], point[0],
                            color='blue', s=20, marker='o')

    axes[1].set_title('Binary Vessel Mask')

    plt.tight_layout()
    plt.show()


def analyze_fundus_image(image_path):
    print(f"Processing image: {image_path}")

    # Step 1: Process the fundus image for vessel segmentation
    print("Segmenting vessels...")
    results = process_fundus_image(image_path)

    # Use binary_image as specified
    binary_image = results.get('binary_image')

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
    search_radius = 55
    Thickness = 1

    # Step 3: Find the best vessel starting point
    print("Finding optimal vessel starting point...")
    main_starting_point = find_best_starting_vessel(
        binary_image,
        optic_disc_location,
        radius=optic_disc_radius
    )

    # Step 4: Find multiple vessel starting points within a circle around the main point
    print("Finding additional vessel starting points...")
    all_starting_points = find_multiple_vessel_points(
        binary_image,
        main_starting_point,
        radius=search_radius,  # Search within this radius of the main point
        min_thickness=Thickness  # Minimum thickness threshold
    )

    # Step 5: Visualize optic disc and all starting points
    visualize_optic_disc_and_starting_points(
        original_image,
        binary_image,
        main_starting_point,
        all_starting_points
    )

    # Step 6: Create a graph representation of the skeleton
    print("Creating graph representation...")
    G = skeleton_to_graph(binary_image)

    # Step 7: Draw vasculature pattern starting from multiple points
    print("Drawing vasculature pattern from multiple starting points...")
    print("Close the visualization window to continue.")
    draw_skeleton_multi_start(G, original_image.shape[:2], all_starting_points, scale=1.0)


if __name__ == "__main__":
    image_path = "test/images/05_test.tif"
    analyze_fundus_image(image_path)