import networkx as nx
import turtle
from collections import namedtuple
from math import atan2, degrees, sqrt
from DIP import *

# Define namedtuples for our structures
Junction = namedtuple('Junction', ['index', 'x', 'y'])
Vector = namedtuple('Vector', ['start_junction', 'end_junction', 'length', 'angle', 'points'])


def extract_junctions_and_vectors(G, branches, image_shape):
    """
    Extract junctions and vectors from the graph and branches.

    Args:
        G (nx.Graph): Graph representation of the skeleton
        branches (list): List of branch paths
        image_shape (tuple): Image dimensions (height, width)

    Returns:
        tuple: (junctions, vectors)
            - junctions: list of Junction objects
            - vectors: list of Vector objects
    """
    # Identify all nodes with degree != 2 (endpoints and branch points)
    junction_nodes = [node for node in G.nodes if G.degree(node) != 2]

    # Create a mapping of node to junction index
    junction_mapping = {}
    junctions = []

    for idx, node in enumerate(junction_nodes):
        # Convert from (row, col) to (x, y) coordinates
        row, col = node
        # Adjust coordinates to be centered relative to image dimensions
        x = col - image_shape[1] / 2
        y = image_shape[0] / 2 - row

        junction = Junction(index=idx, x=x, y=y)
        junctions.append(junction)
        junction_mapping[node] = idx

    # Process branches to create vectors
    vectors = []
    vector_idx = 0

    # For visualization, create a new graph that will have junctions as nodes
    # and vectors as edges
    junction_graph = nx.Graph()
    for j in junctions:
        junction_graph.add_node(j.index, pos=(j.x, j.y))

    for branch in branches:
        if len(branch) < 2:
            continue

        # Find the nearest junction nodes for the start and end of this branch
        start_node = branch[0]
        end_node = branch[-1]

        # If the branch endpoints are not junctions, find the closest junctions
        if start_node not in junction_mapping:
            start_node = min(junction_nodes,
                             key=lambda n: (n[0] - start_node[0]) ** 2 + (n[1] - start_node[1]) ** 2)

        if end_node not in junction_mapping:
            end_node = min(junction_nodes,
                           key=lambda n: (n[0] - end_node[0]) ** 2 + (n[1] - end_node[1]) ** 2)

        start_junction_idx = junction_mapping[start_node]
        end_junction_idx = junction_mapping[end_node]

        # Skip self-loops for simplicity
        if start_junction_idx == end_junction_idx:
            continue

        # Convert branch points to (x, y) coordinates
        points = []
        for node in branch:
            row, col = node
            x = col - image_shape[1] / 2
            y = image_shape[0] / 2 - row
            points.append((x, y))

        # Calculate vector properties
        # Length is the Euclidean distance between start and end points
        start_x, start_y = points[0]
        end_x, end_y = points[-1]
        length = sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

        # Angle is calculated from the vector between start and end points
        dx = end_x - start_x
        dy = end_y - start_y
        angle = degrees(atan2(dy, dx))

        # Create the vector
        vector = Vector(
            start_junction=start_junction_idx,
            end_junction=end_junction_idx,
            length=length,
            angle=angle,
            points=points
        )

        vectors.append(vector)

        # Add edge to junction graph
        junction_graph.add_edge(start_junction_idx, end_junction_idx,
                                vector_idx=vector_idx, length=length, angle=angle)
        vector_idx += 1

    # Visualize junction graph
    plt.figure(figsize=(10, 10))
    pos = {j.index: (j.x, j.y) for j in junctions}
    nx.draw(junction_graph, pos, node_size=100, node_color='red', with_labels=True)
    edge_labels = {(u, v): f"{d['vector_idx']}" for u, v, d in junction_graph.edges(data=True)}
    nx.draw_networkx_edge_labels(junction_graph, pos, edge_labels=edge_labels)
    plt.title("Junction Graph")
    plt.axis('equal')
    plt.axis('off')
    # plt.show()

    return junctions, vectors


def draw_vectors_with_turtle(junctions, vectors, scale=1.0):
    """
    Draw the vasculature using the vector representation.

    Args:
        junctions (list): List of Junction objects
        vectors (list): List of Vector objects
        scale (float): Scale factor for drawing
    """
    # Set up turtle screen
    screen = turtle.Screen()
    screen.setup(800, 800)
    screen.title("Vector RAG Turtle Vasculature")

    # Create and configure the turtle
    t = turtle.Turtle()
    t.hideturtle()
    t.speed(0)
    turtle.tracer(0, 0)  # Disable animation for faster drawing

    # Draw junctions as red dots
    t.penup()
    for j in junctions:
        t.goto(j.x * scale, j.y * scale)
        t.dot(6, "red")

    # Draw vectors
    for vector_idx, vector in enumerate(vectors):
        # Get start and end junctions
        start_j = junctions[vector.start_junction]
        end_j = junctions[vector.end_junction]

        # Start drawing from the start junction
        t.penup()
        t.goto(start_j.x * scale, start_j.y * scale)
        t.pendown()

        # If we have detailed points for the vector, draw them
        if vector.points and len(vector.points) > 2:
            for x, y in vector.points[1:-1]:  # Skip first and last (junctions)
                t.goto(x * scale, y * scale)

        # End at the end junction
        t.goto(end_j.x * scale, end_j.y * scale)

    # Update the screen
    turtle.update()
    turtle.done()


def generate_vasculature(G, branches, image_shape, scale=1.0):
    """
    Main function to generate the vasculature using the Vector RAG approach.

    Args:
        G (nx.Graph): Graph representation of the skeleton
        branches (list): List of branch paths
        image_shape (tuple): Image dimensions (height, width)
        scale (float): Scale factor for drawing
    """
    # Extract junctions and vectors
    junctions, vectors = extract_junctions_and_vectors(G, branches, image_shape)

    # Print summary statistics
    print(f"Extracted {len(junctions)} junctions and {len(vectors)} vectors")

    # Draw the vasculature
    # draw_vectors_with_turtle(junctions, vectors, scale)

    return junctions, vectors


# This function can be used to generate new vasculature based on the extracted features
def generate_new_vasculature(junctions, vectors, variations=0.1, scale=1.0):
    """
    Generate new vasculature by introducing variations to the extracted features.

    Args:
        junctions (list): List of Junction objects
        vectors (list): List of Vector objects
        variations (float): Amount of random variation to introduce (0.0-1.0)
        scale (float): Scale factor for drawing
    """
    # Create modified junctions with small random displacements
    new_junctions = []
    for j in junctions:
        # Add small random variations to positions
        x_var = j.x + np.random.normal(0, variations * 10)
        y_var = j.y + np.random.normal(0, variations * 10)
        new_junctions.append(Junction(j.index, x_var, y_var))

    # Create modified vectors with variations in angles and lengths
    new_vectors = []
    for v in vectors:
        # Add variations to length and angle
        length_var = v.length * (1 + np.random.normal(0, variations))
        angle_var = v.angle + np.random.normal(0, variations * 15)  # Degrees

        # For simplicity, we'll just use the start and end points
        new_vectors.append(Vector(
            start_junction=v.start_junction,
            end_junction=v.end_junction,
            length=length_var,
            angle=angle_var,
            points=None  # We'll generate points on-the-fly when drawing
        ))

    # Set up turtle screen
    screen = turtle.Screen()
    screen.setup(800, 800)
    screen.title("Generated New Vasculature")

    # Create and configure the turtle
    t = turtle.Turtle()
    t.hideturtle()
    t.speed(0)
    turtle.tracer(0, 0)  # Disable animation for faster drawing

    # Draw junctions as red dots
    t.penup()
    for j in new_junctions:
        t.goto(j.x * scale, j.y * scale)
        #t.dot(5, "red")

    # Draw vectors
    for vector in new_vectors:
        # Get start and end junctions
        start_j = new_junctions[vector.start_junction]
        end_j = new_junctions[vector.end_junction]

        # Start drawing from the start junction
        t.penup()
        t.goto(start_j.x * scale, start_j.y * scale)
        t.pendown()

        # If we don't have detailed points, create a slightly curved path
        if vector.points is None:
            # Create a slight curve by adding a midpoint with some deviation
            mid_x = (start_j.x + end_j.x) / 2
            mid_y = (start_j.y + end_j.y) / 2

            # Add some perpendicular deviation for a natural curve
            dx = end_j.x - start_j.x
            dy = end_j.y - start_j.y
            # Perpendicular direction
            perp_x = -dy
            perp_y = dx
            # Normalize and scale by vector length and variation factor
            mag = sqrt(perp_x ** 2 + perp_y ** 2)
            if mag > 0:
                curve_x = mid_x + perp_x / mag * vector.length * np.random.normal(0, variations)
                curve_y = mid_y + perp_y / mag * vector.length * np.random.normal(0, variations)

                # Draw with a curve
                t.goto(curve_x * scale, curve_y * scale)

        # End at the end junction
        t.goto(end_j.x * scale, end_j.y * scale)

    # Update the screen
    turtle.update()
    turtle.done()


# Example usage in main function
if __name__ == "__main__":
    Image = process_fundus_image('test/images/01_test.tif')['skeleton_image']
    image_shape = Image.shape

    # Step 1: Convert the skeleton to a graph
    G = skeleton_to_graph(Image)

    # Step 2: Find meaningful vessel branches
    branches = find_branches_in_skeleton(G)

    # Step 3: Generate vasculature using Vector RAG approach
    junctions, vectors = generate_vasculature(G, branches, image_shape, scale=1.0)

    # Step 4: Generate new vasculature with variations
    # This will create a similar but not identical pattern
    generate_new_vasculature(junctions, vectors, variations=0.1, scale=1.0)