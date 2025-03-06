import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import color, morphology
from sklearn.decomposition import PCA
import networkx as nx


def threshold_level(image):
    # Convert to uint8 and flatten
    image_uint8 = (image * 255).astype(np.uint8)
    hist = cv2.calcHist([image_uint8], [0], None, [256], [0, 256])
    hist = hist.flatten()
    bin_centers = np.arange(256)

    # Initialize
    cumulative_sum = np.cumsum(hist)

    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].bar(bin_centers, hist, width=1, color='blue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Pixel Intensity (0-255)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Histogram of Image Pixel Intensities")
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)

    # Plot cumulative sum
    axes[1].plot(bin_centers, cumulative_sum, color='red', linewidth=2)
    axes[1].set_xlabel("Pixel Intensity (0-255)")
    axes[1].set_ylabel("Cumulative Sum")
    axes[1].set_title("Cumulative Sum of Histogram")
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    # Show plots
    plt.tight_layout()
    # plt.show()

    t = np.zeros(100)  # Allocate array for thresholds

    # Initial threshold
    t[0] = np.sum(bin_centers * hist) / cumulative_sum[-1]
    t[0] = np.round(t[0])

    # Calculate mean below threshold and mean above threshold
    i = 0
    idx = int(t[i])

    if idx >= 256:
        idx = 255
    if idx <= 0:
        idx = 1

    cumulative_sum_below = np.sum(hist[:idx])
    if cumulative_sum_below > 0:
        mbt = np.sum(bin_centers[:idx] * hist[:idx]) / cumulative_sum_below
    else:
        mbt = 0

    cumulative_sum_above = np.sum(hist[idx:])
    if cumulative_sum_above > 0:
        mat = np.sum(bin_centers[idx:] * hist[idx:]) / cumulative_sum_above
    else:
        mat = 0

    # Next threshold
    i = 1
    t[i] = np.round((mat + mbt) / 2)

    # Iterate until convergence
    while abs(t[i] - t[i - 1]) >= 1 and i < 98:
        idx = int(t[i])
        if idx >= 256:
            idx = 255
        if idx <= 0:
            idx = 1

        cumulative_sum_below = np.sum(hist[:idx])
        if cumulative_sum_below > 0:
            mbt = np.sum(bin_centers[:idx] * hist[:idx]) / cumulative_sum_below
        else:
            mbt = 0

        cumulative_sum_above = np.sum(hist[idx:])
        if cumulative_sum_above > 0:
            mat = np.sum(bin_centers[idx:] * hist[idx:]) / cumulative_sum_above
        else:
            mat = 0

        i += 1
        t[i] = np.round((mat + mbt) / 2)

    threshold = t[i]
    level = threshold / 255.0

    return level


def process_fundus_image(image_path):
    """
    Process a fundus image to extract the blood vessels
    """
    # Read the image
    test_image = cv2.imread(image_path)
    if test_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Resize the image
    resized_image = cv2.resize(test_image, (565, 584))

    # Convert to float in range [0,1]
    converted_image = resized_image.astype(np.float32) / np.max(resized_image)

    # Convert BGR to RGB for proper LAB conversion
    converted_image_rgb = cv2.cvtColor(converted_image, cv2.COLOR_BGR2RGB)

    # Convert to LAB color space
    lab_image = color.rgb2lab(converted_image_rgb)

    # Create a multiplier for the L channel only
    fill = np.array([1.0, 0.0, 0.0])
    filled_image = np.zeros_like(lab_image)
    filled_image[:, :, 0] = lab_image[:, :, 0] * fill[0]

    # Reshape for PCA
    h, w, c = filled_image.shape
    reshaped_lab_image = filled_image.reshape(-1, c)

    # Apply PCA
    pca = PCA(n_components=3)
    s = pca.fit_transform(reshaped_lab_image)

    # Get the first principal component and reshape
    s = s.reshape(h, w, 3)
    s = s[:, :, 0]

    # Normalize to [0,1]
    s_min = np.min(s)
    s_max = np.max(s)
    gray_image = (s - s_min) / (s_max - s_min)

    # Apply CLAHE
    gray_image_uint8 = (gray_image * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    enhanced_image_uint8 = clahe.apply(gray_image_uint8)
    enhanced_image = enhanced_image_uint8.astype(np.float32) / np.max(enhanced_image_uint8)

    # Apply average filter
    kernel_size = 9
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    filtered_image = cv2.filter2D(enhanced_image, -1, kernel)

    # Subtract enhanced from filtered
    subtracted_image = filtered_image - enhanced_image

    # Apply custom thresholding
    level = threshold_level(subtracted_image)
    binary_image = (subtracted_image > (level - 0.48)).astype(np.uint8) # 0.48

    # Remove small object
    # Convert to boolean for skimage function, then back to uint8
    clean_image = morphology.remove_small_objects(binary_image.astype(bool), min_size=70)
    clean_image = clean_image.astype(np.uint8)

    # Skeletonize the image
    skeleton_image = morphology.skeletonize(clean_image)

    return {
        'filtered_image': filtered_image,
        'binary_image': binary_image,
        'clean_image': clean_image,
        'skeleton_image': skeleton_image
    }


def skeleton_to_graph(skeleton):
    G = nx.Graph()  # Create an empty graph
    rows, cols = np.where(skeleton > 0)  # Look for all the white pixels in the Image and extract their coordinates
    for y, x in zip(rows, cols):
        G.add_node((y, x))  # Add a node to these extracted coordinates
    for y, x in zip(rows, cols):
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1),
                       (1, 1)]:  # Check in all 8 directions for neighbouring white pixels
            neighbor = (y + dy, x + dx)  # Calculate the new position of this neighbouring pixel
            if neighbor in G.nodes:  # Check if this neighbour is white or black
                G.add_edge((y, x), neighbor)  # If it is white connect the nodes and make it an edge (Line)
    return G


def graph_based_analysis(skeleton_image):
    # Build the graph from the skeleton
    graph = skeleton_to_graph(skeleton_image > 0)

    # Initialize lists to store endpoints and bifurcation points
    endpoints_graph = []
    bifurcations_graph = []

    # Iterate through each node in the graph
    for node in graph.nodes:
        # Get the list of neighbors for the current node
        neighbors = list(graph.neighbors(node))

        # If the node has exactly one neighbor, it is an endpoint
        if len(neighbors) == 1:
            endpoints_graph.append(node)

        # If the node has three or more neighbors, it is a bifurcation point
        elif len(neighbors) >= 3:
            bifurcations_graph.append(node)

    # Store the Coordinates as numpy array
    endpoint_coords_graph = np.array(endpoints_graph)
    bifurcation_coords_graph = np.array(bifurcations_graph)

    return endpoint_coords_graph, bifurcation_coords_graph

def find_branches_in_skeleton(G):
    """
    Extracts a list of vessel 'branches' from a skeleton graph G.

    Each branch is a path between two 'break' nodes:
      - Endpoints (degree == 1)
      - Junctions (degree >= 3)
    Nodes with degree == 2 are considered 'intermediate' and
    lie along the branch rather than splitting it.

    Returns: list of paths, each path = [ (r1, c1), (r2, c2), ... ]
    """
    all_branches = []
    # Process each connected component separately.
    for comp in nx.connected_components(G):
        subG = G.subgraph(comp).copy()
        # Identify break nodes (endpoints or junctions).
        break_nodes = set(n for n in subG.nodes if subG.degree(n) != 2)

        # If there are no break nodes, the component is a cycle or a chain.
        if not break_nodes:
            # For a cycle (or pure chain), simply get one traversal.
            some_node = next(iter(subG.nodes))
            branch = list(nx.dfs_tree(subG, source=some_node).nodes())
            all_branches.append(branch)
            continue

        # To avoid extracting the same branch twice, track visited edges.
        visited_edges = set()

        # For each break node, walk along each adjacent branch.
        for bn in break_nodes:
            for nb in subG.neighbors(bn):
                edge = tuple(sorted((bn, nb)))
                if edge in visited_edges:
                    continue
                # Start a new branch from bn to nb.
                branch = [bn, nb]
                visited_edges.add(edge)
                prev = bn
                current = nb

                # Follow the chain until reaching a break node.
                # For degree-2 nodes, there should be exactly one neighbor that is not the previous node.
                while current not in break_nodes:
                    neighbors = list(subG.neighbors(current))
                    # Since current has degree 2, one neighbor is 'prev', the other is next.
                    next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
                    branch.append(next_node)
                    # Mark the traversed edge as visited.
                    visited_edges.add(tuple(sorted((current, next_node))))
                    prev, current = current, next_node
                all_branches.append(branch)

    return all_branches


if __name__ == '__main__':
    Image = 'test/images/03_test.tif'
    results = process_fundus_image(Image)
    filtered_image = results.get('filtered_image')
    binary_image = results.get('binary_image')
    clean_image = results.get('clean_image')
    skeleton_image = results.get('skeleton_image')

    # Perform graph-based analysis
    endpoint_coords_graph, bifurcation_coords_graph = graph_based_analysis(skeleton_image)

    # Display results
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Filtered Image')

    plt.subplot(2, 2, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Binary Image')

    plt.subplot(2, 2, 3)
    plt.imshow(clean_image, cmap='gray')
    plt.title('Clean Image')

    plt.subplot(2, 2, 4)
    plt.imshow(skeleton_image, cmap='gray')
    plt.title('Skeletonize Image')

    plt.tight_layout()
    # plt.show()

    # Create a figure with two side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the original skeleton image
    axes[0].imshow(skeleton_image, cmap='gray')
    axes[0].set_title("Skeleton")
    axes[0].axis('off')

    # Plot the skeleton with endpoints and bifurcations overlaid
    axes[1].imshow(skeleton_image, cmap='gray')
    if len(endpoint_coords_graph) > 0:
        axes[1].scatter(endpoint_coords_graph[:, 1], endpoint_coords_graph[:, 0], color='blue', label='Endpoints', s=10)
    if len(bifurcation_coords_graph) > 0:
        axes[1].scatter(bifurcation_coords_graph[:, 1], bifurcation_coords_graph[:, 0], color='red',
                        label='Bifurcations', s=2)
    axes[1].set_title("Graph-Based Analysis (DFS/BFS)")
    axes[1].legend()
    axes[1].axis('off')

    plt.tight_layout()
    # plt.show()