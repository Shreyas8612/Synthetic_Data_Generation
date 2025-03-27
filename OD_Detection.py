import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from skimage.measure import label, regionprops
from skimage import color
from sklearn.decomposition import PCA
from DIP import threshold_level


def preprocess_image(image):
    # Downscale to 128x128
    resized = cv2.resize(image, (128, 128))

    # Convert to float in range [0,1]
    converted_image = resized.astype(np.float32) / np.max(resized)

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
    gray = (s - s_min) / (s_max - s_min)

    # Apply median filtering with 3x3 kernel
    median_filtered = median_filter(gray, size=3)

    # Calculate standard deviation
    sd = np.std(median_filtered)
    # Apply intensity transformation based on standard deviation (Gamma Correction)
    bias = 1
    transformed = np.power(median_filtered, 1 / (bias * sd))

    # Background subtraction
    # Estimate background using median filter with 11x11 kernel
    background = median_filter(transformed, size=11)

    subtracted = transformed - background

    # Median filter again with 7x7 kernel
    median_filtered_again = median_filter(subtracted, size=7)

    # Exponentiation to make bright regions brighter and dark regions darker
    a = 2
    exponentiated = np.exp(a * median_filtered_again)

    # Apply anisotropic diffusion
    diffused = anisotropic_diffusion(exponentiated, niter=2, kappa=30, gamma=0.1)

    return diffused


def anisotropic_diffusion(img, niter=1, kappa=50, gamma=0.1):
    # Convert to float32 for numerical stability.
    diffused = img.astype(np.float32)

    for _ in range(niter):
        # Compute finite differences (gradients) in the four directions:
        # North gradient (top neighbor)
        gradN = np.roll(diffused, 1, axis=0) - diffused
        # South gradient (bottom neighbor)
        gradS = np.roll(diffused, -1, axis=0) - diffused
        # East gradient (right neighbor)
        gradE = np.roll(diffused, -1, axis=1) - diffused
        # West gradient (left neighbor)
        gradW = np.roll(diffused, 1, axis=1) - diffused

        # Perona–Malik conduction coefficients in each direction.
        # conduction function: c = exp( - (|gradI| / kappa)^2 )
        cN = np.exp(-(gradN / kappa) ** 2)
        cS = np.exp(-(gradS / kappa) ** 2)
        cE = np.exp(-(gradE / kappa) ** 2)
        cW = np.exp(-(gradW / kappa) ** 2)

        # Update the image by discrete PDE:
        diffused += gamma * (
                cN * gradN + cS * gradS +
                cE * gradE + cW * gradW
        )

    return diffused


def gravitational_edge_detection(img):
    # Get image dimensions
    height, width = img.shape

    # Initialize the edge map
    edge_map = np.zeros((height, width), dtype=np.float64)  # To store the strength of each of the edges

    # For each pixel in the image
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Get the 8-neighborhood of the pixel
            neighborhood = img[i - 1:i + 2, j - 1:j + 2]

            # Calculate average intensity in the neighborhood
            g_avg = np.mean(neighborhood)

            # Calculate standard deviation in the neighborhood
            sigma = np.std(neighborhood)

            # Calculate Gravitational Constant - Strength of the Neighbours
            # If the Current pixel is very different from the mean then this makes C small -> Lowering its influence
            # If the Current pixel is similar to the mean then this makes C large -> Increasing its influence
            C = 1 / (1 + np.exp(sigma * (img[i, j] - g_avg)))

            # Initialize force components
            Fx = 0
            Fy = 0

            # For each neighbor in the 8-neighborhood
            for k in range(i - 1, i + 2):
                for l in range(j - 1, j + 2):
                    # Skip the center pixel itself
                    if k == i and l == j:
                        continue

                    # Calculate vector r from (i,j) to (k,l) -> Euclidean Distance
                    # (up,down,left,right) => r = 1
                    # Diagonals => r = sqrt(2)
                    r_magnitude = np.sqrt((k - i) ** 2 + (l - j) ** 2)

                    # Similar to newtons law -> Find the force
                    # C * img[k, l] is the mass of the neighbours and interactions coefficients
                    f_x = C * img[k, l] * (k - i) / (r_magnitude ** 3)  # (k-i)/r = unit vector in x-direction
                    f_y = C * img[k, l] * (l - j) / (r_magnitude ** 3)  # (l-j)/r = unit vector in y-direction

                    # Add to the total force components
                    Fx += f_x
                    Fy += f_y

            # Calculate the magnitude of the force
            F_magnitude = np.sqrt(Fx ** 2 + Fy ** 2)

            # Assign the magnitude to the edge map
            edge_map[i, j] = F_magnitude

    # Normalize the edge map to [0, 255]
    edge_map = (edge_map - np.min(edge_map)) / (np.max(edge_map) - np.min(edge_map)) * 255

    return edge_map.astype(np.uint8)


def post_process_edge_map(edge_map):
    # Get image dimensions
    height, width = edge_map.shape
    center_y, center_x = height // 2, width // 2

    # Create a mask for filtering
    mask = np.ones_like(edge_map, dtype=bool)

    # Remove pixels outside a radius of 55 pixels from the center
    for i in range(height):
        for j in range(width):
            if (i - center_y) ** 2 + (j - center_x) ** 2 > 55 ** 2:  # Equation of a Circle
                mask[i, j] = False

    # Remove pixels inside a radius of 10 pixels from the center
    for i in range(height):
        for j in range(width):
            if (i - center_y) ** 2 + (j - center_x) ** 2 < 10 ** 2:
                mask[i, j] = False

    # Remove rectangular regions from top and bottom (35 pixels height)
    pixel_rect = 35
    mask[:pixel_rect, :] = False
    mask[-pixel_rect:, :] = False

    # Remove a vertical strip of 12 pixels width from the center
    mask[:, center_x - 6:center_x + 7] = False

    # Apply the mask to the edge map
    filtered_edge_map = edge_map.copy()
    filtered_edge_map[~mask] = 0

    # Thresholding
    level = threshold_level(filtered_edge_map / 255.0)
    binary_edge_map = (filtered_edge_map > level * 255).astype(np.uint8) * 255

    return binary_edge_map


def candidate_selection(binary_edge_map):
    D = 15.7  # Predetermined threshold

    # Label connected components -> Blobs in the image
    labeled = label(binary_edge_map)

    # Get properties like area, centroid, etc of the labelled regions
    regions = regionprops(labeled)

    # Sort regions by area (largest first)
    regions.sort(key=lambda x: x.area, reverse=True)

    # Iterate through regions to find the candidate
    candidate_found = False
    selected_region_idx = -1

    for i, region in enumerate(regions):
        # Get coordinates of the pixels in the region
        coords = region.coords

        # Calculate centroid
        centroid = region.centroid

        # Calculate Euclidean distance from centroid to each pixel in the region
        total_distance = 0
        for coord in coords:
            distance = np.sqrt((coord[0] - centroid[0]) ** 2 + (coord[1] - centroid[1]) ** 2)
            total_distance += distance

        # Average distance of region pixels from its centroid
        # Circular and Compact regions have low mean distance
        # Irregular and spread out regions have high mean distance
        mean_distance = total_distance / len(coords)

        # If mean distance is less than threshold, select this region
        if mean_distance < D:
            candidate_found = True
            selected_region_idx = i
            optic_disc_location = (int(centroid[1]), int(centroid[0]))
            break

    # If no suitable region is found, return the centroid of the largest region
    if not candidate_found and regions:
        selected_region_idx = 0
        centroid = regions[0].centroid
        optic_disc_location = (int(centroid[1]), int(centroid[0]))
    elif not regions:
        optic_disc_location = (64, 64)  # Center of the image

    return optic_disc_location


def detect_optic_disc(image_path):
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to read image: {image_path}")
        return None

    original_height, original_width = image.shape[:2]

    # Preprocess the image
    preprocessed = preprocess_image(image)

    # Apply gravitational edge detection
    edge_map = gravitational_edge_detection(preprocessed)

    # Post-process edge map
    binary_edge_map = post_process_edge_map(edge_map)

    # Candidate selection to find optic disc
    optic_disc_location_128 = candidate_selection(binary_edge_map)

    # Scale the location back to the original image dimensions
    optic_disc_location = (
        int(optic_disc_location_128[0] * original_width / 128),
        int(optic_disc_location_128[1] * original_height / 128)
    )

    return optic_disc_location


# If this file is run directly, test on a sample image
if __name__ == "__main__":
    image_path = "test/images/15_test.tif"
    optic_disc_location = detect_optic_disc(image_path)

    if optic_disc_location is not None:
        print(f"Optic disc detected at: {optic_disc_location}")

        # Visualize the result
        image = cv2.imread(image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.plot(optic_disc_location[0], optic_disc_location[1], 'ro', markersize=10)
        plt.title("Optic Disc Detection")
        plt.axis('off')
        plt.show()
    else:
        print("Failed to detect optic disc.")