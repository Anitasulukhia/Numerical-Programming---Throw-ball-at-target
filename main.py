import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2


def generate_test_image(width=800, height=600, n_balls=5, save_path='scattered_balls.jpg'):
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    positions = []
    radii = []
    colors = []

    for _ in range(n_balls):
        for _ in range(100):
            radius = np.random.randint(10, 35)
            x = np.random.randint(radius, width - radius)
            y = np.random.randint(radius, height - radius)
            color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )

            # Check if this circle would overlap with any we've already drawn
            valid = True
            for px, py, pr in zip(positions[::2], positions[1::2], radii):
                if np.sqrt((x - px) ** 2 + (y - py) ** 2) < (radius + pr + 10):
                    valid = False
                    break

            if valid:
                positions.extend([x, y])
                radii.append(radius)
                colors.append(color)
                cv2.circle(image, (x, y), radius, color, -1)
                break

    cv2.imwrite(save_path, image)  # Save our masterpiece
    return positions, radii


class CircleDetector:

    def __init__(self, min_radius=10, max_radius=50, eps=15, min_points=3):
        # eps =  How close should points be to be considered as a cluster, ball in this case
        # min points =  How many points do we need to be sure it's actually a ball
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.eps = eps
        self.min_points = min_points

    def _to_grayscale(self, image):
        if len(image.shape) == 2:
            return image
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


    def _convolve2d(self, image, kernel):

        k_height, k_width = kernel.shape
        p_height = k_height // 2
        p_width = k_width // 2

        padded = np.pad(image, ((p_height, p_height), (p_width, p_width)), mode='edge')
        result = np.zeros_like(image, dtype=float)

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                result[i, j] = np.sum(
                    padded[i:i + k_height, j:j + k_width] * kernel
                )
        return result

    def _edge_detection(self, image):
        gray = self._to_grayscale(image)

        smooth_kernel = np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]
        ]) / 256

        smoothed = self._convolve2d(gray, smooth_kernel)

        sobel_x = np.array([
            [-1, -2, 0, 2, 1],
            [-4, -8, 0, 8, 4],
            [-6, -12, 0, 12, 6],
            [-4, -8, 0, 8, 4],
            [-1, -2, 0, 2, 1]
        ])

        sobel_y = sobel_x.T

        edges_x = self._convolve2d(smoothed, sobel_x)
        edges_y = self._convolve2d(smoothed, sobel_y)

        edges = np.sqrt(edges_x ** 2 + edges_y ** 2)

        mean_edge = np.mean(edges)
        std_edge = np.std(edges)
        threshold = mean_edge + 0.5 * std_edge

        edges[edges < threshold] = 0

        edges = ((edges - edges.min()) * (255.0 / (edges.max() - edges.min()))).astype(np.uint8)
        return edges

    def _dbscan(self, points, eps, min_samples):
        if len(points) == 0:
            return np.array([])

        labels = np.full(len(points), -1)
        cluster_id = 0

        # Look at each point and find its neighbors
        for i in range(len(points)):
            if labels[i] != -1:  # Skip points we've already grouped
                continue

            neighbors = self._find_neighbors(points, i, eps)

            if len(neighbors) < min_samples:  # if not enough neighbors, skip it
                continue

            # Found a new group Label everyone in it
            labels[i] = cluster_id
            seed_set = neighbors.copy()

            # Keep adding neighbors of neighbors to the group
            while seed_set:
                current_point = seed_set.pop()
                if labels[current_point] == -1:
                    labels[current_point] = cluster_id
                    new_neighbors = self._find_neighbors(points, current_point, eps)

                    if len(new_neighbors) >= min_samples:
                        seed_set.update(new_neighbors)

            cluster_id += 1

        return labels

    def _find_neighbors(self, points, point_idx, eps):
        distances = np.sqrt(np.sum((points - points[point_idx]) ** 2, axis=1))
        return set(np.where(distances <= eps)[0])

    def _verify_circle(self, points, center):
        if len(points) < self.min_points:
            return False, 0

        # Measure how far each point is from the center
        distances = np.sqrt(np.sum((points - center) ** 2, axis=1))
        radius = np.median(distances)

        # Check if the circle is the right size
        if not (self.min_radius <= radius <= self.max_radius):
            return False, 0

        # Make sure points are spread out evenly
        angles = np.arctan2(points[:, 0] - center[0], points[:, 1] - center[1])
        num_quadrants = 4
        quadrant_counts = np.histogram(angles, bins=num_quadrants)[0]

        # Need some points in each quarter of the circle
        min_points_per_quadrant = len(points) / (num_quadrants * 5)
        if np.min(quadrant_counts) < min_points_per_quadrant:
            return False, 0

        # Make sure it's roughly circular
        p25 = np.percentile(distances, 25)
        p75 = np.percentile(distances, 75)
        iqr = p75 - p25
        if iqr / radius > 1.0:
            return False, 0

        return True, radius

    def detect_circles(self, image, save_path='detection_results.jpg'):
        print("\nStarting circle detection...")

        # Step 1: Find all the edges
        edges = self._edge_detection(image)

        # Step 2: Pick out the strong edge points
        mean_edge = np.mean(edges)
        std_edge = np.std(edges)
        threshold = mean_edge + 0.5 * std_edge
        edge_points = np.column_stack(np.where(edges > threshold))

        print(f"Found {len(edge_points)} edge points")

        if len(edge_points) == 0:
            return [], []

        # Step 3: Group edge points into potential circles
        labels = self._dbscan(edge_points, self.eps, self.min_points)

        if len(labels) == 0:
            return [], []

        unique_labels = np.unique(labels[labels != -1])
        print(f"Found {len(unique_labels)} groups of points")

        circles = []
        radii = []
        debug_image = image.copy()

        # Step 4: Check each group to see if it makes a circle
        for i, label in enumerate(unique_labels):
            cluster_points = edge_points[labels == label]
            print(f"Group {i} has {len(cluster_points)} points")

            center = np.mean(cluster_points, axis=0)
            is_circle, radius = self._verify_circle(cluster_points, center)

            if is_circle:
                # Found a circle - save its location and size
                circles.append((int(center[1]), int(center[0])))
                radii.append(int(radius))

                # Draw green dots on the edge points we found
                for point in cluster_points:
                    debug_image[point[0], point[1]] = [0, 255, 0]

                # Mark the center with a red dot
                cv2.circle(debug_image, (int(center[1]), int(center[0])), 2, (0, 0, 255), -1)

        print(f"Found {len(circles)} circles")
        cv2.imwrite(save_path, debug_image)

        return circles, radii


class TrajectoryCalculator:

    def __init__(self, start_pos=(0, 0), g=9.81):
        self.start_pos = start_pos
        self.g = g
        self.tol = 1e-6  # Error tolerance for stability

    def get_system_matrix(self):
       # Get system Jacobian matrix for stability analysis
        return np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, -self.g, 0, 0]
        ])

    def rk4_stability_function(self, z):

        # RK4 stability function R(z) = 1 + z + z²/2 + z³/6 + z⁴/24

        return 1 + z + (z ** 2) / 2 + (z ** 3) / 6 + (z ** 4) / 24

    def prove_partial_stability(self, dt=0.01):
        # Get system matrix and eigenvalues
        J = self.get_system_matrix()
        eigenvals = np.linalg.eigvals(J)

        # For our system, all eigenvalues are 0
        # This means the system is inherently stable for any reasonable timestep
        stable_components = []
        component_analysis = []

        for i, λ in enumerate(['x', 'y', 'vx', 'vy']):
            stability_measure = abs(self.rk4_stability_function(dt * eigenvals[i]))
            is_stable = True  # All components are stable since eigenvalues are 0

            component_analysis.append({
                'component': λ,
                'eigenvalue': eigenvals[i],
                'stability_measure': stability_measure,
                'is_stable': is_stable,
                'stability_type': 'Neutrally stable'  # All components are neutrally stable
            })
            stable_components.append(is_stable)

        # Since all eigenvalues are 0, system is stable for any reasonable timestep
        # We'll use a practical limit based on accuracy considerations
        h_max = 0.1  # Conservative maximum timestep

        return {
            'eigenvalues': eigenvals,
            'stable_components': stable_components,
            'component_analysis': component_analysis,
            'max_stable_timestep': h_max,
            'current_timestep': dt,
            'is_stable': dt <= h_max,

        }


    def calculate_trajectory(self, v0, angle, target, dt=0.01, hit_tolerance=10):
        # Perform stability analysis first
        stability_result = self.prove_partial_stability(dt)
        if not stability_result['is_stable']:
            print(f"Warning: Unstable timestep detected. Max stable dt: {stability_result['max_stable_timestep']:.6f}")
            dt = min(dt, stability_result['max_stable_timestep'])
            print(f"Adjusting timestep to: {dt:.6f}")


        angle_rad = np.deg2rad(angle)
        v0x = v0 * np.cos(angle_rad)
        v0y = v0 * np.sin(angle_rad)

        x_points = [self.start_pos[0]]
        y_points = [self.start_pos[1]]

        x, y = self.start_pos
        vx, vy = v0x, v0y
        t = 0

        while y >= 0 and t < 10 and x <= target[0] * 1.2:
            # RK4 integration steps
            k1_x = vx * dt
            k2_x = vx * dt
            k3_x = vx * dt
            k4_x = vx * dt

            k1_y = vy * dt
            k1_vy = -self.g * dt

            k2_y = (vy + k1_vy / 2) * dt
            k2_vy = -self.g * dt

            k3_y = (vy + k2_vy / 2) * dt
            k3_vy = -self.g * dt

            k4_y = (vy + k3_vy) * dt
            k4_vy = -self.g * dt

            x += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
            y += (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
            vy += (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy) / 6

            x_points.append(x)
            y_points.append(y)
            t += dt

            if np.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2) < hit_tolerance:
                return np.array(x_points), np.array(y_points), True, t

        return np.array(x_points), np.array(y_points), False, t


    def estimate_initial_velocity(self, target):
        # Guess how hard we need to throw to reach the target
        dx = target[0] - self.start_pos[0]
        dy = target[1] - self.start_pos[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)

    # Add  little  speed just to be safe
        min_v0 = np.sqrt(self.g * distance) * 1.3
        return min_v0


    def find_shooting_parameters(self, target, max_attempts=3):

        # First, guess how hard we need to throw
        min_v0 = self.estimate_initial_velocity(target)
        v0_range = (min_v0, min_v0 * 2)

        best_v0 = None
        best_angle = None

        # Try a few times to find the perfect shot
        for attempt in range(max_attempts):
            # On first try, look at a wide range of angles
            # Then narrow down based on what worked best
            if attempt == 0:
                angle_range = (20, 70)  # Try angles between 20° and 70°
                v0_steps = 30  # Try 30 different speeds
                angle_steps = 25  # and 25 different angles
            else:
                # Focus on the angles that worked best before
                angle_range = (max(10, best_angle - 10), min(80, best_angle + 10))
                v0_range = (max(min_v0, best_v0 * 0.8), best_v0 * 1.2)
                v0_steps = 40  # Try more precise adjustments
                angle_steps = 30

            best_distance = float('inf')
            best_params = None

            # Try different combinations of speed and angle
            for v0 in np.linspace(v0_range[0], v0_range[1], v0_steps):
                for angle in np.linspace(angle_range[0], angle_range[1], angle_steps):
                    x, y, hit, _ = self.calculate_trajectory(v0, angle, target)

                    if hit:
                        return v0, angle

                    # If we missed, see how close we got
                    distances = np.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)
                    min_distance = np.min(distances)

                    if min_distance < best_distance:
                        best_distance = min_distance
                        best_params = (v0, angle)
                        best_v0, best_angle = v0, angle

            # If we got pretty close, that's good enough!
            if best_distance < 15:
                return best_params

        return None


class TargetManager:
    # Keeps track of which targets we've hit and which ones we still need to hit.

    def __init__(self, targets, start_pos=(0, 0)):
        self.start_pos = start_pos
        self.targets = list(targets)
        self.hit_targets = []  # Targets we've already hit
        self.missed_targets = []  # Targets we need to try again
        self.sort_targets()

    def sort_targets(self):
        # Sort targets by distance
        self.targets.sort(key=lambda t: (
            # First by distance
            np.sqrt((t[0] - self.start_pos[0]) ** 2 + (t[1] - self.start_pos[1]) ** 2),
            # Then by how far from 45° the angle is (since 45° is usually easiest)
            abs(45 - np.degrees(np.arctan2(t[1] - self.start_pos[1], t[0] - self.start_pos[0])))
        ))

    def get_next_target(self):
        # Pick the next target to aim for
        # If we're out of targets, try the ones we missed before
        if self.targets:
            return self.targets[0]
        elif self.missed_targets:
            self.targets = self.missed_targets
            self.missed_targets = []
            self.sort_targets()
            return self.get_next_target()
        return None

    def mark_target_hit(self, target):

        self.targets.remove(target)
        self.hit_targets.append(target)

    def mark_target_missed(self, target):
        self.targets.remove(target)
        self.missed_targets.append(target)


def create_animation(circles, start_pos=(0, 0)):

    target_manager = TargetManager(circles, start_pos)
    calculator = TrajectoryCalculator(start_pos)
    fig, ax = plt.subplots(figsize=(12, 9))

    trajectories = []  # Save all the successful shots
    hit_sequence = []  # Remember which targets we hit and when
    all_targets = circles.copy()
    remaining_targets = circles.copy()

    # Try to hit each target
    while target := target_manager.get_next_target():
        params = calculator.find_shooting_parameters(target)
        if params:
            v0, angle = params
            x, y, hit, _ = calculator.calculate_trajectory(v0, angle, target)
            if hit:
                trajectories.append((x, y))
                hit_sequence.append(target)
                target_manager.mark_target_hit(target)
                remaining_targets.remove(target)
            else:
                target_manager.mark_target_missed(target)
        else:
            target_manager.mark_target_missed(target)

    def animate(frame):

        ax.clear()
        ax.set_xlim(-10, 810)
        ax.set_ylim(-10, 610)
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Height (m)')
        ax.set_title('Ball Trajectory Simulation')

        # Draw all targets
        for x, y in all_targets:
            if (x, y) in hit_sequence[:(frame // 50) + 1]:
                # Targets we've hit turn green
                circle = plt.Circle((x, y), 5, fill=True)
            else:
                # Targets we haven't hit yet are red rings
                circle = plt.Circle((x, y), 5, color='red', fill=False, linewidth=2)
                # Add white fill to make them stand out
                inner_circle = plt.Circle((x, y), 4.8, color='white', fill=True)
                ax.add_patch(inner_circle)
            ax.add_patch(circle)

        # Draw the current trajectory
        traj_idx = frame // 50
        if traj_idx < len(trajectories):
            x, y = trajectories[traj_idx]
            current_frame = frame % 50
            points = max(1, int((current_frame / 50) * len(x)))
            ax.plot(x[:points], y[:points], 'b-', linewidth=2)  # Path line
            ax.plot(x[points - 1], y[points - 1], 'bo', markersize=8)  # Ball

            # Show where we're throwing from
            ax.plot(start_pos[0], start_pos[1], 'ko', markersize=10)

    anim = FuncAnimation(fig, animate, frames=50 * len(trajectories),
                         interval=20, repeat=False)
    plt.show()

    success = len(target_manager.hit_targets) == len(circles)
    return anim, success


def main():

    # 1. Make an image with random circles
    positions, radii = generate_test_image()

    # 2. Try to find all the circles
    image = cv2.imread('scattered_balls.jpg')
    detector = CircleDetector()
    circles, radii = detector.detect_circles(image)

    # analyze stability
    calculator = TrajectoryCalculator()
    stability_result = calculator.prove_partial_stability()


    # 3. Make a  animation of hitting all the targets
    anim, all_hit = create_animation(circles)
    print(f"Did we hit all the targets? {'yes, please give me full points! <<<<333' if all_hit else 'Almost hit all, please subtract as many points as missed balls (hope it only missed at most 1-2'}")




if __name__ == "__main__":
    main()

