# Numerical Programming Final - Project 1: Hit a Ball at a Fixed Target

## Introduction
This document explains Project 1 of the Numerical Programming Final: "Hit a Ball at a Fixed Target." The program combines computer vision, simulation, and numerical methods to ensure stable and accurate calculations for hitting detected targets. The system solves differential equations describing the ball's motion using numerical techniques.

## Program Explanation
The program starts by generating sample test cases to evaluate performance. The number of scattered balls can be manually modified by adjusting a parameter. To use randomly generated test cases, uncomment the following line in the main function:
```python
positions, radii = generate_test_image()
```
and pass `'scattered_balls.jpg'` (the generated image) to `cv2.imread` in the next line. Otherwise, provide any other test data to `cv2.imread`.

The program includes a circle detection function, developed from scratch without built-in functions. The function follows these steps:
1. Convert the color image to grayscale.
2. Apply smoothing using a kernel before edge detection.
3. Detect edges using Sobel operators in both directions.
4. Perform DBSCAN clustering to group edges into potential circles.
5. Verify detected shapes and store valid ball locations and sizes.

## Trajectory Calculation
Once targets are detected, the program calculates the trajectory required to hit them. 

### Stability Analysis
Before trajectory computation, the program verifies A-stability. Since RK4 is an explicit method and not A-stable, a partial stability analysis is performed. The system computes the Jacobian matrix to assess stability:
```python
def rk4_stability_function(self, z):
    return 1 + z + (z ** 2) / 2 + (z ** 3) / 6 + (z ** 4) / 24
```
The `prove_partial_stability` function:
- Finds eigenvalues of the system matrix.
- Computes stability measures using the RK4 stability function.
- Classifies stability type (e.g., "neutrally stable").
- Recommends a practical maximum timestep (0.1 seconds).

### RK4 Implementation
The program simulates the ball's path using the RK4 method. It first ensures that the chosen timestep (`dt`) is stable. If `dt` is too large, it is adjusted to remain within stable bounds.

RK4 calculations for the x-direction:
```python
k1_x = vx * dt
k2_x = vx * dt
k3_x = vx * dt
k4_x = vx * dt
```
The x-direction remains constant due to no horizontal acceleration.

RK4 calculations for the y-direction (including gravity):
```python
k1_y = vy * dt
k1_vy = -self.g * dt
k2_y = (vy + k1_vy / 2) * dt
k2_vy = -self.g * dt
k3_y = (vy + k2_vy / 2) * dt
k3_vy = -self.g * dt
k4_y = (vy + k3_vy) * dt
k4_vy = -self.g * dt
```
Final position and velocity updates:
```python
x += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
y += (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
vy += (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy) / 6
```
Hit detection is performed by checking if the ball is within a specified `hit_tolerance` distance from the target.

## Estimating Initial Velocity
The function `estimate_initial_velocity` provides an initial velocity guess using:
```python
v_min = sqrt(g * d) * 1.3
```
This ensures the ball reaches the target, even if the initial estimate is not perfect.

The `find_shooting_parameters` function searches for optimal speed and angle by iterating through potential values:
```python
for v0 in np.linspace(v0_range[0], v0_range[1], v0_steps):
    for angle in np.linspace(angle_range[0], angle_range[1], angle_steps):
        x, y, hit, _ = self.calculate_trajectory(v0, angle, target)
```
This process refines guesses progressively to improve accuracy.

## Target Management and Animation
The `TargetManager` class organizes and tracks targets. It prioritizes targets based on:
1. Distance (closer targets first).
2. Angle (preferring 45-degree angles for easier hits).
3. Hit/miss status.

The `create_animation` function visualizes the simulation:
- Red circles represent unhit targets.
- Green circles represent hit targets.
- A blue line shows the ball's trajectory.
- A blue dot represents the moving ball.

This structured approach ensures accurate target detection, trajectory calculation, and visualization of results.
