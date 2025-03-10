introduction
This document presents an explanation of the Numerical Programming Final, project 1, ‘hit a
ball at fixed target’. program combines computer vision, simulation, and numerical methods
to hit detected targets. The system uses numerical techniques to ensure stability and
accuracy while solving the differential equations that describe the balls' motion.
Program Explanation
The program starts with the function that generates sample test cases to check
performance of the program. You can modify number of balls scattered manually by passing
increased or decreased number as a paremeter. So, if you would like to test my program
using randomly generated test cases, you can uncomment
‘positions, radii = generate_test_image()’
from the main function and pass ‘scattered_balls.jpg’ (result of generated image) to
cv2.imread next line.Otherwise, if you have other test data just pass it into to cv2.imread instead of
‘scattered_balls.jpg’
I wont get into depths of regular functions since the task only asks to describe numerical
methods.
After generating test cases, we have circle detection function, which has nearly same logic
as the one edge detection function used in the CP1. It is fully written by hand from scratch
so no built in functions are used. To generally ovrewview that function: first we conert color
image into gray scale, before we start edge detection we smooth the grayscaled image using
kernel. Edge detection is done by sobels, in both directions. Then we perform dbscan, where
we group by clusters and perfom cluserization,( dbscan was used in CP1 so I wont get into
depths of how it works). After dbscan we verify if detected shapes are circles and save
found balls location and size.
Trajectory calculation
After detecting the targets, we move onto the main part of this program, calculating
trajectory to throw the ball at fixed targets.
Before we start we check for A - stability (this part was added 2 days before the deadline,
since we were informed late, so please if there is misunderstanding or little errors do not
subtract too many points, since I didn’t have enough time to work 100% on it). We know that
RK4 is explicit method which can not be A stable, so we have to prove that it is partially
stable. code implements a partial stability analysis for the Runge-Kutta 4 method (RK4). This
function creates Jacobian matrix of system. It represents how each variable affects the rate
of change of the others. The structure shows that position changes are affected by velocity
(the 1's in the first two rows), Vertical velocity is affected by gravity (the -g term), The
system is linear when we ignore air resistance
def rk4_stability_function(self, z):
return 1 + z + (z ** 2) / 2 + (z ** 3) / 6 + (z ** 4) / 24
This is the actual stability polynomial for the RK4 method. When we apply RK4 to a linear
system, this polynomial determines whether our numerical solution will remain bounded.
The main stability analysis happens in ‘prove_partial_stability’. First, it finds the eigenvalues
of the system matrix. In this case, they're all zero because we're looking at a conservative
system.
For each component (x, y, vx, vy), it analyzes stability by:
• Computing the stability measure using the RK4 stability function
• Determining if that component is stable
• Classifying the type of stability (here, "neutrally stable")
The method produces a detailed stability report that includes:• The eigenvalues of the system
• Analysis of each component's stability
• A practical maximum timestep recommendation
• A formal proof explanation
Output shows that while our system is mathematically stable, we still need to consider
practical limitations for accuracy. That's why there's a conservative maximum timestep of
0.1 seconds.
Now, we move onto calculating trajectory. This function implements the Runge-Kutta 4
method to simulate the ball's path through the air. Before we even start calculating, the
function checks if our chosen time step (dt) will give us stable results. This is crucial because
unstable time steps can cause our numerical solution to explode. If our time step is too
large, we automatically adjust it to stay within stable bounds.
Next, we convert our initial velocity and angle into x and y components. This converts the
polar coordinates (speed and angle) into Cartesian coordinates (x and y components) that
we can use in our calculations.
RK4 For the x-direction
k1_x = vx * dt
k2_x = vx * dt
k3_x = vx * dt
k4_x = vx * dt
These terms are all the same because there's no acceleration in the x-direction. The
horizontal velocity remains constant.
For the y-direction, it's more complex because of gravity
k1_y = vy * dt
k1_vy = -self.g * dt
k2_y = (vy + k1_vy / 2) * dt
k2_vy = -self.g * dt
k3_y = (vy + k2_vy / 2) * dt
k3_vy = -self.g * dt
k4_y = (vy + k3_vy) * dt
k4_vy = -self.g * dt
These terms represent the four evaluations of the function at different points that make
RK4 so accurate. Each k-term is using the previous calculations to estimate the next position
and velocity more accurately.Finally, we update our position and velocity:
x += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
y += (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
vy += (k1_vy + 2 * k2_vy + 2 * k3_vy + k4_vy) / 6
This is the classic RK4 weighted average, where the middle terms (k2 and k3) are weighted
more heavily than the endpoint terms (k1 and k4).
The hit detection is handled by checking if we're within our hit_tolerance distance from the
target using the Pythagorean theorem.
Now we have to estimate initial velocity and finally throw ball at the targets using shooting
method.
‘estimate_initial_velocity’ function makes an initial guess at how hard we need to throw. It
uses approximation: the minimum velocity needed to reach a target at distance d is
approximately √(g*d). We multiply by 1.3 to give ourselves a little extra speed, ensuring we
can reach the target even if our first guess isn't perfect.
The main algorithm is in find_shooting_parameters.
First, we set up our search range. We'll look at speeds between our minimum estimate and
twice that value. The search happens in multiple attempts, each time getting more precise.
First attempt: Look broadly, trying many different combinations. Later attempts: Focus on
the most promising area, looking more carefully
for v0 in np.linspace(v0_range[0], v0_range[1], v0_steps):
for angle in np.linspace(angle_range[0], angle_range[1], angle_steps):
x, y, hit, _ = self.calculate_trajectory(v0, angle, target)
For each combination of speed and angle:
1. Calculate the complete trajectory
2. Check if we hit the target
3. If we missed, measure how close we got
The code keeps track of the best attempt.
• It starts with a guess to save computation
• It uses multiple passes, refining its search each time
• It remembers what worked best and focuses there
• It has a "good enough" threshold to avoid excessive precision
To cut a long story short, it starts with a reasonable guess, see what happens, then adjusts
based on how close it got.Now we move onto target management and animation of ball-throwing simulation. First,
the TargetManager class handles how we organize and track our targets. Think of it like a
strategic planner that decides which target to aim for next. It sorts targets in a clever way -
first by distance (closer ones first) and then by angle (preferring 45-degree angles since
they're typically easiest to hit). It also keeps track of which targets we've hit and which ones
we need to try again, much like a game keeping score.
The second major component is the create_animation function, which brings everything
together to create a visual simulation. This function works like a movie director - it plans out
all the shots first, then creates an animation showing each trajectory one by one. The
animation shows red circles for targets we haven't hit yet, and they turn green once we
successfully hit them. The blue line shows the path of the ball, with a blue dot representing
the ball itself.
