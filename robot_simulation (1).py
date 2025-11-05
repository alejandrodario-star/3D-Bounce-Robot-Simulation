"""
3D Bounce Robot Simulation + 3D Visualization
Author: Your Name
Description:
Simulates an underwater robot moving in 3D (x, y, z) with 90° clockwise turns
on impact. Includes an optional real-time 3D visualization using matplotlib.

Controls (when running this file directly):
- Press 'q' to quit the animation window.
- Change PARAMETERS in __main__ to tweak behavior.
"""

from __future__ import annotations
import numpy as np

# Optional visualization imports are done lazily inside visualize_3d()

# ---------------- Core Math ---------------- #

def vel_from_angles(speed: float, theta: float, phi: float) -> np.ndarray:
    """Velocity components from yaw (theta) and elevation (phi)."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    return speed * np.array([cphi*cth, cphi*sth, sphi], dtype=float)


def next_hit_time_and_normal(p: np.ndarray, v: np.ndarray,
                             box_min: np.ndarray, box_max: np.ndarray,
                             r: float = 0.0, eps: float = 1e-12):
    """Return (t_hit, normal) for axis-aligned box. If no hit, (inf, None)."""
    t_candidates = []
    normals = []
    # x faces
    if v[0] >  eps:
        t = (box_max[0]-r - p[0]) / v[0]; t_candidates.append(t); normals.append(np.array([+1,0,0.]))
    elif v[0] < -eps:
        t = (box_min[0]+r - p[0]) / v[0]; t_candidates.append(t); normals.append(np.array([-1,0,0.]))
    # y faces
    if v[1] >  eps:
        t = (box_max[1]-r - p[1]) / v[1]; t_candidates.append(t); normals.append(np.array([0,+1,0.]))
    elif v[1] < -eps:
        t = (box_min[1]+r - p[1]) / v[1]; t_candidates.append(t); normals.append(np.array([0,-1,0.]))
    # z faces
    if v[2] >  eps:
        t = (box_max[2]-r - p[2]) / v[2]; t_candidates.append(t); normals.append(np.array([0,0,+1.]))
    elif v[2] < -eps:
        t = (box_min[2]+r - p[2]) / v[2]; t_candidates.append(t); normals.append(np.array([0,0,-1.]))

    t_pos = [(ti, n) for ti, n in zip(t_candidates, normals) if ti > 0]
    if not t_pos:
        return np.inf, None
    t_star, n_star = min(t_pos, key=lambda x: x[0])
    return t_star, n_star


def wrap_pi(a: float) -> float:
    """wrap angle to (-pi, pi]."""
    return (a + np.pi) % (2*np.pi) - np.pi

# ---------------- Bounce Rules ---------------- #

def bounce_rule_A(v: np.ndarray, n: np.ndarray, clockwise_sign: int = +1) -> np.ndarray:
    """90° turn in the tangent plane everywhere (flip normal, rotate tangent by 90°)."""
    vn = np.dot(v, n) * n
    t  = v - vn
    return -vn + clockwise_sign * np.cross(n, t)


def bounce_rule_B_angles(theta: float, phi: float, n: np.ndarray,
                          yaw_turn: float = +np.pi/2):
    """Walls: yaw += 90° (clockwise via sign); floor/ceiling: reflect elevation."""
    if np.allclose(np.abs(n), np.array([0,0,1.])):  # hit floor or ceiling
        return theta, -phi
    else:  # vertical wall
        return wrap_pi(theta + yaw_turn), phi

# ---------------- Simulation API ---------------- #

def step_to_next_hit(p: np.ndarray, theta: float, phi: float, speed: float,
                     box_min: np.ndarray, box_max: np.ndarray, r: float = 0.0,
                     rule: str = "B", clockwise_sign: int = +1):
    """Advance to next collision and update direction per rule ('A' or 'B')."""
    v = vel_from_angles(speed, theta, phi)
    t_hit, n = next_hit_time_and_normal(p, v, box_min, box_max, r)
    if not np.isfinite(t_hit):
        # No hit (degenerate). Move a small step and keep direction.
        return p + v*1e-3, theta, phi
    p_new = p + t_hit * v
    if rule.upper() == "A":
        v_new = bounce_rule_A(v, n, clockwise_sign)
        speed_new = np.linalg.norm(v_new)
        theta_new = np.arctan2(v_new[1], v_new[0])
        phi_new   = np.arctan2(v_new[2], np.hypot(v_new[0], v_new[1]))
        return p_new, theta_new, phi_new
    else:  # Rule B (wall=+90° yaw; floor/ceiling reflect elevation)
        theta_new, phi_new = bounce_rule_B_angles(theta, phi, n,
                                                  yaw_turn=clockwise_sign*np.pi/2)
        return p_new, theta_new, phi_new


def simulate_impacts(num_impacts: int = 100,
                      speed: float = 1.0,
                      theta0: float = 0.6,
                      phi0: float = 0.3,
                      box_min: np.ndarray = None,
                      box_max: np.ndarray = None,
                      r: float = 0.0,
                      rule: str = "B",
                      clockwise_sign: int = +1):
    """Simulate impacts and return arrays of positions (N+1,3) and angles.
    Positions include the start and every impact point.
    """
    if box_min is None: box_min = np.array([-3., -3., -3.])
    if box_max is None: box_max = np.array([+3., +3., +3.])

    p = np.array([0., 0., 0.], dtype=float)
    theta, phi = float(theta0), float(phi0)

    positions = [p.copy()]
    thetas = [theta]; phis = [phi]

    for _ in range(num_impacts):
        p, theta, phi = step_to_next_hit(p, theta, phi, speed, box_min, box_max, r,
                                         rule=rule, clockwise_sign=clockwise_sign)
        positions.append(p.copy())
        thetas.append(theta); phis.append(phi)

    return np.vstack(positions), np.array(thetas), np.array(phis)


def simulate_print(robot_speed: float = 1.0, steps: int = 10):
    """Text-mode demo: print impact points & updated angles."""
    box_min = np.array([-3., -3., -3.])
    box_max = np.array([+3., +3., +3.])
    pos, th, ph = simulate_impacts(num_impacts=steps,
                                   speed=robot_speed,
                                   theta0=0.5, phi0=0.3,
                                   box_min=box_min, box_max=box_max,
                                   rule="B", clockwise_sign=+1)
    print(f"Initial position: {pos[0]}")
    for i in range(1, len(pos)):
        print(f"Hit {i:>3}: pos={np.round(pos[i], 3)}, θ={th[i]:.3f}, φ={ph[i]:.3f}")

# ---------------- Visualization ---------------- #

def visualize_3d(num_impacts: int = 200, speed: float = 1.0,
                  theta0: float = 0.6, phi0: float = 0.25,
                  box_min: np.ndarray = None, box_max: np.ndarray = None,
                  r: float = 0.0, rule: str = "B", clockwise_sign: int = +1,
                  trail_len: int = 200):
    """Animate the robot bouncing inside the box with matplotlib 3D."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D proj)

    if box_min is None: box_min = np.array([-3., -3., -3.])
    if box_max is None: box_max = np.array([+3., +3., +3.])

    positions, _, _ = simulate_impacts(num_impacts=num_impacts, speed=speed,
                                       theta0=theta0, phi0=phi0,
                                       box_min=box_min, box_max=box_max,
                                       r=r, rule=rule, clockwise_sign=clockwise_sign)

    # Set up figure and axes
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(box_min[0], box_max[0])
    ax.set_ylim(box_min[1], box_max[1])
    ax.set_zlim(box_min[2], box_max[2])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D Bounce Robot')

    # Draw wireframe cube
    def cube_lines(bmin, bmax):
        xs = [bmin[0], bmax[0]]; ys = [bmin[1], bmax[1]]; zs = [bmin[2], bmax[2]]
        segs = []
        for x in xs:
            for y in ys:
                segs.append(((x,y,zs[0]), (x,y,zs[1])))
        for x in xs:
            for z in zs:
                segs.append(((x,ys[0],z), (x,ys[1],z)))
        for y in ys:
            for z in zs:
                segs.append(((xs[0],y,z), (xs[1],y,z)))
        return segs

    for a,b in cube_lines(box_min, box_max):
        ax.plot([a[0],b[0]], [a[1],b[1]], [a[2],b[2]], linewidth=0.8)

    # Moving point and trail
    point_plot, = ax.plot([positions[0,0]], [positions[0,1]], [positions[0,2]],
                          marker='o', markersize=6, linestyle='None')
    trail_plot, = ax.plot([], [], [], linewidth=1.6)

    def update(frame):
        i0 = max(0, frame-trail_len)
        seg = positions[i0:frame+1]
        point_plot.set_data([seg[-1,0]], [seg[-1,1]])
        point_plot.set_3d_properties([seg[-1,2]])
        trail_plot.set_data(seg[:,0], seg[:,1])
        trail_plot.set_3d_properties(seg[:,2])
        return point_plot, trail_plot

    anim = FuncAnimation(fig, update, frames=len(positions), interval=40, blit=True)
    plt.tight_layout()
    plt.show()


# ---------------- Entrypoint ---------------- #
if __name__ == "__main__":
    # PARAMETERS
    SPEED = 1.0
    RULE = "B"           # "A" = 90° everywhere; "B" = wall-90 / floor reflect
    CLOCKWISE = +1        # set to -1 to reverse the 90° direction
    STEPS_PRINT = 8       # text-mode preview of impacts

    # Print a small log to console
    simulate_print(robot_speed=SPEED, steps=STEPS_PRINT)

    # Launch the 3D viewer
    try:
        visualize_3d(num_impacts=250, speed=SPEED, theta0=0.7, phi0=0.28,
                      rule=RULE, clockwise_sign=CLOCKWISE, trail_len=220)
    except Exception as e:
        print("Visualization error (matplotlib may be missing):", e)
