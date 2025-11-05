"""
3D Bounce Robot Simulation
Author: Your Name
Description:
Simulates an underwater robot moving in 3D space (x, y, z)
with 90° clockwise turns upon collision with the workspace walls.
"""

import numpy as np

# ---------------- Core Functions ---------------- #

def vel_from_angles(speed, theta, phi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    return speed * np.array([cphi * cth, cphi * sth, sphi])

def next_hit_time_and_normal(p, v, box_min, box_max, r=0.0):
    eps = 1e-12
    t_candidates = []
    normals = []
    for i in range(3):
        if v[i] > eps:
            t = (box_max[i] - r - p[i]) / v[i]
            n = np.zeros(3); n[i] = +1
            t_candidates.append((t, n))
        elif v[i] < -eps:
            t = (box_min[i] + r - p[i]) / v[i]
            n = np.zeros(3); n[i] = -1
            t_candidates.append((t, n))
    hits = [(t, n) for t, n in t_candidates if t > 0]
    return min(hits, key=lambda x: x[0]) if hits else (np.inf, None)

def bounce_rule_B_angles(theta, phi, n, yaw_turn=np.pi/2):
    if np.allclose(np.abs(n), [0, 0, 1]):
        return theta, -phi
    return (theta + yaw_turn) % (2 * np.pi), phi

def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

# ---------------- Simulation Loop ---------------- #

def simulate(robot_speed=1.0, steps=10):
    box_min = np.array([-3., -3., -3.])
    box_max = np.array([+3., +3., +3.])
    p = np.array([0., 0., 0.])
    theta, phi = 0.5, 0.3  # yaw and elevation in radians

    print(f"Initial position: {p}, θ={theta:.2f}, φ={phi:.2f}")
    for step in range(steps):
        v = vel_from_angles(robot_speed, theta, phi)
        t_hit, n = next_hit_time_and_normal(p, v, box_min, box_max)
        p = p + v * t_hit
        theta, phi = bounce_rule_B_angles(theta, phi, n)
        print(f"Step {step+1}: pos={p.round(2)}, θ={theta:.2f}, φ={phi:.2f}")
    print("Simulation complete.")

if __name__ == "__main__":
    simulate()
