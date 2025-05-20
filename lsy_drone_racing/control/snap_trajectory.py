# snap_trajectory.py
import cvxpy as cp
import numpy as np
import math

class SnapTrajectory:
    def __init__(self, degree=3):  # Reduced to cubic polynomial
        self.degree = degree
        self.variables = []
        self.timestamps = []

    def traj(self, waypoints):
        l = len(waypoints)
        if l != 10 or waypoints[4][0] != 0:  # Expect exactly one segment
            print(f'Error! Expected 10 waypoint entries, got {l}, t0={waypoints[4][0]}')
            return False
        xi = np.zeros(5)
        yi = np.zeros(5)
        zi = np.zeros(5)
        psii = np.zeros(3)
        ti = 0.0
        self.variables = []
        self.timestamps = [0.0]
        min_segment_time = 1.0
        xf = np.zeros(5)
        yf = np.zeros(5)
        zf = np.zeros(5)
        psif = np.zeros(3)
        tf = waypoints[9][0]  # Last entry is tf
        for j in range(len(waypoints[5])):
            xf[j] = waypoints[5][j]
        for j in range(len(waypoints[6])):
            yf[j] = waypoints[6][j]
        for j in range(len(waypoints[7])):
            zf[j] = waypoints[7][j]
        for j in range(len(waypoints[8])):
            psif[j] = waypoints[8][j]
        if tf <= ti + min_segment_time:
            tf = ti + min_segment_time
            print(f'Warning: Adjusted tf from {waypoints[9][0]} to {tf} to ensure time increases')
        print(f"Segment: ti={ti}, tf={tf}, xi={xi[0]}, xf={xf[0]}, yi={yi[0]}, yf={yf[0]}, zi={zi[0]}, zf={zf[0]}, psii={psii[0]}, psif={psif[0]}")
        x_v = np.array(self.min_snap(ti, tf, xi, xf))
        y_v = np.array(self.min_snap(ti, tf, yi, yf))
        z_v = np.array(self.min_snap(ti, tf, zi, zf))
        psi_v = np.array(self.min_acc(ti, tf, psii, psif))
        if np.any(np.isnan(x_v)) or np.any(np.isnan(y_v)) or np.any(np.isnan(z_v)) or np.any(np.isnan(psi_v)):
            print("Error! NaN coefficients in trajectory.")
            return False
        self.variables.append(x_v)
        self.variables.append(y_v)
        self.variables.append(z_v)
        self.variables.append(psi_v)
        self.timestamps.append(tf)
        print(f"Trajectory generated with timestamps: {self.timestamps}, variables: {len(self.variables)}")
        return True

    def min_snap(self, ti, tf, wi, wf):
        C = np.zeros((self.degree + 1, self.degree + 1))
        for i in range(self.degree + 1):
            if i >= 2:  # Minimize acceleration for cubic
                C[i, i] = math.factorial(i) / math.factorial(i - 2)
        T_mat = np.zeros((self.degree + 1, self.degree + 1), dtype=float)
        for i in range(2, self.degree + 1):
            for j in range(2, self.degree + 1):
                T_mat[i, j] = 1.0 / ((i - 2) + (j - 2) + 1) * (tf - ti) ** ((i - 2) + (j - 2) + 1)
        t_i = np.zeros(self.degree + 1)
        t_f = np.zeros(self.degree + 1)
        for i in range(self.degree + 1):
            t_i[i] = ti ** i
            t_f[i] = tf ** i
        dt1_i = np.zeros(self.degree + 1)
        dt1_f = np.zeros(self.degree + 1)
        for i in range(1, self.degree + 1):
            dt1_i[i] = i * ti ** (i - 1)
            dt1_f[i] = i * tf ** (i - 1)
        H = cp.Parameter((self.degree + 1, self.degree + 1), PSD=True)
        H.value = C.T @ T_mat @ C + 1e-3 * np.eye(self.degree + 1)
        V = cp.Variable(self.degree + 1)
        objective = cp.Minimize(cp.quad_form(V, H))
        constraints = [
            t_i @ V == wi[0], dt1_i @ V == wi[1],  # Position and velocity
            t_f @ V == wf[0], dt1_f @ V == wf[1]
        ]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=True)
            if V.value is None or np.any(np.isnan(V.value)):
                raise ValueError("Optimization failed or returned NaN")
            print(f"min_snap solved successfully, V={V.value}, wi={wi[0]}, wf={wf[0]}")
            return V.value
        except Exception as e:
            print(f"min_snap failed: {e}, wi={wi}, wf={wf}, ti={ti}, tf={tf}")
            V_fallback = np.zeros(self.degree + 1)
            V_fallback[0] = wi[0]
            delta_t = tf - ti if tf != ti else 1.0
            delta_pos = wf[0] - wi[0]
            V_fallback[1] = delta_pos / delta_t  # Velocity
            V_fallback[2] = 0.2 * delta_pos / delta_t**2  # Acceleration
            print(f"Using fallback cubic trajectory: {V_fallback}")
            return V_fallback

    def min_acc(self, ti, tf, wi, wf):
        C = np.zeros((self.degree + 1, self.degree + 1))
        for i in range(self.degree + 1):
            if i >= 1:  # Minimize jerk for yaw
                C[i, i] = math.factorial(i) // math.factorial(i - 1)
        T_mat = np.zeros((self.degree + 1, self.degree + 1), dtype=float)
        for i in range(1, self.degree + 1):
            for j in range(1, self.degree + 1):
                T_mat[i, j] = 1.0 / ((i - 1) + (j - 1) + 1) * (tf - ti) ** ((i - 1) + (j - 1) + 1)
        t_i = np.zeros(self.degree + 1)
        t_f = np.zeros(self.degree + 1)
        for i in range(self.degree + 1):
            t_i[i] = ti ** i
            t_f[i] = tf ** i
        dt1_i = np.zeros(self.degree + 1)
        dt1_f = np.zeros(self.degree + 1)
        for i in range(1, self.degree + 1):
            dt1_i[i] = i * ti ** (i - 1)
            dt1_f[i] = i * tf ** (i - 1)
        H = cp.Parameter((self.degree + 1, self.degree + 1), PSD=True)
        H.value = C @ T_mat @ C + 1e-3 * np.eye(self.degree + 1)
        V = cp.Variable(self.degree + 1)
        objective = cp.Minimize(cp.quad_form(V, H))
        constraints = [
            t_i @ V == wi[0], dt1_i @ V == wi[1],
            t_f @ V == wf[0], dt1_f @ V == wf[1]
        ]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.ECOS, verbose=True)
            if V.value is None or np.any(np.isnan(V.value)):
                raise ValueError("Optimization failed or returned NaN")
            print(f"min_acc solved successfully, V={V.value}")
            return V.value
        except Exception as e:
            print(f"min_acc failed: {e}")
            V_fallback = np.zeros(self.degree + 1)
            V_fallback[0] = wi[0]
            delta_t = tf - ti if tf != ti else 1.0
            V_fallback[1] = (wf[0] - wi[0]) / delta_t
            print(f"Using fallback linear trajectory: {V_fallback}")
            return V_fallback

    def evaluate(self, t, segment_idx):
        if segment_idx >= len(self.variables) // 4 or t < self.timestamps[segment_idx] or t > self.timestamps[segment_idx + 1]:
            print(f"Evaluation failed: t={t}, segment_idx={segment_idx}, timestamps={self.timestamps}")
            return None, None, None, None
        t_rel = t - self.timestamps[segment_idx]
        x_coeff = self.variables[4 * segment_idx]
        y_coeff = self.variables[4 * segment_idx + 1]
        z_coeff = self.variables[4 * segment_idx + 2]
        psi_coeff = self.variables[4 * segment_idx + 3]
        
        def poly_eval(coeff, t, deriv=0):
            if deriv == 0:
                powers = [t ** i for i in range(len(coeff))]
            elif deriv == 1:
                powers = [i * t ** (i - 1) if i >= 1 else 0 for i in range(len(coeff))]
            elif deriv == 2:
                powers = [i * (i - 1) * t ** (i - 2) if i >= 2 else 0 for i in range(len(coeff))]
            else:
                return 0
            return np.dot(coeff, powers)
        
        x = poly_eval(x_coeff, t_rel, 0)
        vx = poly_eval(x_coeff, t_rel, 1)
        ax = poly_eval(x_coeff, t_rel, 2)
        y = poly_eval(y_coeff, t_rel, 0)
        vy = poly_eval(y_coeff, t_rel, 1)
        ay = poly_eval(y_coeff, t_rel, 2)
        z = poly_eval(z_coeff, t_rel, 0)
        vz = poly_eval(z_coeff, t_rel, 1)
        az = poly_eval(z_coeff, t_rel, 2)
        psi = poly_eval(psi_coeff, t_rel, 0)
        print(f"Evaluated t={t}: p=[{x:.3f}, {y:.3f}, {z:.3f}], v=[{vx:.3f}, {vy:.3f}, {vz:.3f}], a=[{ax:.3f}, {ay:.3f}, {az:.3f}], psi={psi:.3f}")
        return np.array([x, y, z]), np.array([vx, vy, vz]), np.array([ax, ay, az]), psi