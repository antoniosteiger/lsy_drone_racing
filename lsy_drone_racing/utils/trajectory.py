"""Trajectory Sharing Module.

Explanation:
    This is a global variable that can safely be imported and used for trajectory visualization.
    The visualization thread will read from this variable, while controllers can write to it.
"""

trajectory = None
