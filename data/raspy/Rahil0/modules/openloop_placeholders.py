"""Minimal placeholders for open-loop runs.

This module provides the arrays SJ_4_directions expects without running a
decoder or Kalman filter. It is safe in open-loop because labels come from
the task's own state machine (state_task), not from decoder outputs.
"""

# Fill a 5-way output vector (RLUD + still) with zeros
decoder_output[:] = 0.0

# Provide a 7-dim KF state with constant bias term = 1.0
kf_state[:] = 0.0
kf_state[-1] = 1.0

