"""FastAPI demo server for the Federated Learning Aircraft Engine PHM project.

Exposes the trained-checkpoint zoo + RQ3 explanations as a small HTTP API the
React frontend consumes. The science layer lives entirely in
``src/fl_aircraft/`` — this package is a thin wrapper.
"""
