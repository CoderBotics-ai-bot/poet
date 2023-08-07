from build.lib.poet.poet_solver_gurobi import *
from poet.poet_solver_gurobi import *
import pytest
from poet.poet_solver_gurobi import POETSolverGurobi
import numpy as np
from typing import Optional
from poet.utils.checkmate.core.dfgraph import DFGraph
from poet.utils.checkmate.core.dfgraph import Vertex


# Creating fixture for DFGraph instance
@pytest.fixture
def dfgraph():
    # Create three mock vertexes
    v0, v1, v2 = [Vertex(uid) for uid in range(3)]
    vertices = [v0, v1, v2]
    args = {v0: [v1, v2], v1: [], v2: [v1]}
    return DFGraph(args=args, v=vertices)


# Prepare fixture for numpy arrays
@pytest.fixture
def np_arrays():
    cpu_power_cost_vec_joule = np.ones((3, 1))
    pagein_power_cost_vec_joule = np.ones((3, 1)) * 2
    pageout_power_cost_vec_joule = np.ones((3, 1)) * 3
    return (
        cpu_power_cost_vec_joule,
        pagein_power_cost_vec_joule,
        pageout_power_cost_vec_joule,
    )


# Test case to check time_limit_s parameter during initialization
# excluding edge case of None value which raises error in this case
@pytest.mark.parametrize("time_limit_s", [100.0, 200.0])
def test_init_time_limit_s(dfgraph, np_arrays, time_limit_s):
    (
        cpu_power_cost_vec_joule,
        pagein_power_cost_vec_joule,
        pageout_power_cost_vec_joule,
    ) = np_arrays
    solver = POETSolverGurobi(
        g=dfgraph,
        cpu_power_cost_vec_joule=cpu_power_cost_vec_joule,
        pagein_power_cost_vec_joule=pagein_power_cost_vec_joule,
        pageout_power_cost_vec_joule=pageout_power_cost_vec_joule,
        time_limit_s=time_limit_s,
    )
    assert solver.m.Params.TimeLimit == time_limit_s


# Test case to check solve_threads parameter during initialization
@pytest.mark.parametrize("solve_threads", [1, 2, None])
def test_init_solve_threads(dfgraph, np_arrays, solve_threads):
    (
        cpu_power_cost_vec_joule,
        pagein_power_cost_vec_joule,
        pageout_power_cost_vec_joule,
    ) = np_arrays
    solver = POETSolverGurobi(
        g=dfgraph,
        cpu_power_cost_vec_joule=cpu_power_cost_vec_joule,
        pagein_power_cost_vec_joule=pagein_power_cost_vec_joule,
        pageout_power_cost_vec_joule=pageout_power_cost_vec_joule,
        solve_threads=solve_threads,
    )
    assert solver.m.Params.Threads == (solve_threads if solve_threads else 0)
