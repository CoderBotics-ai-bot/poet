from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from poet.architectures.bert import BERTBase
from poet.architectures.linear import make_linear_network
from poet.architectures.resnet import resnet18, resnet18_cifar, resnet50
from poet.architectures.vgg import vgg16
from poet.chipsets import M4F, MKR1000, JetsonTX2, RPi, RPiNoCache
from poet.poet_solver import POETSolution
from poet.power_computation import DNNLayer, GradientLayer, get_net_costs
from poet.utils.checkmate.core.dfgraph import DFGraph
from poet.utils.checkmate.core.graph_builder import GraphBuilder
from poet.utils.checkmate.core.utils.definitions import PathLike
from poet.utils.checkmate.plot.graph_plotting import plot_dfgraph
from typing import List, Tuple
import warnings
from typing import Tuple


@dataclass
class POETResult:
    ram_budget: float
    runtime_budget_ms: float
    paging: bool
    remat: bool
    total_power_cost_page: float
    total_power_cost_cpu: float
    total_runtime: float
    feasible: bool
    solution: POETSolution


def save_network_repr(net: List[DNNLayer], readable_path: PathLike = None, pickle_path: PathLike = None):
    if readable_path is not None:
        with Path(readable_path).open("w") as f:
            for layer in net:
                f.write("{}\n".format(layer))
    if pickle_path is not None:
        with Path(pickle_path).open("wb") as f:
            pickle.dump(net, f)



def make_dfgraph_costs(
    net: List[DNNLayer], device: str
) -> Tuple[DFGraph, np.ndarray, np.ndarray, np.ndarray]:
    """Computes the power costs for each layer in a given network using the specified device.

    Args:
        net (List[DNNLayer]): The DNN layer network.
        device (str): The device on which the power costs will be computed.

    Returns:
        Tuple[DFGraph, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following:
            - DFGraph: The computational graph representing the network.
            - np.ndarray: The compute costs for each layer.
            - np.ndarray: The page in costs for each layer.
            - np.ndarray: The page out costs for each layer.
    """

    # Helper function to get layer name
    def get_layer_name(layer, idx):
        return f"layer{idx}_{layer.__class__.__name__}"

    # Create graph builder and initialize dicts
    gb = GraphBuilder()
    compute_costs, page_in_costs, page_out_costs = [], [], []
    layer_names = {}

    # Iterate over network layers and compute power costs
    for idx, layer in enumerate(net):
        layer_name = get_layer_name(layer, idx)
        (
            runtime_ms,
            memory_bytes,
            param_memory_bytes,
            pagein_cost_joules,
            pageout_cost_joules,
            compute_cost_joules,
        ) = get_net_costs([layer], device)[0]

        # Add node to graph builder
        gb.add_node(
            layer_name,
            cpu_cost=runtime_ms,
            ram_cost=memory_bytes,
            backward=isinstance(layer, GradientLayer),
        )
        gb.set_parameter_cost(gb.parameter_cost + param_memory_bytes)

        # Update dictionaries
        layer_names[layer] = layer_name
        compute_costs.append(compute_cost_joules)
        page_in_costs.append(pagein_cost_joules)
        page_out_costs.append(pageout_cost_joules)

        # Add dependencies to graph builder
        for dep in layer.depends_on:
            gb.add_deps(layer_name, layer_names[dep])

    # Make graph from the graph builder
    g = gb.make_graph()

    # Get ordered names
    ordered_names = [(topo_idx, name) for topo_idx, name in g.node_names.items()]
    ordered_names.sort(key=lambda x: x[0])
    ordered_names = [x for _, x in ordered_names]

    # Convert lists to numpy arrays
    compute_costs = np.asarray(compute_costs).reshape((-1, 1))
    page_in_costs = np.asarray(page_in_costs).reshape((-1, 1))
    page_out_costs = np.asarray(page_out_costs).reshape((-1, 1))

    return g, compute_costs, page_in_costs, page_out_costs


def extract_costs_from_dfgraph(g: DFGraph, sd_card_multipler=5.0):
    T = g.size
    cpu_cost_vec = np.asarray([g.cost_cpu[i] for i in range(T)])[np.newaxis, :].T
    page_in_cost_vec = cpu_cost_vec * sd_card_multipler
    page_out_cost_vec = cpu_cost_vec * sd_card_multipler
    return cpu_cost_vec, page_in_cost_vec, page_out_cost_vec



def get_chipset(platform: str) -> dict:
    """
    Returns the chipset for the given platform.

    Args:
        platform (str): The platform name. Supported platforms are: "m0", "a72", "a72nocache", "m4", "jetsontx2".

    Returns:
        dict: The chipset dictionary.

    Raises:
        NotImplementedError: If the platform is not supported.
    """
    platforms = {
        "m0": MKR1000,
        "a72": RPi,
        "a72nocache": RPiNoCache,
        "m4": M4F,
        "jetsontx2": JetsonTX2,
    }

    chipset = platforms.get(platform)
    if chipset is None:
        raise NotImplementedError("Unsupported platform: {}".format(platform))

    return chipset


def get_network(
    model: str, batch_size: int, type: Tuple[int, int, int] = (3, 32, 32)
) -> object:
    """
    Returns the network object for the given model.

    Args:
        model (str): The model name. Supported models are: "linear", "vgg16", "vgg16_cifar", "resnet18", "resnet50", "resnet18_cifar", "bert", "transformer".
        batch_size (int): The batch size.
        type (Tuple[int, int, int], optional): The input tensor shape. Defaults to (3, 32, 32).

    Returns:
        object: The network object.

    Raises:
        NotImplementedError: If the model is not supported.
    """
    models = {
        "linear": make_linear_network,
        "vgg16": lambda: vgg16(batch_size),
        "vgg16_cifar": lambda: vgg16(batch_size, 10, type),
        "resnet18": lambda: resnet18(batch_size),
        "resnet50": lambda: resnet50(batch_size),
        "resnet18_cifar": lambda: resnet18_cifar(batch_size, 10, type),
        "bert": lambda: BERTBase(
            SEQ_LEN=512, HIDDEN_DIM=768, I=64, HEADS=12, NUM_TRANSFORMER_BLOCKS=12
        ),
        "transformer": lambda: BERTBase(
            SEQ_LEN=512, HIDDEN_DIM=768, I=64, HEADS=12, NUM_TRANSFORMER_BLOCKS=1
        ),
    }

    network = models.get(model)
    if network is None:
        raise NotImplementedError("Unsupported model: {}".format(model))

    return network()


def get_chipset_and_net(
    platform: str, model: str, batch_size: int, mem_power_scale: float = 1.0
) -> Tuple[dict, object]:
    """
    Returns the chipset and network for the given platform and model.

    Args:
        platform (str): The platform name. Supported platforms are: "m0", "a72", "a72nocache", "m4", "jetsontx2".
        model (str): The model name. Supported models are: "linear", "vgg16", "vgg16_cifar", "resnet18", "resnet50", "resnet18_cifar", "bert", "transformer".
        batch_size (int): The batch size.
        mem_power_scale (float, optional): The memory power scale. Defaults to 1.0.

    Returns:
        Tuple[dict, object]: A tuple containing the chipset dictionary and the network object.

    Raises:
        NotImplementedError: If the platform or model is not supported.
    """
    chipset = get_chipset(platform)
    chipset["MEMORY_POWER"] *= mem_power_scale

    network = get_network(model, batch_size)
    return chipset, network


def plot_network(
    platform: str, model: str, directory: str, batch_size: int = 1, mem_power_scale: float = 1.0, format="pdf", quiet=True, name=""
):
    chipset, net = get_chipset_and_net(platform, model, batch_size, mem_power_scale)
    g, *_ = make_dfgraph_costs(net, chipset)
    plot_dfgraph(g, directory, format, quiet, name)


def print_result(result: POETResult):
    solution = result.solution
    if solution.feasible:
        solution_msg = "successfully found an optimal solution" if solution.finished else "found a feasible solution"
        print(
            f"POET {solution_msg} with a memory budget of {result.ram_budget} bytes that consumes {result.total_power_cost_cpu:.5f} J of CPU power and {result.total_power_cost_page:.5f} J of memory paging power"
        )
        if not solution.finished:
            print("This solution is not guaranteed to be optimal - you can try increasing the time limit to find an optimal solution")

        plt.matshow(solution.R)
        plt.title("R")
        plt.show()

        plt.matshow(solution.SRam)
        plt.title("SRam")
        plt.show()

        plt.matshow(solution.SSd)
        plt.title("SSd")
        plt.show()
    elif solution.finished:
        print(
            "The problem is infeasible since the provided memory budget and/or runtime budget are too small to run the network on the given platform."
        )
    else:
        print(
            "POET failed to find a feasible solution within the provided time limit. \n Either a) increase the memory and training time budgets, and/or b) increase the solve time (time_limit_s)"
        )
