import pathlib
from typing import Optional

import numpy as np
from graphviz import Digraph

from poet.utils.checkmate.core.dfgraph import DFGraph



def plot_dfgraph(
    g: DFGraph, directory: str, format: str = "pdf", quiet: bool = True, name: str = ""
) -> None:
    """
    Generate a Graphviz-formatted edge list for visualization and save it as a pdf file.

    Args:
        g (DFGraph): The DFGraph object representing the network architecture.
        directory (str): The path to the directory where the plot should be saved.
        format (Optional[str]): The format of the plot file. Defaults to "pdf".
        quiet (bool): Whether to suppress the rendering progress message. Defaults to True.
        name (str): The name of the plot file. Defaults to an empty string.

    Raises:
        TypeError: If the rendering fails for any reason.

    Returns:
        None: This function does not return anything.
    """
    print("Plotting network architecture...")

    dot = _create_dot_figure(g)

    _render_dot_figure(dot, directory, format, quiet)

    print("Saved network architecture plot to directory:", directory)


def _create_dot_figure(g: DFGraph) -> Digraph:
    """
    Create a Dot graph for visualization.

    Args:
        g (DFGraph): The DFGraph object representing the network architecture.

    Returns:
        Digraph: The Dot graph representation of the network architecture.
    """
    dot = Digraph("render_dfgraph")

    dot.attr("graph")

    for u in g.v:
        node_name = g.node_names.get(u)
        node_name = (
            node_name if node_name is None else "{} ({})".format(node_name, str(u))
        )
        attrs = {"style": "filled"} if g.is_backward_node(u) else {}
        dot.node(str(u), node_name, **attrs)

    for edge in g.edge_list:
        dep_order = str(g.args[edge[-1]].index(edge[0]))
        dot.edge(*map(str, edge), label=dep_order)

    return dot


def _render_dot_figure(dot: Digraph, directory: str, format: str, quiet: bool) -> None:
    """
    Render and save the Dot figure to the specified directory.

    Args:
        dot (Digraph): The Dot graph representation of the network architecture.
        directory (str): The path to the directory where the plot should be saved.
        format (str): The format of the plot file.
        quiet (bool): Whether to suppress the rendering progress message.

    Raises:
        TypeError: If the rendering fails for any reason.
    """
    try:
        dot.render(directory=directory, format=format, quiet=quiet)
    except TypeError:
        dot.render(directory=directory, format=format)
