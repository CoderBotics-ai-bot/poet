import pathlib
from typing import Optional

import numpy as np
from graphviz import Digraph

from poet.utils.checkmate.core.dfgraph import DFGraph


def plot_dfgraph(
    g: DFGraph,
    directory: str,
    format: str = "pdf",
    quiet: bool = True,
    name: Optional[str] = "",
) -> None:
    """
    Generate Graphviz-formatted edge list for visualization and write the graph to a file.

    Args:
        g (DFGraph): The graph to plot.
        directory (str): The directory to save the plot in.
        format (str, optional): The file format of the plot (default is "pdf").
        quiet (bool, optional): Whether to suppress output while rendering the plot (default is True).
        name (str, optional): A name for the plot (default is an empty string).

    Raises:
        TypeError: If the format argument is not supported.

    Returns:
        None

    """
    print("Plotting network architecture...")

    dot = Digraph("render_dfgraph" + str(name))
    dot.attr("graph")

    _add_nodes(dot, g)
    _add_edges(dot, g)

    try:
        dot.render(directory=directory, format=format, quiet=quiet)
    except TypeError:
        dot.render(directory=directory, format=format)

    print("Saved network architecture plot to directory:", directory)


def _add_nodes(dot: Digraph, g: DFGraph) -> None:
    for u in g.v:
        node_name = g.node_names.get(u)
        node_name = (
            node_name if node_name is None else "{} ({})".format(node_name, str(u))
        )
        attrs = {"style": "filled"} if g.is_backward_node(u) else {}
        dot.node(str(u), node_name, **attrs)


def _add_edges(dot: Digraph, g: DFGraph) -> None:
    for edge in g.edge_list:
        dep_order = str(g.args[edge[-1]].index(edge[0]))
        dot.edge(*map(str, edge), label=dep_order)
