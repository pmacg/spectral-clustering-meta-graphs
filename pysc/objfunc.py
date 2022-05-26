"""
Provides a general definition of a clustering objective function.
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import List
import sgtl


class ClusteringObjectiveFunction(ABC):
    """
    An abstract class representing a clustering objective function.
    """

    @staticmethod
    @abstractmethod
    def apply(graph: sgtl.Graph, clusters: List[List[int]]) -> float:
        """
        Apply the objective function to the given clusters in the graph.

        :param graph: The underlying graph.
        :param clusters: The clusters we are applying the objective function to.
        :return: The value of the objective function.
        """
        pass

    @staticmethod
    @abstractmethod
    def better(val_1: float, val_2: float) -> bool:
        """
        Compare the two objective values val_1 and val_2.

        Returns True is val_1 is better than val_2.

        :param val_1:
        :param val_2:
        :return: True if the val_1 objective value is better than val_2. False otherwise.
        """
        pass


class KWayExpansion(ClusteringObjectiveFunction):
    """
    The k-way expansion objective function. This is defined to be

    .. math:

       \\rho(C_1, \\ldots, C_k) \\triangleq \\max_{i \\in [1, k]} \\phi(C_i)

    """

    @staticmethod
    def apply(graph: sgtl.Graph, clusters: List[List[int]]) -> float:
        try:
            conductances = []
            for cluster in clusters:
                if len(cluster) > 0:
                    conductances.append(graph.conductance(cluster))
                else:
                    conductances.append(1)
        except ZeroDivisionError:
            # In the case of a zero division error, it must be that one of the clusters is empty, return 1
            return 1
        return max(conductances)

    @staticmethod
    def better(val_1: float, val_2: float) -> bool:
        return val_1 < val_2

