"""
Data: 2021/09/21
Target: For NASBench101 Dataset architecture spec build, including prune non-unsed part, generate the spec hash for query
Modified from https://github.com/google-research/nasbench
Apache License 2.0: https://github.com/google-research/nasbench/blob/master/LICENSE
"""

from .model_spec import ModelSpec

__all__ = [ModelSpec]