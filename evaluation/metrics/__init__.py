import numpy as np
import pyximport

# Cython Compilation
pyximport.install(setup_args={"include_dirs": np.get_include()}, language_level=3)

from .rank_cy import evaluate_cy


def compute_CMC_mAP(
    distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20, use_metric_cuhk03=False
):
    """Evaluates CMC rank.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 20.
        use_metric_cuhk03 (bool, optional): use single-gallery-shot setting for cuhk03.
            Default is False. This should be enabled when using cuhk03 classic split.
    """
    return evaluate_cy(
        distmat=distmat,
        q_pids=q_pids,
        g_pids=g_pids,
        q_camids=q_camids,
        g_camids=g_camids,
        max_rank=max_rank,
        use_metric_cuhk03=use_metric_cuhk03,
    )
