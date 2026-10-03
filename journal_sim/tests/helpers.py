import numpy as np
from journal_sim.config import JournalConfig


def tiny_config(**kwargs):
    return JournalConfig(L=1, K=1, M=1, N=2, bits=1, bs_x=(-100.,), bs_y=(0.,),
                         ue_center_x=(15.,), ue_center_y=(0.,), noise_power_dbm=-30.,
                         channel_scale=1., gamma=1.).with_overrides(**kwargs)


def scalar_case(epsilon_boundary=0.2):
    cfg = tiny_config()
    H = np.ones((1, 1, 1, 1), complex)
    w = np.full((1, 1, 1), np.sqrt(cfg.noise_power) / (1 - epsilon_boundary), complex)
    return cfg, w, H
