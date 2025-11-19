
import pdb

import os
import numpy as np

from .elite_structured_archive import EliteStructuredArchive
from configs.qd_config import FillArchiveStrategy
import configs.eval_config as eval_cfg


class OutcomeArchive(EliteStructuredArchive):
    def __init__(self, search_space_bb, fill_archive_strat=FillArchiveStrategy.STRUCTURED_ELITES):
        super().__init__(search_space_bb=search_space_bb, fill_archive_strat=fill_archive_strat)

        self._it_export_success = 0
        self._at_least_one_success = False
        
        # Track ALL robust grasps (not just best per cell)
        self._all_robust_grasps = []

    def get_n_successful_cells(self):
        return int(np.sum([self._map_infos[key][eval_cfg.IS_SUCCESS_KEY_ID] for key in self._map_infos]))

    def update(self, pop):
        self._add_inds(pop=pop)
        self._add_all_robust(pop=pop)

    def _add_inds(self, pop):
        self.fill_elites(pop)

        if not self._at_least_one_success:
            # avoid useless for loops when checking for scs data (e.g. during export)
            self._at_least_one_success = np.sum(
                [self._map_infos[key][eval_cfg.IS_SUCCESS_KEY_ID] for key in self._map_infos]
            ) > 0
    
    def _add_all_robust(self, pop):
        """Store ALL robust grasps from the population."""
        valid_mask = pop.are_valid_inds()
        robust_mask = pop.infos[:, eval_cfg.IS_ROBUST_GRASP_KEY_ID].astype(bool)
        save_mask = valid_mask & robust_mask
        
        if np.sum(save_mask) > 0:
            for i in np.where(save_mask)[0]:
                self._all_robust_grasps.append({
                    'ind': pop.inds[i],
                    'bd': pop.bds[i],
                    'fit': pop.fits[i],
                    'info': pop.infos[i]
                })

    def export(self, run_name, curr_neval, elapsed_time, verbose=False, only_scs=True):

        if self._at_least_one_success:
            inds, bds, fits, infos = self.get_successful_inds_data()
        else:
            inds, bds, fits, infos = np.array([]), np.array([]), np.array([]), np.array([])

        export_target_name = 'success_archives' if only_scs else 'outcome_archives'
        export_archive_path = str(run_name) + '/' + export_target_name
        if not os.path.isdir(export_archive_path):
            os.mkdir(export_archive_path)

        saving = {
            # Original fields (grid-based, best per cell)
            "inds": inds,
            "behavior_descriptors": bds,
            "fitnesses": fits,
            'infos': infos,
            'infos_keys': eval_cfg.INFO_KEYS,
            "nevals": curr_neval,
            "elapsed_time": elapsed_time,
            
            # Single field for ALL robust grasps
            "robust_grasps": np.array(self._all_robust_grasps, dtype=object),
        }

        it_export = self._it_export_success if only_scs else self._it_export
        np.savez_compressed(file=export_archive_path + f'/individuals_{it_export}', **saving)

        if verbose:
            print(f'{export_target_name} n°{it_export} has been successfully dumped at {export_archive_path}.')
            print(f'  Grid archive (best per cell): {len(inds)} grasps')
            print(f'  All robust grasps: {len(self._all_robust_grasps)} grasps')

        if only_scs:
            self._it_export_success += 1
        else:
            self._it_export += 1


