
import evolutionary_process
import sys
# import multiprocessing 
# multiprocessing.set_start_method('fork', force=True)
import concurrent.futures
import os

from algorithms.evaluate import init_worker_env

from utils.args_processor import get_qd_algo_args, get_input_arguments
from utils.common_tools import get_new_run_name

from utils.io_run_data import export_dict_pickle

import configs.exec_config as ex_cfg
import configs.eval_config as eval_cfg
import environments.src.env_constants as env_consts


def dump_input_arguments(qd_algo_args, global_config):
    """Dump arguments that will be given to the QD algorithm to generate grasping trajectories."""
    export_dict_pickle(run_name=qd_algo_args['run_name'], dict2export=global_config, file_name='config')


def end_of_run_routine(archive_success_len):
    """Must only be called in the main thread."""
    if archive_success_len == 0:
        print("Empty success archive.")
        return ex_cfg.RETURN_SUCCESS_CODE

    print(f"End of running. Size of output success archive : {archive_success_len}")
    print("Success.")

    return ex_cfg.RETURN_SUCCESS_CODE


def init_run_dump_folder(global_config):
    """Must only be called in the main thread."""
    run_name = get_new_run_name(
        log_path=global_config['output']['log_path'], folder_name=global_config['output']['folder_name']
    )
    return run_name


def init_grasping_env(env_class, env_kwargs):
    env = env_class(**env_kwargs)
    return env


def get_global_config(input_args, env_config):
    """
    Build global config from input args and extracted env config.
    No live env object needed!
    """
    n_domain_randomization_perturbations = eval_cfg.DOMAIN_RANDOMIZATION_N_NOISY_TRIALS \
        if input_args['eval_kwargs']['domain_randomization_fitness'] else None
    shaking_params = env_consts.SHAKING_PARAMETERS

    global_config = {
        'algorithm': input_args['qd_method'],

        'evo_proc': {
            'archive_type': input_args['archive_type'],
            'pop_size': input_args['pop_size'],
            'n_budget_rollouts': input_args['n_budget_rollouts'],
            'mut_strat': input_args['mut_strat'],
            'prob_cx': input_args['prob_cx'],
            'sigma_mut': input_args['sigma_mut'],
            'select_off_strat': input_args['select_off_strat'],
            'replace_pop_strat': input_args['replace_pop_strat'],
            'archive_limit_strat': input_args['archive_limit_strat'],
            'is_novelty_required': input_args['is_novelty_required'],
            'is_pop_based': input_args['is_pop_based'],
            'qd_method_genotype_len': input_args['qd_method_genotype_len'],
            'include_invalid_inds': input_args['include_invalid_inds'],
        },

        'env': {
            'kwargs': input_args['env_kwargs'],
            'class': input_args['env_class'],  # Add class for worker creation
        },

        'evaluate': {
            'kwargs': input_args['eval_kwargs'],
            'search_space_bb': env_config['search_space_bb'],  # From extracted config
            'n_domain_randomization_perturbations': n_domain_randomization_perturbations,
            'shaking_params': shaking_params,
        },

        'robot': {
            'name': input_args['robot'],
            'env_class': input_args['env_class'],
            'name_str': input_args['robot_str']
        },

        'object': {
            'name': input_args['object'],
            'stabilized_obj_pose': env_config['stabilized_obj_pose'],  # From extracted config
        },

        'output': {
            'log_path': input_args['log_path'],
            'folder_name': input_args['folder_name'],
        },

        'parallelize': input_args['parallelize'],
        'search_representation': input_args['search_representation'],
        'debug': input_args['debug'],
    }

    return global_config

def run_qd_routine(**qd_algo_args):
    """Run QD algorithm with optional parallelization."""
    
    env_class = qd_algo_args['env_class']
    env_kwargs = qd_algo_args['env_kwargs']
        
    if qd_algo_args['parallelize']:
        n_workers = os.cpu_count()
        
        print(f"Running with {n_workers} parallel workers")
        
        # Prepare worker initialization
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_worker_env,
            initargs=(env_class, env_kwargs)
        ) as executor:
            qd_algo_args['multiproc_pool'] = executor
            archive_success_len = evolutionary_process.run_qd(**qd_algo_args)
    else:
        print("Running in serial mode")
        # In serial mode, we must manually initialize the global environment 
        # because the partial function from args_processor expects it to exist.
        init_worker_env(env_class, env_kwargs)
        archive_success_len = evolutionary_process.run_qd(**qd_algo_args)
    
    return archive_success_len

# def run_qd_routine(**qd_algo_args):

#     if qd_algo_args['parallelize']:
#         with concurrent.futures.ProcessPoolExecutor() as executor:
#             qd_algo_args['multiproc_pool'] = executor
#             archive_success_len = evolutionary_process.run_qd(**qd_algo_args)
#     else:
#         archive_success_len = evolutionary_process.run_qd(**qd_algo_args)

#     return archive_success_len

def extract_env_config(env_class, env_kwargs):
    """
    Create env temporarily just to extract configuration, then destroy it.
    This happens once in the main process.
    
    Returns:
        dict: Static configuration that can be pickled
    """
    env = env_class(**env_kwargs)
    
    config = {
        'stabilized_obj_pose': env.mj_client.default_object_pose,
        'search_space_bb': env.search_space_bb,
    }
    
    # Explicitly clean up
    if hasattr(env, 'close'):
        env.close()
    del env
    
    return config

def main():
    """QD-Grasp entry point - clean and simple!"""
    
    # Get arguments
    input_args = get_input_arguments()
    
    # Extract static configuration from a temporary env
    env_config = extract_env_config(
        env_class=input_args['env_class'],
        env_kwargs=input_args['env_kwargs']
    )
    
    # Build global config (no live env needed!)
    global_config = get_global_config(
        input_args=input_args,
        env_config=env_config
    )
    
    # Create output folder
    dump_folder_name = init_run_dump_folder(global_config=global_config)
    
    # Get QD algorithm args (creates evaluation function internally)
    qd_algo_args = get_qd_algo_args(
        cfg=global_config,
        dump_folder_name=dump_folder_name
    )
    
    # Save configuration
    dump_input_arguments(qd_algo_args=qd_algo_args, global_config=global_config)
    
    # Run QD algorithm
    archive_success_len = run_qd_routine(**qd_algo_args)
    
    # Finish
    return end_of_run_routine(archive_success_len)

# def main():
#     """QD-Grasp entry point."""

#     # Initialize arguments
#     input_args = get_input_arguments()
#     env = init_grasping_env(env_class=input_args['env_class'], env_kwargs=input_args['env_kwargs'])
#     global_config = get_global_config(input_args=input_args, env=env)

#     dump_folder_name = init_run_dump_folder(global_config=global_config)

#     qd_algo_args = get_qd_algo_args(cfg=global_config, env=env, dump_folder_name=dump_folder_name)

#     # Locally save params
#     dump_input_arguments(qd_algo_args=qd_algo_args, global_config=global_config)

#     # QD algorithm execution
#     archive_success_len = run_qd_routine(**qd_algo_args)

#     # End of running
#     return end_of_run_routine(archive_success_len)


if __name__ == "__main__":
    sys.exit(main())


