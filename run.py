# -*- coding: utf-8 -*-
"""The main module of premise.

Example:
    $ python3 run.py
"""

import os
import weka.core.jvm as jvm
import util.localizer_log as localizer_log
import util.localizer_config as localizer_config
from util.localizer_config import config
import util.runtime as runtime
import util.kpi_info as kpi_info
import component.preprocess as preprocess
import component.arff_gen as arff_gen
import component.exp_filter_manager as exp_filter_manager
import component.weka_predict as weka_predict


def run():
    """Main function of premise.

    Args:
        None

    Returns:
        None
    """

    #if config.has_option('default', 'max_heapsize'):
        # jvm.start(config.get('default', 'max_heapsize'))
    #else:
        # jvm.start()
    # Create the target folder
    dst_folder = localizer_config.get_folder('dst')
    localizer_config.reset_path(dst_folder)

    # !!! Commented by Rahim. kpi_indices.txt is generated by separate script and put into classifications folder manually. Commented code implements initialization of KPIs from SCAPI provided file
    # Load KPI
    # kpi_meta = localizer_config.get_meta_path('kpi')
    #
    # with open(kpi_meta) as f:
    #    localizer_log.msg("Reading kpi file...")
    #    kpi_info.initialize(f.read())
    # Write KPI indices
    # kpi_info.write_kpi_indices(
    #    localizer_config.get_dst_path('kpi_indices.txt'))
    kpi_info.init(localizer_config.get_dst_path('kpi_indices.txt'))
    print(kpi_info.kpi_list)
    exit()


    # Process the original file and put it to
    if localizer_config.component_enabled('preprocess'):
        preprocess.preprocess()

    # Read the experiment data
    localizer_log.msg("Reading experiments...")
    training_dir = localizer_config.get_src_path('training')
    runtime.add_all(training_dir)
    if config.has_option('folder', 'target'):
        target_dir = localizer_config.get_src_path('target')
        runtime.add_target(target_dir)
    localizer_log.msg("Finish Reading.")

    if localizer_config.component_enabled('exp_filter'):
        exps = exp_filter_manager.filter_(runtime.all_exps)
    else:
        exps = runtime.all_exps

    arff_path = localizer_config.get_dst_path('training.arff')
    arff_gen.gen_file(exps, arff_path)

    if localizer_config.component_enabled('predictor'):
        localizer_log.msg("Prediction enabled.")
        localizer_log.msg("Runing prediction")
        dst_path = localizer_config.get_dst_path('predictions.txt')
        weka_predict.init(arff_path, dst_path)
        for exp_id, exp in runtime.targets_exps.items():
            exp_dst_path = localizer_config.get_dst_path(
                exp.exp_info['full_name'])
            localizer_config.reset_path(exp_dst_path)
            exp_arff_path = os.path.join(exp_dst_path, 'target.arff')
            arff_gen.gen_file({exp_id: exp}, exp_arff_path, fromzero=True)
            weka_predict.pred_seq(exp, exp_arff_path, exp_dst_path)
    else:
        localizer_log.msg("Prediction not enabled.")
    jvm.stop()


if __name__ == '__main__':
    run()
