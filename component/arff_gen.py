"""This module will genearte arff files required.

Arff file is supported by WEKA machine learning library, which contains the
data for learning and verification. For more informations, read WEKA's
documentation.

Attributes:
    all_classes(list): A list of string which contains all fault tags.
"""
from functools import reduce
import util.localizer_config as localizer_config
from util.localizer_config import config
import util.kpi_info as kpi_info


def gen_file(exps, arff_path, fromzero=False):
    """Generates the WEKA arff file.

    The anomalies will be aggregated by a fixed length of window. For instance,
    if this length is set to 3, then the anomalies will be [(anomalies at T1),
    (anomalies at T1 to T2), (anomalies at T2 to T4), ... (anomalies at TN-2 to
    TN)]. The length is defined in '[predictor]/<sliding_window>' in the config
    file.

    Args:
        exps(dict): A dictionary which maps the id of an target experiment to
            its util.runtime.Observation class.
        arff_path(str): A string that represent the path of the arff file.
        fromzero(boolean): If set to True, then the end of the window starts
            from 0, else starts from window size.

    Returns:
        None
    """
    lines = []
    lines.append('@relation anomalies')

    # attach attributes
    for i, kpi in enumerate(kpi_info.kpi_list):
        lines.append("@attribute {kpi_name} {{TRUE, FALSE}}".format(kpi_name='_'.join(eval(kpi.tag))))

    # attach fault types
    exptag_name = config.get('exp_tag', 'tag')
    exptag_klass = localizer_config.get_plugin('exp_tag', exptag_name)
    import util.runtime as runtime
    #lines.append('@attribute class {{{types}}}'.format(types=','.join(runtime.all_classes)))
    lines.append('@attribute class {{{types}}}'.format(types="failurefree_none,MemL_0,MemL_1,MemL_2,MemL_3,MemL_4,MemL_5,MemL_6,MemL_7,MemL_8,MemL_9,PacL_0,PacL_1,PacL_2,PacL_3,PacL_4,PacL_5,PacL_6,PacL_7,PacL_8,PacL_9,CpuH_0,CpuH_1,CpuH_2,CpuH_3,CpuH_4,CpuH_5,CpuH_6,CpuH_7,CpuH_8,CpuH_9"))
    lines.append('')

    # attach instances
    lines.append('@data')
    sliding_window = config.getint('predictor', 'sliding_window')
    for exp_id, exp in exps.items():
        for uu in exp.exp_info:
            print(uu, ": ", exp.exp_info[uu])
        input(exp.exp_info['fault_type'])

        continue

        data = exp.exp_data
        tag = exptag_klass.tag(exp)

        for current, d in enumerate(data):
            if not fromzero and current + 1 < sliding_window:
                continue

            start = max(0, current + 1 - sliding_window)
            anomalies = reduce(lambda s1, s2: s1.union(s2),
                               [set(d) for d in data[start:current]],
                               set())
            # create boolean list
            booleans = ["FALSE"] * len(kpi_info.kpi_list)
            # print(len(booleans))
            # print(len(anomalies))
            for idx in anomalies:
                booleans[idx] = "TRUE"

            lines.append("{booleans}, {tag}"
                         .format(booleans=', '.join(booleans),
                                 tag=tag))

    with open(arff_path, 'w') as f:
        f.writelines('\n'.join(lines))
