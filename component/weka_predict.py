"""Core module of the premise prediction."""
import re
import os
import json
from collections import OrderedDict
import util.localizer_config as localizer_config
import util.localizer_log as localizer_log


if os.path.isfile('weka.json'):
    with open('weka.json') as f:
        try:
            __OPTIONS = json.load(f)
        except:
            localizer_log.error('weka.json not in right format.')
            __OPTIONS = []
else:
    localizer_log.error('weka.json not found.')
__classifiers = OrderedDict(__OPTIONS)
__predictors = {}


def init(arff_path, dst_path):
    """Main training function of the premise prediction.

    The function uses the WEKA machine learning library, implemented by
    python-weka-wrapper Python library. It will divide the data into given
    folds, and do the training and varification. Results and summary will be
    written to a file.

    Args:
        arff_path(str): The string that represents the path of the input arff
            file.
        dst_path(str): The string that represents the path of the output
        summary file.

    Returns:
        None
    """
    import weka.core.converters as converters
    from weka.classifiers import Classifier
    from weka.classifiers import Evaluation
    from weka.core.classes import Random

    global __classifiers
    global __predictors

    data = converters.load_any_file(arff_path)
    print("arff loaded")
    data.class_is_last()

    lines = []
    summaries = []
    summary_line = ['Model'.ljust(16),
                    'Precision'.ljust(12),
                    'Recall'.ljust(12),
                    'F-measure'.ljust(12),
                    'Accuracy'.ljust(12),
                    'FPR'.ljust(12)]
    summaries.append('\t'.join(summary_line))
    for classifier, option_str in __classifiers.items():
        option_list = re.findall(r'"(?:[^"]+)"|(?:[^ ]+)', option_str)
        option_list = [s.replace('"', '') for s in option_list]

        classifier_name = classifier.split('.')[-1]
        info_str = "Using classifier: {classifier}, options: {options}"\
            .format(classifier=classifier_name,
                    options=str(option_list))
        localizer_log.msg(info_str)
        lines.append(info_str)

        cache = localizer_config.load_model(classifier_name)
        if cache:
            cls, data_ = cache
        else:
            cls = Classifier(classname=classifier, options=option_list)
            cls.build_classifier(data)
            localizer_config.save_model(cls, data, classifier_name)

        __predictors[classifier_name] = cls

        evl = Evaluation(data)
        evl.crossvalidate_model(cls, data, 10, Random(1))

        print(evl.percent_correct)
        print(evl.summary())
        print(evl.class_details())

        lines.append(evl.summary())
        lines.append(evl.class_details())

        summary_line = []
        summary_line.append(classifier_name.ljust(16))
        summary_line.append("{:.3f}".format(evl.weighted_precision *
                                            100).ljust(12))
        summary_line.append("{:.3f}".format(evl.weighted_recall *
                                            100).ljust(12))
        summary_line.append("{:.3f}".format(evl.weighted_f_measure *
                                            100).ljust(12))
        summary_line.append("{:.3f}".format(evl.percent_correct).ljust(12))
        summary_line.append("{:.3f}".format(evl.weighted_false_positive_rate *
                                            100)
                            .ljust(12))
        summaries.append('\t'.join(summary_line))

    with open(dst_path, 'w') as f:
        f.writelines('\n'.join(lines))
        f.writelines('\n'*5)
        f.writelines('\n'.join(summaries))


def pred_seq(exp, arff_path, dst_folder):
    """The function to generate a detailed prediction sequence of the experiment.

    Args:
        exp(obj): An util.runtime.Observation object.
        arff_path(str): The string that represents the path of the input arff
            file.
        dst_folder(str): The path of the folder to put the result.

    Returns:
        None
    """
    global __predictors
    import util.runtime as runtime
    import weka.core.converters as converters

    data = converters.load_any_file(arff_path)
    data.class_is_last()

    for cls_name, cls in __predictors.items():
        f_path = os.path.join(dst_folder, cls_name + '.txt')
        with open(f_path, 'w') as f:
            lines = []
            for index, inst in enumerate(data):
                pred = cls.classify_instance(inst)
                lines.append(runtime.all_classes[int(pred)])
            f.writelines('\n'.join(lines))


def check_stable_prediction(prediction):
    """Remains compatibility with ranker code.

    Args
        prediction(list): A list of True and/or False.

    Returns:
        None
    """
    pass
