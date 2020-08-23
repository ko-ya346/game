from logging import DEBUG
from lightgbm.callback import _format_eval_result
from base_log import get_logger


def lgbm_logger(VERSION, level=DEBUG, period=1, show_stdv=True):

    def _callback(env):
        if period > 0 and env.evaluation_result_list and (env.iteration + 1) % period == 0:
            result = '\t'.join([_format_eval_result(x, show_stdv) for x in env.evaluation_result_list])
            get_logger(VERSION).log(level, "[%d]\t%s" % (env.iteration + 1, result))
    _callback.order = 10
    return _callback