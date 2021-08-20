"""
A module containing routines utilities for running the model.

"""

COVID_START = '2020-03'


def compute_induced_suicides(suicide, pre_preds):
    date_end = suicide.index[-1]
    induced_suicide = suicide[COVID_START:] - pre_preds[COVID_START:date_end]
    return induced_suicide
