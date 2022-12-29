/*
 * 
 * import warnings
from time import time
from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt, ticker
from scipy.optimize import fmin_slsqp
from scipy.signal import find_peaks
from tabulate import tabulate

from autoeq.constants import Constants.DEFAULT_SHELF_FILTER_MIN_FC, Constants.DEFAULT_SHELF_FILTER_MAX_FC, Constants.DEFAULT_SHELF_FILTER_MIN_Q, \
    Constants.DEFAULT_SHELF_FILTER_MAX_Q, Constants.DEFAULT_SHELF_FILTER_MIN_GAIN, Constants.DEFAULT_SHELF_FILTER_MAX_GAIN, \
    Constants.DEFAULT_PEAKING_FILTER_MIN_FC, Constants.DEFAULT_PEAKING_FILTER_MAX_FC, Constants.DEFAULT_PEAKING_FILTER_MIN_Q, \
    Constants.DEFAULT_PEAKING_FILTER_MAX_Q, Constants.DEFAULT_PEAKING_FILTER_MIN_GAIN, Constants.DEFAULT_PEAKING_FILTER_MAX_GAIN, \
    Constants.DEFAULT_PEQ_OPTIMIZER_MIN_F, Constants.DEFAULT_PEQ_OPTIMIZER_MAX_F, Constants.DEFAULT_PEQ_OPTIMIZER_MAX_TIME, \
    Constants.DEFAULT_PEQ_OPTIMIZER_TARGET_LOSS, Constants.DEFAULT_PEQ_OPTIMIZER_MIN_CHANGE_RATE, Constants.DEFAULT_PEQ_OPTIMIZER_MIN_STD
*/

using System;

namespace AutoEQ2.Core;

public class OptimizationFinished : TimeoutException
{

}
