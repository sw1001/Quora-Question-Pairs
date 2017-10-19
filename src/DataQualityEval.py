import pandas as pd
import numpy as np
import sys
from Evaluator import Evaluator

'''
Author: Shaobo Wang

This code script checks the data quality to identify the missing and incorrect values
The fractions of both will be recorded as indicators for future analysis
For each column, the fraction of clean data = 1 - fraction of issued values in this column
Return 3 attributes having the lowest clean data fraction
The total data quality score = sum of each column's clean data fraction / number_of_columns

To check noise values:
For nominal data, simply check whether the value is in wrong data type or format
For numeric data, use box plot to help identify the outliers
'''


class DataQualityEval:
    options = ''

    # constructor
    def __init__(self, options):
        self.options = options

    # main method
    def main(self):
        evaluator = Evaluator(self.options)
        evaluator.evaluate()


if __name__ == '__main__':
    dq_eval = DataQualityEval(sys.argv)
    dq_eval.main()
