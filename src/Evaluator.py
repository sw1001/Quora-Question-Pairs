import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
Author: Shaobo Wang

Evaluate the data cleanliness for each column

This code script checks the data quality to identify the missing and incorrect values
The fractions of both will be recorded as indicators for future analysis
For each column, the fraction of clean data = 1 - fraction of issued values in this column
The total data quality score = sum of each column's clean data fraction / number_of_columns

To check noise values:
For nominal data, simply check whether the value is in wrong data type or format
For numeric data, use box plot to help identify the outliers
'''


class Evaluator:
    # data members
    file_name = ''
    data_frame = pd.DataFrame()

    # constructor
    def __init__(self, options):
        self.file_name = options[1]
        self.set_input_data()

    # read data from csv
    def set_input_data(self):
        self.data_frame = pd.read_csv(self.file_name)

    # get input data
    def get_input_data(self):
        return self.data_frame

    # calculate missing value fraction
    def get_missing_values_fraction(self):
        # get the sum of nulls for each column
        nulls = self.data_frame.isnull().sum()
        # calculate and return
        fractions = []
        for i in range(0, len(nulls)):
            fractions.append(nulls[i] / self.data_frame.shape[0])
        return fractions

    # calculate the noise value fraction
    def get_noise_values_fraction(self):
        fractions = []
        print(self.data_frame.dtypes)
        # detect suspect outliers
        i = 0
        for column in self.data_frame.columns:
            # only detect outliers for float attribute
            if self.data_frame.dtypes[i] == 'object':
                fractions.append(0.0)
            # detect outliers using box plot
            elif self.data_frame.dtypes[i] == 'float64' or self.data_frame.dtypes[i] == 'int64':
                print(column)
                plt.figure()
                # list stores all un null values
                data = self.data_frame[column]
                data_list = list(data[pd.notnull(data)])
                print(len(data_list))
                bp = plt.boxplot(data_list)
                # bp['fliers'] is a list containing one Line2D object holding the outliers data
                # bp['fliers'][0].get_data() returns a tuple containing two arrays for x-axis and y-axis respectively
                # get_data()[0] == get_xdata()
                # get_data()[1] == get_ydata()
                outliers = bp['fliers'][0].get_ydata()
                print(outliers)
                print('Num of outliers: ' + str(len(outliers)))
                plt.title(column)
                # uncomment the following line to see the box plots
                # plt.show()
                fractions.append(len(outliers) / self.data_frame.shape[0])
            else:
                print(column)
                fractions.append(0.0)
            i += 1
        return fractions

    # evaluate the results
    def evaluate(self):
        miss_fractions = self.get_missing_values_fraction()
        noise_fractions = self.get_noise_values_fraction()
        # calculate the quality score
        clean_fractions = []
        for i in range(0, len(miss_fractions)):
            clean_fractions.append(1 - miss_fractions[i] - noise_fractions[i])
        quality_score = sum(clean_fractions) / self.data_frame.shape[1]
        # print results
        print(miss_fractions)
        print(noise_fractions)
        print(clean_fractions)
        # print in csv style
        print('Attributes, Missing frac, Noise frac, Clean frac')
        i = 0
        for column in self.data_frame.columns:
            print(str(column) + ', ' + str(miss_fractions[i]) + ', ' +
                  str(noise_fractions[i]) + ', ' + str(clean_fractions[i]))
            i += 1
        print(quality_score)

