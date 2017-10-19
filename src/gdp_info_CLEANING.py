# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 00:05:14 2017

@author: Armaan Khullar
"""

import pandas as pd
import sys

def main():
    """This function reads in the gdp_info.csv and cleans it by making sure
        that the year and the per_capita_gdp values are appropriate. It also
        cleans rows that are missing data.
    """
    df = pd.read_csv("gdp_info.csv") #read in the input file into the dataframe.
    df = df.dropna() #dropping the rows with missing observations.
    
    
    df = df.drop(df[(df["year"]) < 2000].index) #drop rows where the year is below 2000.
    df = df.drop(df[(df["year"]) > 2017].index) #drop rows where the year is greater than 2017.
    df = df.drop(df[(df["per_capita_gdp"]) < 2000].index) #drop rows where the gdp per capital is very low.
    
    df.to_csv("gdp_info_CLEANED.csv", sep=",") #Write the resulting dataframe to a csv
    
    
if __name__ == "__main__":
    main()


