# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 14:52:34 2017

In this file, we clean the earning_info.csv. We validate by year and make sure that
the categories of earnings for each row all add up to 100 percent. Otherwise, we remove
the row.

@author: Armaan Khullar
"""

import pandas as pd
import sys

def main():
    df = pd.read_csv("earning_info.csv") #read in the input file into the dataframe.
    df = df.dropna() #dropping the rows with missing observations.
    
    #Dropping all rows with any negative values.
    df = df[df>0].dropna() 
    
    #Dropping rows where the year is less than 2000.
    df = df.drop(df[(df["year"]) < 2000].index)
    
    #Dropping rows where the year greater than 2017, the current year.
    df = df.drop(df[(df["year"]) > 2017].index)
    
    #Dropping all rows where the percentages of categories of earnings don't add up to 100.
    df = df.drop(df[100 != (df["percent_with_earnings_10000_to_14999"] + df["percent_with_earnings_15000_to_24999"] + df["percent_with_earnings_1_to_9999"] + df["percent_with_earnings_25000_to_34999"] + df["percent_with_earnings_35000_to_49999"] + df["percent_with_earnings_50000_to_64999"] + df["percent_with_earnings_65000_to_74999"] + df["percent_with_earnings_75000_to_99999"] + df["percent_with_earnings_over_100000"]  )].index)
    
    #Splitting the area_name column into City and State.
    df["City"], df["State"] = zip(*df["area_name"].str.split("- ").tolist())     #Splitting the area_name column into City and State.
    del df["area_name"] #now city and state are separated so we get rid of "City, State"
    
    #Finally outputting the dataframe to a csv file.
    df.to_csv("earning_info_CLEANED.csv")

if __name__ == "__main__":
    main()


