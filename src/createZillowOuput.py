# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 17:55:21 2017

@author: Armaan Khullar
"""

import pandas as pd
import sys

def main():
    """
    This method takes in the three csv files for the Zillow data and combines 
    them into a single csv file called Zillow_House_Output.csv.
    """
    df1 = pd.read_csv("Zillow_House_Detail_40000.csv")
    df2 = pd.read_csv("Zillow_House_Info_40000.csv")
    
    #Merge the two dataframes together based on zpid, latitude, longitude, and amount.
    df12 = pd.merge(df1, df2, on=["zpid", "latitude", "longitude", "amount"], how = "inner") 
    
    df3 = pd.read_csv("Zillow_House_Update_40000.csv") 
    df = pd.merge(df12, df3, on="zpid") #Merge the third dataframe to the previously merged dataframe.
    
    df.to_csv("Zillow_House_Output.csv", sep=",") #Write the resulting dataframe to a csv


if __name__ == "__main__":
    main()   
    