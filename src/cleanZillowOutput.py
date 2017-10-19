# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 19:41:43 2017

In this file, we clean the Zillow output data. We drop all negative values and 
also fix missing values for some attributes. We also ensure that the year built
is reasonable. 

@author: Armaan Khullar
"""

import pandas as pd
import numpy as np
import requests
import sys


def initialize(df):
    #Drop the column that only lists the indices.
    df = df.drop(df.columns[0], axis=1)
    return df

def dropNegBathrooms(df):
    #Now, we drop rows where the number of bathrooms is negative.
    df = df.drop(df[df.bathrooms < 0].index)
    return df

def dropNegBedrooms(df):
    #Now, we drop rows where the number of bedrooms is negative.
    df = df.drop(df[df.bedrooms < 0].index)
    return df

def dropNegFullBathrooms(df):
    #Now, we drop rows where the number of fullbathrooms is negative.
    df = df.drop(df[df.fullbathrooms < 0].index)
    return df

def fixMissingPools(df):
    #Replace missing values of pools with 0
    df.pools = df[['pools']].convert_objects(convert_numeric=True).fillna(0)
    return df

def fixMissingACT(df):
    #Replace missing values of airconditiontype with 0
    df.airconditiontype = df[['airconditiontype']].convert_objects(convert_numeric=True).fillna(0)
    return df    

def fixMissingHST(df):
    #Replace missing values of heat system type with the mode.
    df["heatsystemtype"].fillna(df["heatsystemtype"].mode()[0], inplace=True)
    return df

def fixMissingBQ(df):
    #Replace missing values of building qaulity with the mode
    df["heatsystemtype"].fillna(df["buildingquality"].mode()[0], inplace=True)  

    #Drop rows with building quality not between 1 and 10.
    df = df.drop(df[(df.buildingquality > 10)].index)
    df = df.drop(df[(df.buildingquality < 1)].index)       
    return df

def fixYearBuilt(df):
    #Drop rows where the year built is not at least 1600.
    #There are some homes in America that date back to the colonial era.
    #We give a benchmark of at least 1700 as that there was some colonial settlements back then.
    #Any year below would be removed as it is very unlikely that it is correct. 
    df = df.drop(df[df.yearbuilt < 1700].index)
    #Remove rows in which the year built is greater than 2017
    df = df.drop(df[df.yearbuilt > 2017].index)
    return df

def fixAmount(df):
    #We drop rows where the amount is less than $10,000, as the house would
    #be suspiciously cheap. Delete all houses over $50 million as that 
    #would also be too expensive and, therefore, noisy data. 
    df = df.drop(df[df.amount < 10000].index)
    df = df.drop(df[df.amount > 50000000].index)
    return df
    
    


def main():
    df = pd.read_csv("Zillow_House_Output.csv") #read in the input file into the dataframe.
    df = initialize(df)     #Drop the column that only lists the indices.

    #Drop rows where the number of bathrooms is negative.
    df = dropNegBathrooms(df) 
    
    #Now, we drop rows where the number of bedrooms is negative.
    #df = df.drop(df[df.bedrooms < 0].index)
    df = dropNegBedrooms(df)
    
    #Now, we drop rows where the number of fullbathrooms is negative.
    df = dropNegFullBathrooms(df)
    
    #Drop rows where the number of fullbathrooms is less than the number of bathrooms
    for index, row in df.iterrows():
        if row["fullbathrooms"] > row["bathrooms"]:
           row["fullbathrooms"] = row["bathrooms"]
        
    #Replace missing values of pools with 0
    df = fixMissingPools(df)
    
    #Replace missing values of airconditiontype with 0
    df = fixMissingACT(df)
    
    #Replace missing values of heatsystemtype with the mode
    df = fixMissingHST(df)

    #Replace missing values of buildingquality with the mode
    #Enforce the rule that building qaulity scores must be between 1 and 10.
    df = fixMissingBQ(df)
         
    #Drop rows where the year built is not at least 1600.
    #There are some homes in America that date back to the colonial era.
    #We give a benchmark of at least 1700 as that there was some colonial settlements back then.
    #Any year below would be removed as it is very unlikely that it is correct. 
    df = fixYearBuilt(df)
    
    #Drop all rows where the finished area is greater  than the lot size, which is the
    #total area of the property.
    df = df.drop(df[(df.finishedSqFt > df.lotsizeSqFt)].index)
    
    df = fixAmount(df)
    
    df = df.drop(df.columns[13], axis=1) #removing the rooms column  as it is unnecessary.
    
    #We now drop all rows for which the values are missing.
    df = df.dropna()
    df.to_csv("Zillow_Cleaned.csv")

if __name__ == "__main__":
    main()

