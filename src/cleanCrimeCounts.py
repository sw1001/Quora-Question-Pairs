# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 19:41:43 2017

In this file, we clean the crime_counts.csv. We validate the year,
split the area_name into City and State, ensure that the count of crimes
of each category (within each row) adds up to All Crimes. We make sure that
each crime number is nonnegative and we output our cleaned dataframe to a csv file.

@author: Armaan Khullar
"""

import pandas as pd
import sys

def main():
    df = pd.read_csv("crime_counts.csv") #read in the input file into the dataframe.
    df = df.dropna() #dropping the rows with missing observations.
    
    #Dropping rows where the year is less than 2000.
    df = df.drop(df[(df["year"]) < 2000].index)
    
    #Dropping rows where the year greater than 2017, the current year.
    df = df.drop(df[(df["year"]) > 2017].index)
    
    df["City"], df["State"] = zip(*df["area_name"].str.split("- ").tolist()) #Split city and state into seperate columns.
    del df["area_name"] #now city and state are separated so we get rid of "City, State"
    
    #Now we will make sure that the sum of all crimes all up.
    df = df.drop(df[(df["All Crimes"]) != (df["Burglary"] + df["Larceny"] + df["Motor vehicle theft"] + df["Murder and nonnegligent manslaughter"] + df["Property crime"] + df["Rape (revised definition)"] + df["Robbery"] + df["Violent crime"]  )].index)
    
    #Now, we drop rows where the number of "All Crimes is negative.
    df = df.drop(df[(df["All Crimes"] < 0)].index)
    
    #Now, we drop rows where the number of burglaries is negative.
    df = df.drop(df[df.Burglary < 0].index)
    
    #Now, we drop rows where the number of larcenies is negative.
    df = df.drop(df[df.Larceny < 0].index)
    
    #Now, we drop rows where the number of motor vehicle thefts is negative.
    df = df.drop(df[(df["Motor vehicle theft"] < 0)].index)
    
    #Now, we drop rows where the number of murder & nonnegligent manslaughter is negative.
    df = df.drop(df[(df["Murder and nonnegligent manslaughter"] < 0)].index)
    
    #Now, we drop rows where the number of property crimes is negative.
    df = df.drop(df[(df["Property crime"] < 0)].index)
    
    #Now, we drop rows where the number of rapes is negative.
    df = df.drop(df[(df["Rape (revised definition)"] < 0)].index)
    
    #Now, we drop rows where the number of robberies is negative.
    df = df.drop(df[(df["Robbery"] < 0)].index)
    
    #Now, we drop rows where the number of violent crimes is negative.
    df = df.drop(df[(df["Violent crime"] < 0)].index)
    
    df.to_csv("crime_counts_CLEANED.csv")
    
    
if __name__ == "__main__":
    main()
    
