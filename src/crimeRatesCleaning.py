# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 19:41:43 2017

@author: Armaan Khullar
"""

import pandas as pd
import sys


def fixYear(df):
    #Dropping rows where the year is less than 2000.
    df = df.drop(df[(df["year"]) < 2000].index)
        
    #Dropping rows where the year greater than 2017, the current year.
    df = df.drop(df[(df["year"]) > 2017].index)
    return df

def splitAreaName(df):
    #Split area_name into City and State
    df["City"], df["State"] = zip(*df["area_name"].str.split("- ").tolist()) #Split city and state into seperate columns.
    del df["area_name"] #now city and state are separated so we get rid of "area_name"
    return df

def fixAggravatedAssault(df):
    #Now, we drop rows where the number of aggravated assaults is negative.
    df = df.drop(df[(df["Aggravated assault"] < 0)].index)
    return df

def fixAllCrimes(df):
    #Now, we drop rows where total number of crimes, All Crimes, is negative.
    df = df.drop(df[(df["All Crimes"] < 0)].index)
    return df

def fixBurglaries(df):
    #Now, we drop rows where the number of burglaries is negative.
    df = df.drop(df[df.Burglary < 0].index)
    return df

def fixLarcenies(df):
    #Now, we drop rows where the number of larcenies is negative.
    df = df.drop(df[df.Larceny < 0].index)
    return df


def fixMVTs(df):
    #Now, we drop rows where the number of motor vehicle thefts is negative.
    df = df.drop(df[(df["Motor vehicle theft"] < 0)].index)
    return df


def fixMNM(df):
    #Now, we drop rows where the number of murder & nonnegligent manslaughter is negative.
    df = df.drop(df[(df["Murder and nonnegligent manslaughter"] < 0)].index)
    return df

def fixPropertyCrimes(df):
    #Now, we drop rows where the number of property crimes is negative.
    df = df.drop(df[(df["Property crime"] < 0)].index) 
    return df

def fixRapes(df):
    #Now, we drop rows where the number of rapes is negative.
    df = df.drop(df[(df["Rape (revised definition)"] < 0)].index)
    return df

def fixRobberies(df):
    #Now, we drop rows where the number of robberies is negative.
    df = df.drop(df[(df["Robbery"] < 0)].index)
    return df

def fixViolentCrimes(df):
    #Now, we drop rows where the number of violent crimes is negative.
    df = df.drop(df[(df["Violent crime"] < 0)].index)    
    return df

def fixCrimeRates(df):
    #Now we will make sure that the sum of all the crime rates for each category are within 2000 of the
    #number reported by All Crimes. 
    df = df.drop(df[2000 < (df["All Crimes"]) - (df["Aggravated assault"] + df["Burglary"] + df["Larceny"] + df["Motor vehicle theft"] + df["Murder and nonnegligent manslaughter"] + df["Property crime"] + df["Rape (revised definition)"] + df["Robbery"] + df["Violent crime"]  )].index)
    return df

def main():
    df = pd.read_csv("crime_rates.csv") #read in the input file into the dataframe.
    df = df.dropna() #dropping the rows with missing observations.
    
    #Dropping rows where the year is less than 2000 or greater than 2017
    df = fixYear(df)
    
    df = splitAreaName(df) #Split area_name into City and State
    
    #Now, we drop rows where the number of aggravated assaults is negative.
    df = fixAggravatedAssault(df)
    
    #Now, we drop rows where total number of crimes, All Crimes, is negative.
    df = fixAllCrimes(df)
    
    #Now, we drop rows where the number of burglaries is negative.
    df = fixBurglaries(df)
    
    #Now, we drop rows where the number of larcenies is negative.
    df = fixLarcenies(df)
    
    #Now, we drop rows where the number of motor vehicle thefts is negative.
    df = fixMVTs(df)
    
    #Now, we drop rows where the number of murder & nonnegligent manslaughter is negative.
    df = fixMNM(df)
    
    #Now, we drop rows where the number of property crimes is negative.
    df = fixPropertyCrimes(df)    
    
    #Now, we drop rows where the number of rapes is negative.
    df = fixRapes(df)
    
    #Now, we drop rows where the number of robberies is negative.
    df = fixRobberies(df)
    
    #Now, we drop rows where the number of violent crimes is negative.
    df = fixViolentCrimes(df)
    
    #Now we will make sure that the sum of all the crime rates for each category are within 2000 of the
    #number reported by All Crimes. 
    df = fixCrimeRates(df)
    
    df.to_csv("crime_rates_CLEANED.csv")
    
if __name__ == "__main__":
    main()   
    
