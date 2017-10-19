# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 02:13:29 2017

In this file, we attempt to cleaen graduation_rates. We make sure the year 
and the percentages of each category is consistent. In addition, we make 
sure that they all add up to 100 percent. 

@author: Armaan Khullar
"""


import pandas as pd
import sys

def main():
    df = pd.read_csv("graduation_rates.csv") #read in the input file into the dataframe.
    df = df.dropna() #dropping the rows with missing observations.
    
    #Dropping rows where the year is less than 2000.
    df = df.drop(df[(df["year"]) < 2000].index)
    
    #Dropping rows where the year greater than 2017, the current year.
    df = df.drop(df[(df["year"]) > 2017].index)
        
    #Dropping rows with % of associates degrees less than 0
    df = df.drop(df[(df["percent_associates_degree"]) < 0].index)
    
    #Dropping rows with % of bachelors degrees or higher is less than 0
    df = df.drop(df[(df["percent_bachelors_degree_or_higher"]) < 0].index)
    
    #Dropping rows with % of graduate_or_professional_degree less than 0
    df = df.drop(df[(df["percent_graduate_or_professional_degree"]) < 0].index)
    
    #Dropping rows with % of high_school_graduate_or_higher less than 0
    df = df.drop(df[(df["percent_high_school_graduate_or_higher"]) < 0].index)
    
    #Dropping rows with % of less_than_9th_grade less than 0
    df = df.drop(df[(df["percent_less_than_9th_grade"]) < 0].index)
    
    #Dropping all rows where the percentages of categories of earnings don't add up to 100.
    df = df.drop(df[100 != (df["percent_associates_degree"] + df["percent_bachelors_degree_or_higher"] + df["percent_graduate_or_professional_degree"] + df["percent_high_school_graduate_or_higher"] + df["percent_less_than_9th_grade"]   )].index)
    
    #Splitting the area_name column into City and State.
    df["City"], df["State"] = zip(*df["area_name"].str.split("- ").tolist())     #Splitting the area_name column into City and State.
    del df["area_name"] #now city and state are separated so we get rid of "City, State"
    
    
    #Finally outputting the dataframe to a csv file.
    df.to_csv("graduation_rates_CLEANED.csv")

if __name__ == "__main__":
    main()


