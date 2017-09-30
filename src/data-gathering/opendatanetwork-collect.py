"""
	Author: Kornraphop Kawintiranon
	Email : kk1155@georgetown.edu
"""
##### Purpose
# - Collect data from opendatanetwork API
# - https://www.opendatanetwork.com

import json
import requests
import numpy as np
import pandas as pd
import os.path
from states import states 	# From states.py

# Globle variables
APP_TOKEN="cQovpGcdUT1CSzgYk0KPYdAI0"

##### Get areas available from the API using search
# entity_name = word for search, typically use State such as VA, MD, DC, CA, etc.
# area_types = specific scope of area result - ["region.msa", "region.county", "region.place"]
def getAreas(entity_name, area_types=[]):
	# Setup API caller
	url = "http://api.opendatanetwork.com/entity/v1"
	payload = {
		"app_token" : APP_TOKEN,
		"entity_name" : entity_name
	}

	# Response from API
	res = requests.get(url, params=payload)

	# If call API success
	if res.status_code == 200:
		areas = res.json()["entities"]
		scoped_areas = []

		# If specify area types
		if len(area_types) > 0:
			for area in areas:
				if area["type"] in area_types:
					scoped_areas.append(area)
			return scoped_areas

		# Return areas with all types
		else:
			return areas

	# API failed
	return []


##### Gather Education data by specific area
def getGraduationRates(area_id, year):
	# Setup API caller
	url = "http://api.opendatanetwork.com/data/v1/values"
	payload = {
		"app_token" : APP_TOKEN,
		"describe" : "false",
		"format" : "",
		"variable" : "education.graduation_rates",
		"entity_id" : area_id,
		"forecast" : 0,
		"year" : year
	}

	# Response from API
	res = requests.get(url, params=payload)

	# If call API success
	if res.status_code == 200:
		res = res.json()["data"][1:]	# First element contains just variable names
		return res
	else:
		return []


def collectGraduationRates(state_abbr=[]):
	# Variables
	graduation_rates_header = ["area_id", "area_name", "area_type", "year"]
	graduation_rate_types = ["percent_associates_degree", "percent_bachelors_degree_or_higher", "percent_graduate_or_professional_degree", "percent_high_school_graduate_or_higher", "percent_less_than_9th_grade"]

	# Delete all old data
	graduation_rates_file_name = "graduation_rates.csv"
	with open(graduation_rates_file_name, 'w') as f:
		f.write(",".join(graduation_rates_header + graduation_rate_types) + '\n')


	count = 0
	graduation_rates_count = 0
	for state in list(states.keys()):
		if len(state_abbr) > 0 and not state in state_abbr: continue	# If specify states
		areas = getAreas(state, ["region.place"])
		count += len(areas)

		for area in areas:
			for year in range(2013, 2018):
				print("Gathering - State: " + state + " Area: " + area["name"] + " Year: " + str(year))
				
				##### Graduation Rates
				graduation_rates = getGraduationRates(area["id"], year)
				if len(graduation_rates) > 0:	# Not empty
					graduation_rates_count += 1
					row = [str(area["id"]), area["name"], area["type"], str(year)]
					rates = [""] * len(graduation_rate_types)
					for graduation_rate in graduation_rates:
						rate_name = graduation_rate[0]

						# There are 5 graduation rate types in general
						for idx, graduation_rate_type in enumerate(graduation_rate_types):
							if rate_name == graduation_rate_type: rates[idx] = str(graduation_rate[1])
				
					# Write to file
					with open(graduation_rates_file_name, 'a') as f:
						row = list(map((lambda x: x.replace(",", "-")), row))	# Remove ',' in area names
						line = ",".join(row + rates)
						f.write(line+"\n")

	# General Summary
	print("\n########## General Summary ##########")
	print("Place number in US: " + str(count))
	print("Place with graduation_rates number in US: " + str(graduation_rates_count))


def getCrimeCounts(area_id, year):
	# Setup API caller
	url = "http://api.opendatanetwork.com/data/v1/values"
	payload = {
		"app_token" : APP_TOKEN,
		"describe" : "false",
		"format" : "",
		"variable" : "crime.fbi_ucr.count",
		"entity_id" : area_id,
		"forecast" : 0,
		"year" : year
	}

	# Response from API
	res = requests.get(url, params=payload)

	# If call API success
	if res.status_code == 200:
		res = res.json()["data"][1:]	# First element contains just variable names
		return res
	else:
		return []


def collectCrimeCounts(state_abbr=[]):
	# Variables
	crime_counts_header = ["area_id", "area_name", "area_type", "year"]
	crime_count_types = ["Aggravated assault", "All Crimes", "Burglary", "Larceny", "Motor vehicle theft", "Murder and nonnegligent manslaughter", "Property crime", "Rape (revised definition)", "Robbery", "Violent crime"]

	# Delete all old data
	crime_counts_file_name = "crime_counts.csv"
	with open(crime_counts_file_name, 'w') as f:
		f.write(",".join(crime_counts_header + crime_count_types) + '\n')

	place_in_us_count = 0
	crime_counts_count = 0	# How many places we found for crime counts
	for state in list(states.keys()):
		if len(state_abbr) > 0 and not state in state_abbr: continue	# If specify states
		areas = getAreas(state, ["region.place"])
		place_in_us_count += len(areas)

		for area in areas:
			for year in range(2000, 2018):
				print("Gathering - State: " + state + " Area: " + area["name"] + " Year: " + str(year))

				##### Crime Counts
				crime_counts = getCrimeCounts(area["id"], year)
				if len(crime_counts) > 0:	# Not empty
					crime_counts_count += 1
					row = [str(area["id"]), area["name"], area["type"], str(year)]
					crime_counts_row = [""] * len(crime_count_types)

					for crime_count in crime_counts:
						crime_name = crime_count[0]

						for idx, crime_count_type in enumerate(crime_count_types):
							if crime_name == crime_count_type: crime_counts_row[idx] = str(crime_count[1])

					# Write to file
					with open(crime_counts_file_name, 'a') as f:
						row = list(map((lambda x: x.replace(",", "-")), row))	# Remove ',' in area names
						line = ",".join(row + crime_counts_row)
						f.write(line+"\n")

	# General Summary
	print("\n########## General Summary ##########")
	print("Place number in US: " + str(place_in_us_count))
	print("Place with crime_counts number in US: " + str(crime_counts_count))


def getCrimeRates(area_id, year):
	# Setup API caller
	url = "http://api.opendatanetwork.com/data/v1/values"
	payload = {
		"app_token" : APP_TOKEN,
		"describe" : "false",
		"format" : "",
		"variable" : "crime.fbi_ucr.rate",
		"entity_id" : area_id,
		"forecast" : 0,
		"year" : year
	}

	# Response from API
	res = requests.get(url, params=payload)

	# If call API success
	if res.status_code == 200:
		res = res.json()["data"][1:]	# First element contains just variable names
		return res
	else:
		return []


def collectCrimeRates(state_abbr=[]):
	# Variables
	crime_rates_header = ["area_id", "area_name", "area_type", "year"]
	crime_rate_types = ["Aggravated assault", "All Crimes", "Burglary", "Larceny", "Motor vehicle theft", "Murder and nonnegligent manslaughter", "Property crime", "Rape (revised definition)", "Robbery", "Violent crime"]

	# Delete all old data
	crime_rates_file_name = "crime_rates.csv"
	with open(crime_rates_file_name, 'w') as f:
		f.write(",".join(crime_rates_header + crime_rate_types) + '\n')

	place_in_us_count = 0
	crime_rates_count = 0	# How many places we found for crime rates
	for state in list(states.keys()):
		if len(state_abbr) > 0 and not state in state_abbr: continue	# If specify states
		areas = getAreas(state, ["region.place"])
		place_in_us_count += len(areas)

		for area in areas:
			for year in range(2000, 2018):
				print("Gathering - State: " + state + " Area: " + area["name"] + " Year: " + str(year))

				##### Crime Rates
				crime_rates = getCrimeRates(area["id"], year)
				if len(crime_rates) > 0:	# Not empty
					crime_rates_count += 1
					row = [str(area["id"]), area["name"], area["type"], str(year)]
					crime_rates_row = [""] * len(crime_rate_types)

					for crime_rate in crime_rates:
						crime_name = crime_rate[0]

						for idx, crime_rate_type in enumerate(crime_rate_types):
							if crime_name == crime_rate_type: crime_rates_row[idx] = str(crime_rate[1])

					# Write to file
					with open(crime_rates_file_name, 'a') as f:
						row = list(map((lambda x: x.replace(",", "-")), row))	# Remove ',' in area names
						line = ",".join(row + crime_rates_row)
						f.write(line+"\n")

	# General Summary
	print("\n########## General Summary ##########")
	print("Place number in US: " + str(place_in_us_count))
	print("Place with crime_rates number in US: " + str(crime_rates_count))


def getEarningInfo(area_id, year):
	# Setup API caller
	url = "http://api.opendatanetwork.com/data/v1/values"
	payload = {
		"app_token" : APP_TOKEN,
		"describe" : "false",
		"format" : "",
		"variable" : "jobs.earnings",
		"entity_id" : area_id,
		"forecast" : 0,
		"year" : year
	}

	# Response from API
	res = requests.get(url, params=payload)

	# If call API success
	if res.status_code == 200:
		res = res.json()["data"][1:]	# First element contains just variable names
		return res
	else:
		return []


def collectEarningInfo(state_abbr=[]):
	# Variables
	earning_info_header = ["area_id", "area_name", "area_type", "year"]
	earning_info_types = ["female_full_time_median_earnings", "female_median_earnings", 
		"male_full_time_median_earnings", "male_median_earnings", "median_earnings",
		"median_earnings_bachelor_degree", "median_earnings_graduate_or_professional_degree",
		"median_earnings_high_school", "median_earnings_less_than_high_school", "median_earnings_some_college_or_associates",
		"percent_with_earnings_10000_to_14999", "percent_with_earnings_15000_to_24999", "percent_with_earnings_1_to_9999",
		"percent_with_earnings_25000_to_34999", "percent_with_earnings_35000_to_49999", "percent_with_earnings_50000_to_64999",
		"percent_with_earnings_65000_to_74999", "percent_with_earnings_75000_to_99999", "percent_with_earnings_over_100000"]

	# Delete all old data
	earning_info_file_name = "earning_info.csv"
	with open(earning_info_file_name, 'w') as f:
		f.write(",".join(earning_info_header + earning_info_types) + '\n')

	place_in_us_count = 0
	earning_info_count = 0	# How many places we found for earning info
	for state in list(states.keys()):
		if len(state_abbr) > 0 and not state in state_abbr: continue	# If specify states
		areas = getAreas(state, ["region.place"])
		place_in_us_count += len(areas)

		for area in areas:
			for year in range(2000, 2018):
				print("Gathering - State: " + state + " Area: " + area["name"] + " Year: " + str(year))

				##### Earning Info
				earning_infos = getEarningInfo(area["id"], year)
				if len(earning_infos) > 0:	# Not empty
					earning_info_count += 1
					row = [str(area["id"]), area["name"], area["type"], str(year)]
					earning_info_row = [""] * len(earning_info_types)

					for earning_info in earning_infos:
						earning_info_name = earning_info[0]

						for idx, earning_info_type in enumerate(earning_info_types):
							if earning_info_name == earning_info_type: earning_info_row[idx] = str(earning_info[1])

					# Write to file
					with open(earning_info_file_name, 'a') as f:
						row = list(map((lambda x: x.replace(",", "-")), row))	# Remove ',' in area names
						line = ",".join(row + earning_info_row)
						f.write(line+"\n")

	# General Summary
	print("\n########## General Summary ##########")
	print("Place number in US: " + str(place_in_us_count))
	print("Place with earning_info number in US: " + str(earning_info_count))

def main():
	# Specify states by using params list such as ["VA", "DC", ...]
	# If not assign params, default is to use every states
	
	# collectGraduationRates()			
	# collectCrimeCounts()
	# collectCrimeRates(["DC"])				# Crime rate per 100k people
	collectEarningInfo(["DC"])

















if __name__ == "__main__":
	main()