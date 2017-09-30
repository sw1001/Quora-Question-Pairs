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


def collectGraduationRates():
	# Example
	# areas = getAreas("DC", ["region.place"])
	# for area in areas:
	# 	print(getGraduationRates(area["id"], "2013"))

	# Variables
	graduation_rates_header = ["area_id", "area_name", "area_type", "year"]
	graduation_rate_types = ["percent_associates_degree", "percent_bachelors_degree_or_higher", "percent_graduate_or_professional_degree", "percent_high_school_graduate_or_higher", "percent_less_than_9th_grade"]

	# Delete all old data
	graduation_rates_file_name = "graduation_rates.csv"
	if os.path.isfile(graduation_rates_file_name):
		os.remove(graduation_rates_file_name)
		with open(graduation_rates_file_name, 'w') as f:
			f.write(",".join(graduation_rates_header + graduation_rate_types) + '\n')


	count = 0
	graduation_rates_count = 0
	for state in list(states.keys()):
		# if state != "DC": continue
		areas = getAreas(state, ["region.place"])
		count += len(areas)

		for area in areas:
			for year in range(2013, 2018):
				print("Gathering - State: " + state + " Area: " + area["name"] + " Year: " + str(year))
				
				##### Graduation Rates
				graduation_rates = getGraduationRates(area["id"], year)
				graduation_rates_count += len(graduation_rates)
				if len(graduation_rates) > 0:	# Not empty
					row = [str(area["id"]), area["name"], area["type"], str(year)]
					rates = [None] * len(graduation_rate_types)
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
	print("Place with graduation_rates number in US: " + str(count))


def main():
	collectGraduationRates()
	



	














if __name__ == "__main__":
	main()