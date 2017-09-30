"""
	Author: Kornraphop Kawintiranon
	Email : kk1155@georgetown.edu
"""
##### Purpose 
# - Collect data from opendatanetwork API
# - https://www.opendatanetwork.com

import json
import requests

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
	return None



def main():
	# Example
	print(getAreas("DC", ["region.msa", "region.place"]))


if __name__ == "__main__":
	main()