"""
File: zipcode.py
Author: Zhuoran Wu
Email: zw118@georgetown.edu
"""

import pandas as pd
import math
import json
import googlemaps

data_filename = 'temp.csv'
GoogleAPIKey = 'AIzaSyCSRNt6-gf8XiTCUKP2udaW6WJZWh7ykGE'


def transfer_zipcode_to_city_state(dataframe):
    """

    :param dataframe:
    :return:
    """
    dataframe['city'] = ""
    dataframe['county'] = ""
    dataframe['state'] = ""

    gmaps = googlemaps.Client(key=GoogleAPIKey)

    for zpid in dataframe['zpid']:
        code = dataframe.loc[dataframe['zpid'] == zpid]['zipcode']
        print(code)
        if math.isnan(float(code)):
            continue
        result = gmaps.geocode(str(int(code)))

        if len(result) == 0:
            continue
        lens = len(result[0]['address_components'])
        if result[0]['address_components'][lens - 1]['short_name'] != 'US':
            continue
        print(result)
        city = result[0]['address_components'][1]['short_name']
        county = result[0]['address_components'][2]['short_name']
        state = result[0]['address_components'][3]['short_name']

        dataframe.loc[dataframe['zpid'] == zpid, 'city'] = city
        dataframe.loc[dataframe['zpid'] == zpid, 'county'] = county
        dataframe.loc[dataframe['zpid'] == zpid, 'state'] = state

    return dataframe


def main():
    dataframe = pd.read_csv('../input/' + data_filename)
    transfer_zipcode_to_city_state(dataframe)
    dataframe.to_csv('../input/' + data_filename)


if __name__ == "__main__":
    main()