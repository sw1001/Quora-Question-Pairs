"""
File: GetZillowData.py
Author: Zhuoran Wu
Email: zw118@georgetown.edu
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
import re


def get_zestimate_zpi(
        zwisd,
        zpid):
    """
    Get House Info via GetZestimateAPI
    :param ZWISD: The API KEY for API
    :param zpid: The id for single house
    :return: Response from Request
    """
    get_zestimate_zpi_baseurl = 'http://www.zillow.com/webservice/GetZestimate.htm'
    get_zestimate_zpi_urlpost = {'zws-id': zwisd,
                                 'zpid': str(zpid)}
    response = requests.get(get_zestimate_zpi_baseurl, get_zestimate_zpi_urlpost)
    return response

ZWSID1 = 'X1-ZWz1g08hy6pukr_3csw9'
ZWSID2 = 'X1-ZWz1g0k86stlaj_1pt0f'
ZWSID3 = 'X1-ZWz18zy8rmtc7f_1r7kw'
ZWSID4 = 'X1-ZWz1bgfgbsq58r_1llb0'

HouseInfoColumn = ['zpid',
                   'street',
                   'zipcode',
                   'city',
                   'state',
                   'latitude',
                   'longitude',
                   'amount',
                   'last-updated',
                   'valueChange']

HouseDetailColumn = ['zpid',
                     'Yearbuilt',
                     'lotsizeSqFt',
                     'finishedSqFt',
                     'latitude',
                     'longitude',
                     'Bathrooms',
                     'Bedroomes',
                     'LastSoldPrice']

data = pd.read_csv('../input/Zillow_House_zpid_list_small.csv')

data = data.reindex(columns=HouseInfoColumn)
data[HouseInfoColumn[1:]] = ""

for zpid in data.zpid:
    response = get_zestimate_zpi(ZWSID1, zpid)
    contents = re.sub('<!--(.*?)-->', '', str(response.text))
    contents = re.sub(':zestimate+.*xsd/Zestimate.xsd"', '', contents)
    contents = re.sub(':zestimate', '', contents)
    print(contents)
    root = ET.fromstring(contents)

    if root[1][1].text != '0':
        continue

    # root[1][1].text       message code
    # root[2][0].text       zpid
    # root[2][2][0].text    street
    # root[2][2][1].text    zipcode
    # root[2][2][2].text    city
    # root[2][2][3].text    state
    # root[2][2][4].text    latitude
    # root[2][2][5].text    longitude
    # root[2][3][0].text    amount
    # root[2][3][1].text    last-updated date
    # root[2][3][3].text    valueChange
    data.loc[data['zpid'] == zpid, 'street'] = root[2][2][0].text
    data.loc[data['zpid'] == zpid, 'zipcode'] = root[2][2][1].text
    data.loc[data['zpid'] == zpid, 'city'] = root[2][2][2].text
    data.loc[data['zpid'] == zpid, 'state'] = root[2][2][3].text
    data.loc[data['zpid'] == zpid, 'latitude'] = root[2][2][4].text
    data.loc[data['zpid'] == zpid, 'longitude'] = root[2][2][5].text
    data.loc[data['zpid'] == zpid, 'amount'] = root[2][3][0].text
    data.loc[data['zpid'] == zpid, 'last-updated'] = root[2][3][1].text
    data.loc[data['zpid'] == zpid, 'valueChange'] = root[2][3][3].text


data.to_csv('../input/Zillow_House_Info_1000_Sample.csv', index=False)

