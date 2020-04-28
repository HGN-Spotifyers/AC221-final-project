import os
import numpy as np
import pandas as pd

from parsing_utilities import unpack_multi_index


# Check for input and output folder
all_folders = os.listdir('data') 

if not 'census_data' in all_folders:
    raise Exception('Error: Missing input folder - ensure data is in data/census_data/')
    
if not 'clean_census_data' in all_folders:
    raise Exception('Error: Missing input folder - ensure empty output data folder exists:  data/clean_census_data/')


# ========= USER SETUP ============
base_path = 'data/census_data/'

age_file = 'population_age_sex_by_county_usa_18.csv'
employment_file = 'employment_status_by_county_usa_18.csv'
computer_internet_file = 'computer_and_internet_by_county_usa_18.csv'
urban_rural_file = 'NCHSURCodes2013.xlsx'

base_out_path = 'data/clean_census_data/'


# ========= HELPER FUNCTIONS ============

# ========= DATA PROCESSING ============

#############
# Age Table #
#############

age_df = pd.read_csv(base_path + age_file)

clean_age = unpack_multi_index(age_df)
clean_age_id = clean_age[[("id"),("Geographic Area Name")]]
clean_age_id.columns = ["id","Geographic Area Name"]
clean_age_id.columns.name = None

# Get age breakdowns for total population:
clean_age_total  = clean_age[("Estimate","Total" ,"Total population","SELECTED AGE CATEGORIES",)]
clean_age_male   = clean_age[("Estimate","Male"  ,"Total population","SELECTED AGE CATEGORIES",)]
clean_age_female = clean_age[("Estimate","Female","Total population","SELECTED AGE CATEGORIES",)]

# Add identifier columns:
clean_age_total = pd.concat([clean_age_id,clean_age_total],axis=1,sort=False)
clean_age_male = pd.concat([clean_age_id,clean_age_male],axis=1,sort=False)
clean_age_female = pd.concat([clean_age_id,clean_age_female],axis=1,sort=False)

# Remove column index name:
clean_age_total.columns.name = None
clean_age_male.columns.name = None
clean_age_female.columns.name = None

####################
# Employment Table #
####################

employment_df = pd.read_csv(base_path + employment_file)

clean_employment = unpack_multi_index(employment_df)
clean_employment_id = clean_employment[[("id"),("Geographic Area Name")]]
clean_employment_id.columns = ["id","Geographic Area Name"]
clean_employment_id.columns.name = None

# Get employment breakdowns for total population:
clean_employment= clean_employment[("Estimate","Labor Force Participation Rate","Population 16 years and over")]
clean_employment = clean_employment[[""]]
clean_employment.columns = ["Labor Force Participation Rate"]

# Add identifier columns:
clean_employment = pd.concat([clean_employment_id,clean_employment],axis=1,sort=False)

###################
# Smartphone data #
###################

computer_internet_df = pd.read_csv(base_path + computer_internet_file)

clean_computer_internet = unpack_multi_index(computer_internet_df)
clean_computer_internet_id = clean_computer_internet[[("id"),("Geographic Area Name")]]
clean_computer_internet_id.columns = ["id","Geographic Area Name"]
clean_computer_internet_id.columns.name = None

# Get employment breakdowns for total population:
col_total_households = clean_computer_internet[(
    "Estimate","Total","Total households",
)].replace('N',np.nan).astype(int)
col_has_smartphone = clean_computer_internet[(
    "Estimate","Total","TYPES OF COMPUTER",
    "Has one or more types of computing devices","Smartphone",""
)].replace('N',np.nan).astype(int)

clean_computer_internet = clean_computer_internet_id
clean_computer_internet['total_households'] = col_total_households
clean_computer_internet['smartphone_households'] = col_has_smartphone
clean_computer_internet['smartphone_ownership'] = clean_computer_internet['smartphone_households']/clean_computer_internet['total_households']


####################
# Urban/Rural data #
####################

urban_rural_df = pd.read_excel(base_path + urban_rural_file)

def _convert_code(number):
    return {
        1 : 'Large central metro',
        2 : 'Large fringe metro',
        3 : 'Medium metro',
        4 : 'Small metro',
        5 : 'Micropolitan',
        6 : 'Noncore',
    }[number]

def _format_id(number):
    return '0500000US{:0>5}'.format(number)

clean_urban_rural = urban_rural_df.copy()

clean_urban_rural.insert(0,'id',clean_urban_rural['FIPS code'].apply(_format_id))
clean_urban_rural['type_2013'] = clean_urban_rural['2013 code'].apply(_convert_code)


# ========= SAVE OUTPUTS =========

clean_age_total.to_csv(base_out_path + 'age_information_total.csv', index = False)
clean_age_male.to_csv(base_out_path + 'age_information_male.csv', index = False)
clean_age_female.to_csv(base_out_path + 'age_information_female.csv', index = False)

clean_employment.to_csv(base_out_path + 'employment_status.csv', index = False)

clean_computer_internet.to_csv(base_out_path + 'computer_internet.csv', index = False)

clean_urban_rural.to_csv(base_out_path + 'urban_rural_by_county.csv', index = False)
