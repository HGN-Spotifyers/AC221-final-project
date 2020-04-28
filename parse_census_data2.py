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


# ========= SAVE OUTPUTS =========

clean_age_total.to_csv(base_out_path + 'age_information_total.csv', index = False)
clean_age_male.to_csv(base_out_path + 'age_information_male.csv', index = False)
clean_age_female.to_csv(base_out_path + 'age_information_female.csv', index = False)

clean_employment.to_csv(base_out_path + 'employment_status.csv', index = False)
