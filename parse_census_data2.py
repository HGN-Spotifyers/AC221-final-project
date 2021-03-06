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

population_file = 'ACSST1Y2018.S0101/ACSST1Y2018.S0101_data_with_overlays.csv'
income_file = 'ACSST1Y2018.S1901/ACSST1Y2018.S1901_data_with_overlays.csv'
employment_file = 'ACSST1Y2018.S2301/ACSST1Y2018.S2301_data_with_overlays.csv'
computer_internet_file = 'ACSST1Y2018.S2801/ACSST1Y2018.S2801_data_with_overlays.csv'
urban_rural_file = 'NCHSURCodes2013.xlsx'
commuting_file = 'Residence County to Workplace County Commuting Flows.xlsx'

base_out_path = 'data/clean_census_data/'


# ========= HELPER FUNCTIONS ============

# ========= DATA PROCESSING ============

#############
# Age Table #
#############

population_df = pd.read_csv(base_path + population_file)

clean_population = unpack_multi_index(population_df)
clean_population_id = clean_population[[("id"),("Geographic Area Name")]]
clean_population_id.columns = ["id","Geographic Area Name"]
clean_population_id.columns.name = None

# Get population breakdown by age and gender:
clean_population_all       = clean_population[("Estimate","Total" ,"Total population","SELECTED AGE CATEGORIES",)]
clean_population_male      = clean_population[("Estimate","Male"  ,"Total population","SELECTED AGE CATEGORIES",)]
clean_population_female    = clean_population[("Estimate","Female","Total population","SELECTED AGE CATEGORIES",)]
clean_population_total_col = clean_population[("Estimate","Total" ,"Total population","",)]

# Prepend subcategory to column name:
clean_population_male.columns = ["Male - {}".format(col) for col in clean_population_male.columns]
clean_population_female.columns = ["Female - {}".format(col) for col in clean_population_male.columns]

# Add identifier columns:
clean_population = pd.concat([clean_population_id,clean_population_all,clean_population_male,clean_population_female],axis=1,sort=False)
clean_population.insert(0,"Total Population",clean_population_total_col)

# Remove column index name:
clean_population.columns.name = None

################
# Income Table #
################

income_df = pd.read_csv(base_path + income_file)

clean_income = unpack_multi_index(income_df)
clean_income_id = clean_income[[("id"),("Geographic Area Name")]]
clean_income_id.columns = ["id","Geographic Area Name"]
clean_income_id.columns.name = None

# Get income breakdowns for total population:
clean_income_breakdown = clean_income[("Estimate","Households","Total")]
clean_income_median_col = clean_income[("Estimate","Households","Median income (dollars)")]
clean_income_mean_col = clean_income[("Estimate","Households","Mean income (dollars)")]

# Add identifier columns:
clean_income = pd.concat([clean_income_id,clean_income_breakdown],axis=1,sort=False)
clean_income["Median income (dollars)"] = clean_income_median_col
clean_income["Mean income (dollars)"] = clean_income_mean_col

####################
# Employment Table #
####################

employment_df = pd.read_csv(base_path + employment_file)

clean_employment = unpack_multi_index(employment_df)
clean_employment_id = clean_employment[[("id"),("Geographic Area Name")]]
clean_employment_id.columns = ["id","Geographic Area Name"]
clean_employment_id.columns.name = None

# Get employment breakdowns for total population:
clean_employment = clean_employment[("Estimate","Labor Force Participation Rate","Population 16 years and over")]
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

####################
# Urban/Rural data #
####################

commuting_df = pd.read_excel('data/census_data/Residence County to Workplace County Commuting Flows.xlsx',skiprows=6)
commuting_df = commuting_df.iloc[:-2]

clean_commuting = commuting_df.copy()

def _is_same_county(row):
    try:
        if int(row['State FIPS Code'])!=int(row['State FIPS Code.1']):
            return False
        elif int(row['County FIPS Code'])!=int(row['County FIPS Code.1']):
            return False
        else:
            return True
    except:
        return False

clean_commuting['same_county'] = clean_commuting.apply(_is_same_county,axis=1)

clean_commuting = clean_commuting.groupby([
    'State FIPS Code','County FIPS Code','State Name','County Name','same_county',
])[['Workers in Commuting Flow']].sum()

clean_commuting = clean_commuting.unstack(-1).reset_index()
clean_commuting.columns = [
    'State FIPS Code','County FIPS Code','State Name','County Name','number_work_in_county','number_work_out_of_county'
]
clean_commuting['id'] = [
    "0500000US{:0>2}{:0>3}".format(int(state),int(county))
    for state,county in zip(clean_commuting['State FIPS Code'],clean_commuting['County FIPS Code'])
]
clean_commuting['number_work_in_county'] = clean_commuting['number_work_in_county'].fillna(0).astype(int)
clean_commuting['number_work_out_of_county'] = clean_commuting['number_work_out_of_county'].fillna(0).astype(int)

clean_commuting = clean_commuting[[
    'id','State Name','County Name','number_work_in_county','number_work_out_of_county',
]]


# ========= SAVE OUTPUTS =========

clean_population.to_csv(base_out_path + 'ACSST1Y2018_S0101_population.csv', index = False)

clean_income.to_csv(base_out_path + 'ACSST1Y2018_S2503_income.csv', index = False)

clean_employment.to_csv(base_out_path + 'ACSST1Y2018_S2301_employment.csv', index = False)

clean_computer_internet.to_csv(base_out_path + 'ACSST1Y2018_S2801_internet.csv', index = False)

clean_urban_rural.to_csv(base_out_path + 'NCHSURCodes2013_urbanrural.csv', index = False)

clean_commuting.to_csv(base_out_path + 'ACSCommutingFlows.csv', index = False)


