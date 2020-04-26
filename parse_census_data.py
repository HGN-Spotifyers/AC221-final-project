import pandas as pd
import numpy as np
import re
import os

# Check for input and output folder
all_folders = os.listdir('data') 

if not 'census_data' in all_folders:
    raise Exception('Error: Missing input folder - ensure data is in data/census_data/')
    
if not 'clean_census_data' in all_folders:
    raise Exception('Error: Missing input folder - ensure empty output data folder exists:  data/clean_census_data/')


# ========= USER SETUP ============
base_path = 'data/census_data/'

commuting_file = 'commuting_by_county_usa_18.csv'
employment_file = 'employment_by_county_usa_18.csv'
income_file = 'income_by_county_usa_18.csv'
population_file = 'population_density_by_county_usa_10-18.csv'
poverty_file = 'poverty_by_county_usa_18.csv'


base_out_path = 'data/clean_census_data/'


# ========= HELPER FUNCTIONS ============

def clean_census_columns(data, col_words_to_remove):
    '''
    Given the usual flat format census data, reset headers and then remove any columns which include 
    the words specified in the list of words to remove.
    
    inputs: data = census_data_frame (pandas df)
            col_words_to_remove = list
    
    ouputs: clean data_frame (pandas df)
    '''
    data2 = data.copy()
    
    data2.columns = data2.iloc[0,:]
    data2 = data2.iloc[1:,]
    
    for word in col_words_to_remove:
        # Regex to find and remove specific words
        columns_re = [re.search(word, x) for x in data2.columns]
        columns_to_keep = [False if column_re else True for column_re in columns_re]
        data2 = data2.iloc[:, columns_to_keep]

    return data2
    
    
def split_to_specific_tables(data, table_keywords, keep_index):
    '''
    Seeing as the census tables are compound and with mutliple sublevel headings
    
    Given a heading keyword - create a new table with just those columns
    
    NOTE: This will drop some of the column heading information.
    
    inputs: data = cleaned census data (pandas df)
            table_keywords = a string representing the columns to extract (string)
            keep_index = the position of the compound column name string to keep (aka which element
            between the !!'s (Total!!Household!!Income = [1] to keep 'Household')')
    
    outputs: clean data_frame (pandas df)
    '''
    data2 = data.copy()
    
    # Regex to find columns which are relevant to keyword
    columns_re = [re.search(table_keywords, x) for x in data2.columns]
    columns_to_keep = [True if column_re else False for column_re in columns_re]
    
    # ALways keep the id and geo columns
    columns_to_keep[0] = True
    columns_to_keep[1] = True
    
    data2 = data2.iloc[:, columns_to_keep]
    
    # Now clean all headers except the ID columns to remove unnecessary nested headers
    cur_headers = data2.columns
    id_headers = list(cur_headers[0:2])
    other_headers = cur_headers[2:]
    other_headers = [header.split('!!')[keep_index] for header in other_headers]
    for h in other_headers:
        id_headers.append(h)

    data2.columns = id_headers
    
    return data2



# ========= DATA PROCESSING ============

# Commuting Table
commuting_df = pd.read_csv(base_path + commuting_file)

clean_commuting = clean_census_columns(commuting_df, ['Margin', 'Male', 'Female'])
commute_place_of_work_out = split_to_specific_tables(clean_commuting, "PLACE OF WORK", keep_index=-1)
commute_travel_time_out = split_to_specific_tables(clean_commuting, "TRAVEL TIME TO WORK", keep_index=-1)


# Employment Table
employ_df = pd.read_csv(base_path + employment_file)

clean_employ = clean_census_columns(employ_df, ['Margin'])
employment_total_in_work_out = split_to_specific_tables(clean_employ, "Total", keep_index=-1)
employment_labour_force_perc_out = split_to_specific_tables(clean_employ, "Labor Force Participation Rate", keep_index=-1)
employment_population_ratio_out = split_to_specific_tables(clean_employ, "Employment/Population Ratio", keep_index=-1)


# Income Table
income_df = pd.read_csv(base_path + income_file)

clean_income = clean_census_columns(income_df, ['Margin'])
income_totals_household_out = split_to_specific_tables(clean_income, "Households", keep_index=-1)


# Population Table
population_df = pd.read_csv(base_path + population_file, encoding = 'latin')

population_df = population_df[population_df['YEAR'] == 11]
population_df_out = population_df[['STNAME', 'CTYNAME', 'TOT_POP']]
new_col = population_df['CTYNAME'] + ', ' + population_df['STNAME']
population_df['Geographic Area Name'] = new_col


# Poverty Table
poverty_df = pd.read_csv(base_path + poverty_file)

clean_poverty = clean_census_columns(poverty_df, ['Margin'])
poverty_df_out = split_to_specific_tables(clean_poverty, "Population for whom poverty status is determined", keep_index=-2)
poverty_df_out = poverty_df_out.iloc[:,0:5]


# ========= SAVE OUTPUTS =========
commute_place_of_work_out.to_csv(base_out_path + 'commute_place_of_work.csv', index = False)
commute_travel_time_out.to_csv(base_out_path + 'commute_travel_time.csv', index = False)

employment_total_in_work_out.to_csv(base_out_path + 'employment_total_in_work.csv', index = False)
employment_labour_force_perc_out.to_csv(base_out_path + 'employment_labour_force_perc.csv', index = False)
employment_population_ratio_out.to_csv(base_out_path + 'employment_population_ratio.csv', index = False)

income_totals_household_out.to_csv(base_out_path + 'income_totals_household.csv', index = False)

population_df_out.to_csv(base_out_path + 'population_totals.csv', index = False)
poverty_df_out.to_csv(base_out_path + 'poverty_information.csv', index = False)


