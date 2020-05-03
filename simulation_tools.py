import numpy as np
import pandas as pd
import itertools

from tqdm.notebook import tqdm


class Population:
    
    POPULATIONS = 0
    PEOPLE = 0
    
    def __init__(self,location_attributes,location_profiles,population_name=None,random_state=None):
        """
            Object that defines the enviroment of a simulation and generates a population.
            
            The location attribues are fixed for each location and are used for high-level calibration
                of the appoximate distributions of key features.
                
            The location profiles are used to determine characteristics of individuals generated
                in population subgroups of each location.
                
            :param location_attributes: pandas.DataFrame with the following fields:
                location_name : (string) location identifier.
                density : (string) rural or urban.
                population : (int) Number of residents.
                employment_rate : (float) Rate between 0 and 1 of employed (vs. unemployed) residents.
                wealth_rate : (float) Rate between 0 and 1 of high-income (vs. low-income) residents.
                
            :param location_profiles: pandas.DataFrame with the following fields:
                location_name : (string) location identifier.
                wealth_status : (boolean) 0 represents low income and 1 represents high income.
                employment_status : (string) 0 represents unemployed and 1 represents employed.
                phoneownership_rate : (float) Probability between 0 and 1 that a resident owns a cellpone.
                worktravel_baseline : (float) Distance in miles of baseline (mean) for work-related travel.
                socialtravel_baseline : (float) Distance in miles of baseline (mean) for social-related travel.
                grocerytravel_baseline : (float) Distance in miles of baseline (mean) for grocery-related travel.
                worktravel_std : (float) Standard deviation baseline for work-related travel.
                socialtravel_std : (float) Standard deviation baseline for social-related travel.
                grocerytravel_std : (float) Standard deviation baseline for grocery-related travel.
                
            :param population_name: (string) Unique identifier of simulation run, or None.
                
            :param random_state: integer representing the random state, or None.
                
            :return: Population object.
        """
        
        # Set random state:
        np.random.seed(random_state)
        
        # Increment counter and store simulation label
        Population.POPULATIONS += 1
        if population_name is None:
            population_name = "population{}".format(Population.POPULATIONS)
        self._population_name = population_name
        
        # Define lookup dictionaries:
        wealth_labels = {
            0 : "lowincome", False : "lowincome",
            1 : "highincome", True : "highincome",
        }
        employment_labels = {
            0 : "unemployed", False : "unemployed",
            1 : "employed", True : "employed",
        }
        
        # Verify and store input data:
        self._random_state = random_state
        self.location_attributes = location_attributes.copy()
        self.location_profiles = location_profiles.copy()
        self.check_location_profiles()
        
        # Define properties:
        people_calibration = []
        people_simulation = []
        
        # Assign properties:
        for i,attribute_row in self.location_attributes.iterrows():
            # Get location attributes:
            location_name = attribute_row['location_name']
            wealth_rate = attribute_row['wealth_rate']
            employment_rate = attribute_row['employment_rate']
            #lowincome_rate = (1-attribute_row['wealth_rate'])
            #highincome_rate = (attribute_row['wealth_rate'])
            #unemployed_rate = (1-attribute_row['employment_rate'])
            #employed_rate = (attribute_row['employment_rate'])
            # Iterate through relevant location profiles:
            location_profiles = self.location_profiles[self.location_profiles['location_name']==location_name]
            for i,profile_row in location_profiles.iterrows():
                # Build subgroup labels:
                wealth_status = profile_row['wealth_status']
                employment_status = profile_row['employment_status']
                wealth_label = wealth_labels[wealth_status]
                employment_label = employment_labels[employment_status]
                subgroup_label = "{}_{}".format(wealth_label,employment_label)
                # Determine subgroup population:
                #employment_prob = employment_rate if employment_status==1 else (1-employment_rate)
                #wealth_prob = wealth_rate if wealth_status==1 else (1-wealth_rate)
                subgroup_population = attribute_row['population']
                subgroup_population *= employment_rate if employment_status==1 else (1-employment_rate)
                subgroup_population *= wealth_rate if wealth_status==1 else (1-wealth_rate)
                # Generate people according to specified distributions:
                location_name = attribute_row['location_name']
                location_density = attribute_row['density']
                phoneownership_prob = profile_row['phoneownership_rate']
                worktravel_mean = profile_row['worktravel_mean']
                socialtravel_mean = profile_row['socialtravel_mean']
                grocerytravel_mean = profile_row['grocerytravel_mean']
                worktravel_variance = profile_row['worktravel_std']**2
                socialtravel_variance = profile_row['socialtravel_std']**2
                grocerytravel_variance = profile_row['grocerytravel_std']**2
                subgroup_population_rounded = int(np.round(subgroup_population,0))
                for p in range(subgroup_population_rounded):
                    Population.PEOPLE += 1
                    person_name = "person{}".format(Population.PEOPLE)
                    person = {
                        'person_name' : person_name,
                        'population_name' : self._population_name,
                        'location_name' : location_name,
                        'location_density' : location_density,
                        'wealth' : wealth_label,
                        'employment' : employment_label,
                        'phoneownership' : np.random.binomial(1,phoneownership_prob),
                        'worktravel' : max(0,np.random.normal(worktravel_mean,worktravel_variance)),
                        'socialtravel' : max(0,np.random.normal(socialtravel_mean,socialtravel_variance)),
                        'grocerytravel' : max(0,np.random.normal(grocerytravel_mean,grocerytravel_variance)),
                    }
                    people_simulation.append(person)
                people_calibration.append({
                    'population_name' : self._population_name,
                    'location_name' : location_name,
                    'location_density' : location_density,
                    'wealth' : wealth_label,
                    'employment' : employment_label,
                    'people' : subgroup_population,
                    'phoneownership' : subgroup_population*phoneownership_prob,
                    'worktravel' : subgroup_population*worktravel_mean,
                    'socialtravel' : subgroup_population*socialtravel_mean,
                    'grocerytravel' : subgroup_population*grocerytravel_mean,
                })
        people_simulation = pd.DataFrame(people_simulation)
        people_calibration = pd.DataFrame(people_calibration)
        
        # Compute census-like figures:
        census_simulation = self.build_census(people_simulation)
        census_calibration = self.build_census(people_calibration)
        
        # Store contructed values:
        self._people_simulation = people_simulation
        self._census_simulation = census_simulation
        self._people_calibration = people_calibration
        self._census_calibration = census_calibration
        
        # Unset random state:
        np.random.seed(None)
        
    def build_census(self,people,wide=False):
        """
            Convert list of people into census (long format or wide format).
        """
        group_cols = ['population_name','location_name','location_density','wealth','employment']
        value_cols = ['people','phoneownership','worktravel','socialtravel','grocerytravel']
        census = people.copy()
        if 'people' not in set(census.columns):
            census['people'] = 1
        census = census.groupby(group_cols)[value_cols].sum()
        census['travel'] = census[['worktravel','socialtravel','grocerytravel']].sum(axis=1)
        census = census.sort_index()
        if wide:
            raise NotImplementedError("Wide format is not implemented.")
        return census
        
    def check_location_profiles(self):
        """
            Check validity of subgroup profiles based on hardcoded rules.
        """
        location_profiles = self.location_profiles
        subgroup_cols = ['employment_status','wealth_status']
        assert pd.isnull(location_profiles).sum().sum()==0, "Location profile table has blank values."
        for location_name,subgroup_profiles in location_profiles.groupby('location_name'):
            # Get list of possible values in each grouping column:
            possible_values = []
            for col in subgroup_cols:
                vals = sorted(set(location_profiles[col]))
                possible_values.append( vals )
            # Get all possible combinations of grouping columns:
            possible_combos = list(itertools.product(*possible_values))
            possible_combos = [tuple(possible_combo) for possible_combo in possible_combos]
            # Make sure that each combination is represented:
            def subgroup_label(group_cols,group_vals):
                group_cols = np.array([group_cols]).flatten()
                group_vals = np.array([group_vals]).flatten()
                label = [
                    "{}={}".format(group_col,group_val)
                    for group_col,group_val in zip(group_cols,group_vals)
                ]
                label = ",".join(label)
                return label
            actual_combos = subgroup_profiles[subgroup_cols].to_records(index=False)
            actual_combos = [tuple(actual_combo) for actual_combo in actual_combos]
            for possible_combo in possible_combos:
                assert possible_combo in actual_combos, "Missing combination in {}: {}".format(
                    location_name,subgroup_label(subgroup_cols,possible_combo)
                )
            # Verify values:
            def verify_unique(value_col,condition_cols):
                if (condition_cols is None) or (len(condition_cols)==0):
                    # Unconditional:
                    vals = set(subgroup_profiles[value_col])
                    assert len(vals)==1, "Found multiple values for column {} : {}".format(
                        value_col,vals
                    )
                else:
                    # Conditional:
                    for g,grp in subgroup_profiles.groupby(condition_cols):
                        vals = set(grp[value_col])
                        assert len(vals)==1, "Found multiple values for column {} conditional on {}: {}".format(
                            value_col,subgroup_label(condition_cols,g),vals
                        )
            # Make sure that phoneownership rate depends only on wealth status (for this location):
            verify_unique('phoneownership_rate',['wealth_status'])
            # Make sure that worktravel baseline depends only on wealth and employment status (for this location):
            verify_unique('worktravel_mean',['wealth_status','employment_status'])
            verify_unique('worktravel_std',['wealth_status','employment_status'])
            # Make sure that social baseline depends only on wealth status (for this location):
            verify_unique('socialtravel_mean',['wealth_status'])
            verify_unique('socialtravel_std',['wealth_status'])
            # Make sure that grocerytravel baseline depends only on wealth status (for this location):
            verify_unique('grocerytravel_mean',['wealth_status'])
            verify_unique('grocerytravel_std',['wealth_status'])
    
    @property
    def population_name(self):
        """Label for the simulation."""
        return self._population_name.copy()
    
    @property
    def people(self):
        """Table of people generated by the simulation."""
        return self._people_simulation.copy()
        
    @property
    def census(self):
        """Census describing the people generated by the simulation"""
        return self._census_simulation.copy()
        
    @property
    def people_simulation(self):
        """Table of people generated by the simulation."""
        return self._people_simulation.copy()
        
    @property
    def census_simulation(self):
        """Census describing the people generated by the simulation"""
        return self._census_simulation.copy()
        
    @property
    def people_calibration(self):
        """Table representing the number of people in the target calibration."""
        return self._people_calibration.copy()
        
    @property
    def census_calibration(self):
        """Census describing the people in the target calibration."""
        return self._census_calibration.copy()
    
    def __str__(self):
        return "Population : {}".format(self._population_name)


class Transformation:
    
    def __init__(self,population,behavior='normal',random_state=None):
        
        """
            
            An object that represents the new data after a change in behavior.
        
            :param population: A Population object.
            
            :param behavior: A string indicating the transformed behavior:
            
                - "normal" : Baseline behavior.
                - "total_compliance" : All social and work travel are stopped (grocery travel continues).
                - "essential_workers" : Total compliance, except for a random subset of people 
                    who continue to have work travel.
                - "partial_workers" : A random subset of the population ceases all social and work travel
                    and the rest continue all travel as normal.
                    
        """
        
        valid_behaviors = ['normal','total_compliance','essential_workers','partial_compliance']
        assert behavior in valid_behaviors, "{} is not a valid behavior: {} .".format(
            behavior,", ".join(valid_behaviors)
        )
        
        # Set random state:
        np.random.seed(random_state)
        
        # Get/set helper values:
        N = len(population.people)
        people = population.people
        census = population.census_simulation
        
        if behavior=='normal':
        
            pass
            
        elif behavior=='total_compliance':
            
            people['worktravel'] = 0.0
            people['socialtravel'] = 0.0
            people['grocerytravel'] = people['grocerytravel']
            census = population.build_census(people)
            
        elif behavior=='essential_workers':
        
            essential_workers_proportion = 0.10
            people['essential'] = np.random.binomial(1,essential_workers_proportion,size=N)
            people['worktravel'] = np.where(people['essential']==1,people['worktravel'],0.0)
            people['socialtravel'] = 0.0
            people['grocerytravel'] = people['grocerytravel']
            census = population.build_census(people)
            
        elif behavior=='partial_compliance':
        
            partial_compliance_rate = 0.40
            people['compliant'] = np.random.binomial(1,partial_compliance_rate,size=N)
            people['worktravel'] = np.where(people['compliant']==1,0.0,people['worktravel'])
            people['socialtravel'] = 0.0
            people['grocerytravel'] = people['grocerytravel']
            census = population.build_census(people)
        
        # Store generated value:
        self._random_state = random_state
        self._population = population
        self._behavior = behavior
        self._people_transformed = people
        self._census_transformed = census
        
        # Unset random state:
        np.random.seed(None)
        
    @property
    def people_transformed(self):
        """Table representing the activity of the simulated people after transformation."""
        return self._people_transformed.copy()
        
    @property
    def census_transformed(self):
        """Census representing the activity of the simulated people after transformation."""
        return self._census_transformed.copy()
        
    @property
    def people(self):
        """Table representing the activity of the simulated people after transformation."""
        return self._people_transformed.copy()
        
    @property
    def census(self):
        """Census representing the activity of the simulated people after transformation."""
        return self._census_transformed.copy()
    
    def __str__(self):
        return "Transformation : {}".format(self._behavior)

class Metric:
    
    def __init__(self,before,after,metric_method='median_person',random_state=None):
        
        """
            
            An object that represents a measure derived from population data.
                The `before` and `after` objects must have `people` and `census` properties.
                The metric is calculated with and without perfect information, for comparison.
        
            :param before: A table of people and their characteristics before the change.
                (generally obtained from Population.people)
            
            :param after: A table of people and their characteristics after the change.
                (generally obtained from Transformation.people)
            
            :param metric_method: A string specifying which metric to compute.
            
                - "median_person" : Returns the total travel of median person.
                - "average_person" : Returns the total travel of average (mean) person.
                - "skews_grocery" : Returns the median observed travel using a
                    high proportion of grocery travel and low propertion of work and social travel.
                - "skews_work" : Returns the median observed travel using a
                    high proportion of work travel and low propertion of grocery and social travel.
                - "skews_backlash" : Sets a random proportion (based on p = 0.3) of smartphones to disappear due to people 
                    attempt to avoid being tracked (only affects the after metrics)
                    
        """

        # Get tables from objects:
        before_people = before.people.copy()
        after_people = after.people.copy()

        # Check inputs:
        assert len(before_people) == len(after_people)
        
        valid_methods = ['median_person','average_person','skews_grocery', 'skews_work', 'backlash']
        assert metric_method in valid_methods, "{} is not a valid method: {} .".format(
            metric_method,", ".join(valid_methods)
        )
        
        # Set random state:
        np.random.seed(random_state)
        
        # Store inputs:
        self._random_state = random_state
        self._metric_method = metric_method
        self._before = before
        self._after = after
        
        # Get/set helper values:
        N = len(before_people)
        
        if metric_method=="median_person":
            
            def _actual(df):
                df['measure'] = df[['worktravel','socialtravel','grocerytravel']].sum(axis=1)
                return df
            def _observed(df):
                df['measure'] = df[['worktravel','socialtravel','grocerytravel']].sum(axis=1)
                df['measure'] = np.where(df['phoneownership']==1,df['measure'],np.nan)
                return df
            def _measure(vals):
                return np.nanmedian(vals)
            
        elif metric_method=="average_person":
            
            def _actual(df):
                df['measure'] = df[['worktravel','socialtravel','grocerytravel']].sum(axis=1)
                return df
            def _observed(df):
                df['measure'] = df[['worktravel','socialtravel','grocerytravel']].sum(axis=1)
                df['measure'] = np.where(df['phoneownership']==1,df['measure'],np.nan)
                return df
            def _measure(vals):
                return np.nanmean(vals)
            
        elif metric_method=="skews_grocery":
            
            captured_pct_grocery = 0.9
            captured_pct_other = 0.6
            
            def _actual(df):
                df['measure'] = df[['worktravel','socialtravel','grocerytravel']].sum(axis=1)
                return df
            def _observed(df):
                df['measure'] = 0
                df['measure'] += df['worktravel']*captured_pct_other
                df['measure'] += df['socialtravel']*captured_pct_other
                df['measure'] += df['grocerytravel']*captured_pct_grocery
                df['measure'] = np.where(df['phoneownership']==1,df['measure'],np.nan)
                return df
            def _measure(vals):
                return np.nanmean(vals)
            
        elif metric_method=="skews_work":
            
            captured_pct_work = 0.8
            captured_pct_other = 0.3
            
            def _actual(df):
                df['measure'] = df[['worktravel','socialtravel','grocerytravel']].sum(axis=1)
                return df
            def _observed(df):
                df['measure'] = 0
                df['measure'] += df['worktravel']*captured_pct_work
                df['measure'] += df['socialtravel']*captured_pct_other
                df['measure'] += df['grocerytravel']*captured_pct_other
                df['measure'] = np.where(df['phoneownership']==1,df['measure'],np.nan)
                return df
            def _measure(vals):
                return np.nanmean(vals)
            
        elif metric_method=="backlash":
            
            def _actual(df):
                df['measure'] = df[['worktravel','socialtravel','grocerytravel']].sum(axis=1)
                return df
            def _observed(df, rate):
                df['measure'] = df[['worktravel','socialtravel','grocerytravel']].sum(axis=1)
                df['measure'] = np.where(df['phoneownership']==1,df['measure'],np.nan)
                df['backlash'] = np.random.binomial(1,rate,len(df))
                df['measure'] = np.where(df['backlash']==0,df['measure'],np.nan)
                return df
            def _measure(vals):
                return np.nanmean(vals)
            
            
        # Compute results:
        group_cols = ['population_name','location_name']
        results = pd.concat([before_people,after_people],axis=0,sort=False)[group_cols].drop_duplicates()
        results = results.set_index(group_cols).sort_index()
        results['actual_before'] = _actual(before_people).groupby(group_cols,sort=True)['measure'].apply(_measure)
        results['actual_after'] = _actual(after_people).groupby(group_cols,sort=True)['measure'].apply(_measure)
        results['actual_delta'] = results['actual_after'] - results['actual_before']
        results['actual_change'] = results['actual_delta']/results['actual_before']
        results['observed_before'] = _observed(before_people, 0).groupby(group_cols,sort=True)['measure'].apply(_measure)
        results['observed_after'] = _observed(after_people, 0.35).groupby(group_cols,sort=True)['measure'].apply(_measure)
        results['observed_delta'] = results['observed_after'] - results['observed_before']
        results['observed_change'] = results['observed_delta']/results['observed_before']
            
        # Store results:
        self._results = results
        
        # Unset random state:
        np.random.seed(None)
        
    @property
    def metric_method(self):
        """Results the name of the method applied for the metric calculations."""
        return self._metric_method
        
    @property
    def results(self):
        """Results (before_people,after_people,delta) for both ground truth and observed measure."""
        return self._results
    
    def __str__(self):
        return "Metric : {}".format(self._metric_method)
        
class Summary:
    
    def __init__(self,metric,summary_method='full_results',random_state=None):
        
        """
            
            An object that represents a summarize version of the results from a metric.
                The `metric` objects must have a `results` property.
                The result columns are preserved.
        
            :param metric: A Metric object with calculated results (before/after, actual/observed).
                (generally obtained from Population.people)
            
            :param summary_method: A string specifying which summary to use.
            
                - "full_results" : Returns the metric results with no summarization.
                - "best5_values" : Returns the values (and sets others to null) for the
                    5 locations with the largest reduction in travel (by percent).
                - "best5_status" : Returns an indicator of whether or not the location appears in the
                    5 locations with the largest reduction in travel (by percent).
                - "worst5_values" : Returns the values (and sets others to null) for the
                    5 locations with the smallest reduction in travel (by percent).
                - "worst5_status" : Returns an indicator of whether or not the location appears in the
                    5 locations with the smallest reduction in travel (by percent).
                    
        """
        
        valid_methods = [
            'full_results',
            'best5_values', 'best5_status',
            'worst5_values', 'worst5_status',
        ]
        assert summary_method in valid_methods, "{} is not a valid method: {} .".format(
            summary_method,", ".join(valid_methods)
        )
        
        # Set random state:
        np.random.seed(random_state)
        
        # Store inputs:
        self._random_state = random_state
        self._summary_method = summary_method
        self._metric = metric
        
        # Get metric results:
        results = metric.results.copy()

        # Helper functions:

        def best_n_values(result,n):
            assert (n>=0)
            n = min(n,len(results))
            # Best: Largest reduction in tavel (proportional) --> Most negative value.
            summary = results
            actual_filter = np.argsort(metric.results['actual_change'])<n
            summary['actual_before'] = np.where(actual_filter,summary['actual_before'],np.nan)
            summary['actual_after'] = np.where(actual_filter,summary['actual_after'],np.nan)
            summary['actual_delta'] = np.where(actual_filter,summary['actual_delta'],np.nan)
            summary['actual_change'] = np.where(actual_filter,summary['actual_change'],np.nan)
            observed_filter = np.argsort(metric.results['observed_change'])<n
            summary['observed_before'] = np.where(observed_filter,summary['observed_before'],np.nan)
            summary['observed_after'] = np.where(observed_filter,summary['observed_after'],np.nan)
            summary['observed_delta'] = np.where(observed_filter,summary['observed_delta'],np.nan)
            summary['observed_change'] = np.where(observed_filter,summary['observed_change'],np.nan)
            return summary

        def best_n_status(result,n):
            assert (n>=0)
            n = min(n,len(results))
            # Best: Largest reduction in tavel (proportional) --> Most negative value.
            summary = results
            actual_filter = np.argsort(metric.results['actual_change'])<n
            summary['actual_before'] = np.where(actual_filter,True,False)
            summary['actual_after'] = np.where(actual_filter,True,False)
            summary['actual_delta'] = np.where(actual_filter,True,False)
            summary['actual_change'] = np.where(actual_filter,True,False)
            observed_filter = np.argsort(metric.results['observed_change'])<n
            summary['observed_before'] = np.where(observed_filter,True,False)
            summary['observed_after'] = np.where(observed_filter,True,False)
            summary['observed_delta'] = np.where(observed_filter,True,False)
            summary['observed_change'] = np.where(observed_filter,True,False)
            return summary

        def worst_n_values(result,n):
            assert (n>=0)
            n = min(n,len(results))
            # Worst 5: Largest reduction in tavel (proportional) --> Least negative value.
            summary = results
            actual_filter = np.argsort(-metric.results['actual_change'])<5
            summary['actual_before'] = np.where(actual_filter,summary['actual_before'],np.nan)
            summary['actual_after'] = np.where(actual_filter,summary['actual_after'],np.nan)
            summary['actual_delta'] = np.where(actual_filter,summary['actual_delta'],np.nan)
            summary['actual_change'] = np.where(actual_filter,summary['actual_change'],np.nan)
            observed_filter = np.argsort(-metric.results['observed_change'])<5
            summary['observed_before'] = np.where(observed_filter,summary['observed_before'],np.nan)
            summary['observed_after'] = np.where(observed_filter,summary['observed_after'],np.nan)
            summary['observed_delta'] = np.where(observed_filter,summary['observed_delta'],np.nan)
            summary['observed_change'] = np.where(observed_filter,summary['observed_change'],np.nan)
            return summary

        def worst_n_status(result,n):
            assert (n>=0)
            n = min(n,len(results))
            # Worst 5: Largest reduction in tavel (proportional) --> Least negative value.
            summary = results
            actual_filter = np.argsort(-metric.results['actual_change'])<5
            summary['actual_before'] = np.where(actual_filter,True,False)
            summary['actual_after'] = np.where(actual_filter,True,False)
            summary['actual_delta'] = np.where(actual_filter,True,False)
            summary['actual_change'] = np.where(actual_filter,True,False)
            observed_filter = np.argsort(-metric.results['observed_change'])<5
            summary['observed_before'] = np.where(observed_filter,True,False)
            summary['observed_after'] = np.where(observed_filter,True,False)
            summary['observed_delta'] = np.where(observed_filter,True,False)
            summary['observed_change'] = np.where(observed_filter,True,False)
            return summary
        
        if summary_method=="full_results":
            summary = results

        elif summary_method=="best5_values":
            summary = best_n_values(results,5)
        elif summary_method=="best5_status":
            summary = best_n_status(results,5)
        elif summary_method=="worst5_values":
            summary = worst_n_values(results,5)
        elif summary_method=="worst5_status":
            summary = worst_n_status(results,5)
            
        # Store results:
        self._summary = summary
        
        # Unset random state:
        np.random.seed(None)
        
    @property
    def summary_method(self):
        """Results the name of the summary applied to the calculated metrics."""
        return self._summary_method
        
    @property
    def summary(self,style='list'):
        """List of `results` tables."""
        return self._summary
        
    @property
    def results(self,style='list'):
        """List of `results` tables."""
        return self._summary
    
    def __str__(self):
        return "Summary : {}".format(self._summary_method)

class Experiment:
    
    EXPERIMENTS = 0
    
    def __init__(self,
        location_attributes,location_profiles,
        behavior='normal',
        metric_method='median_person',
        summary_method='full_results',
        trials=1,show_progress=True,experiment_name=None,random_state=None,
    ):
        
        """
            An object to conduct experiments where each trial generates a new population
            for the specified attributes, simulates the specified behavior, 
            and computes the the specified metric in order to compare the results.
            
            :param location_attributes: (pd.DataFrame) Attribute table used for the Population.
            :param location_profiles:   (pd.DataFrame) Attribute table used for the Population.
            :param behavior:            (string) Behavior used for the Transformation.
            :param metric_method:       (string) Method used for the Metric (how to aggregate results in each location).
            :param summary_method:      (string) Method used for the Summary (how to summarize results across locations).
            :param trials:              (int) Number of trials for the Experiment.
            :param experiment_name:     (string or None) Label for the Experiment.
            :param random_state:        (int or None) Random state used for the Experiment.
            
        """
        
        assert trials>0, "Must have at least 1 trial."
        assert int(trials)==trials, "Number of trials must be an integer."
        
        # Set random state:
        np.random.seed(random_state)
        
        # Increment counter and store experiment label
        Experiment.EXPERIMENTS += 1
        if experiment_name is None:
            experiment_name = "experiment{}".format(Experiment.EXPERIMENTS)
        self._experiment_name = experiment_name
        
        # Store input parameters:
        self._random_state = random_state
        self._location_attributes = location_attributes.copy()
        self._location_profiles = location_profiles.copy()
        self._behavior = behavior
        self._metric_method = metric_method
        self._summary_method = summary_method
        self._trials = trials
        
        # Store results:
        self._before = []
        self._after = []
        self._results = []
        self._summaries = []
        
        # Conduct trials:
        if show_progress: progress_bar = tqdm(total=trials)
        random_states = np.random.randint(10**7,size=(trials,3))
        for trial in range(trials):
            population_name = "trial{}".format(trial+1)
            # Get new random seeds:
            rs1, rs2, rs3 = random_states[trial,:]
            # Perform experiment for this trial:
            if show_progress: progress_bar.set_description('population')
            population = Population(location_attributes,location_profiles,population_name=population_name,random_state=rs1)
            if show_progress: progress_bar.set_description('transformation')
            transformation = Transformation(population,behavior=behavior,random_state=rs2)
            if show_progress: progress_bar.set_description('metric')
            before = population
            after = transformation
            metric = Metric(before=population,after=transformation,metric_method=metric_method,random_state=rs3)
            summary = Summary(metric=metric,summary_method=summary_method)
            # Append results:
            if show_progress: progress_bar.set_description('')
            self._before.append(before.people)
            self._after.append(after.people)
            self._results.append(metric.results)
            self._summaries.append(summary.summary)
            if show_progress: progress_bar.update()
        if show_progress: progress_bar.close()
        
        # Unset random state:
        np.random.seed(None)
    
    @property
    def experiment_name(self):
        """Label for the experiment."""
        return self._experiment_name
    
    @property
    def location_attributes(self):
        """Location attributes (Population)."""
        return self._location_attributes
    
    @property
    def location_profiles(self):
        """Location profiles (Population)."""
        return self._location_profiles
    
    @property
    def behavior(self):
        """Behavior (Transformation)."""
        return self._behavior
    
    @property
    def metric_method(self):
        """Method method (Metric)."""
        return self._metric_method
    
    @property
    def summary_method(self):
        """Summary method (Summary)."""
        return self._summary_method
    
    @property
    def trials(self):
        """Number of trials."""
        return self._trials
    
    @property
    def after(self):
        """List of `after` tables."""
        return self._after
    
    @property
    def before(self):
        """List of `before` tables."""
        return self._before
        
    @property
    def results(self,style='list'):
        """List of `results` tables."""
        return self._results
        
    @property
    def summaries(self,style='list'):
        """List of `summary` tables."""
        return self._summaries
        
    def get_results(self,style='list'):
        """
            List of `results` tables, in one of the following styles:
                - "list" (default) : python list of pd.DataFrames.
                - "df" : pd.DataFrames with the concatenation of all results.
                - "array" : numpy.array (3-dimensional).
                - "mean" : pd.DataFrames where each entry is the mean across all trials.
                - "std" : pd.DataFrames where each entry is the standard deviation across all trials.
        """
        valid_styles = ['list','df','array','mean','std']
        assert style in valid_styles, "{} is not a valid style: {}".format(style,", ".join(valid_styles))
        if style=='list':
            return self._results
        if style=='df':
            results = pd.concat(self._results,axis=0,sort=False).reset_index()
            results['experiment_name'] = self._experiment_name
            results = results.set_index(['experiment_name','population_name','location_name'])
            return results
        elif style=='mean':
            results = self.get_results(style='df').reset_index()
            results = results.groupby(['experiment_name','location_name']).aggregate(np.nanmean).sort_index()
            results.columns.name = 'mean across {} trial(s)'.format(self._trials)
            return results
        elif style=='std':
            results = self.get_results(style='df').reset_index()
            results = results.groupby(['experiment_name','location_name']).aggregate(np.nanstd).sort_index()
            results.columns.name = 'std across {} trial(s)'.format(self._trials)
            return results
        elif style=='array':
            return np.array([results.to_numpy() for results in self._results])
        
    def get_summaries(self,style='list'):
        """
            List of `summary` tables, in one of the following styles:
                - "list" (default) : python list of pd.DataFrames.
                - "df" : pd.DataFrames with the concatenation of all summary tables.
                - "array" : numpy.array (3-dimensional).
                - "mean" : pd.DataFrames where each entry is the mean across all trials.
                - "std" : pd.DataFrames where each entry is the standard deviation across all trials.
        """
        valid_styles = ['list','df','array','mean','std']
        assert style in valid_styles, "{} is not a valid style: {}".format(style,", ".join(valid_styles))
        if style=='list':
            return self._summaries
        if style=='df':
            summaries = pd.concat(self._summaries,axis=0,sort=False).reset_index()
            summaries['experiment_name'] = self._experiment_name
            summaries = summaries.set_index(['experiment_name','population_name','location_name'])
            return summaries
        elif style=='mean':
            summaries = self.get_summaries(style='df').reset_index()
            summaries = summaries.groupby(['experiment_name','location_name']).aggregate(np.nanmean).sort_index()
            summaries.columns.name = 'mean across {} trial(s)'.format(self._trials)
            return summaries
        elif style=='std':
            summaries = self.get_summaries(style='df').reset_index()
            summaries = summaries.groupby(['experiment_name','location_name']).aggregate(np.nanstd).sort_index()
            summaries.columns.name = 'std across {} trial(s)'.format(self._trials)
            return summaries
        elif style=='array':
            return np.array([summaries.to_numpy() for summaries in self._summaries])
    
    def __str__(self):
        return "Experiment : {}".format(self._experiment_name)
    
    