#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
from datetime import time
df = pd.read_csv('dataset-3.csv')
print(df)


# In[38]:


# Question 1: Distance Matrix Calculation
def calculate_distance_matrix(df):
    G = nx.DiGraph()

    for _, row in df.iterrows():
        G.add_edge(row['id_start'], row['id_end'], weight=row['distance'])
        G.add_edge(row['id_end'], row['id_start'], weight=row['distance'])

    all_pairs_shortest_paths = dict(nx.all_pairs_dijkstra_path_length(G))

    nodes = sorted(G.nodes)
    distance_matrix = pd.DataFrame(index=nodes, columns=nodes)

    for start_node, distances in all_pairs_shortest_paths.items():
        for end_node, distance in distances.items():
            distance_matrix.loc[start_node, end_node] = distance

    distance_matrix = distance_matrix.fillna(0)
    distance_matrix = 0.5 * (distance_matrix + distance_matrix.T)

    return distance_matrix

# Sample result dataframe
distance_matrix_result = calculate_distance_matrix(df)
print(distance_matrix_result)


# In[39]:


# Question 2: Unroll Distance Matrix


def unroll_distance_matrix(distance_matrix):
    data = []

    for start_id in distance_matrix.index:
        for end_id in distance_matrix.columns:
            if start_id != end_id:
                data.append({
                    'id_start': start_id,
                    'id_end': end_id,
                    'distance': distance_matrix.loc[start_id, end_id]
                })

    distance_df = pd.concat([pd.DataFrame([row]) for row in data], ignore_index=True)

    return distance_df

# Sample result dataframe
unrolled_distance_matrix_result = unroll_distance_matrix(distance_matrix_result)
print(unrolled_distance_matrix_result)



# In[40]:


#Question 3: Finding IDs within Percentage Threshold
def find_ids_within_ten_percentage_threshold(distance_df, reference_value):
    avg_distance = distance_df[distance_df['id_start'] == reference_value]['distance'].mean()
    threshold_lower = avg_distance * 0.9
    threshold_upper = avg_distance * 1.1

    selected_ids = distance_df[(distance_df['distance'] >= threshold_lower) & (distance_df['distance'] <= threshold_upper)]['id_start'].unique()
    sorted_selected_ids = np.sort(selected_ids)
    
    return sorted_selected_ids

reference_value = 1001400
selected_ids_result = find_ids_within_ten_percentage_threshold(unrolled_distance_matrix_result, reference_value)
print(selected_ids_result)


# In[41]:


# Question 4: Calculate Toll Rate
def calculate_toll_rate(distance_df):
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle_type in rate_coefficients.keys():
        distance_df[vehicle_type] = distance_df['distance'] * rate_coefficients[vehicle_type]

    return distance_df

# Sample result dataframe
toll_rate_result = calculate_toll_rate(unrolled_distance_matrix_result)
print(toll_rate_result)
# Question 5: Calculate Time-Based Toll Rates
def calculate_time_based_toll_rates(distance_df):
    def apply_discount(row):
        weekday_ranges = [(time(0, 0, 0), time(10, 0, 0)),
                          (time(10, 0, 0), time(18, 0, 0)),
                          (time(18, 0, 0), time(23, 59, 59))]

        if row['start_day'] in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            for start_time, end_time in weekday_ranges:
                if start_time <= row['start_time'] <= end_time:
                    return row['distance'] * 0.8 if (start_time, end_time) == (time(0, 0, 0), time(10, 0, 0)) else row['distance'] * 1.2
        else:
            return row['distance'] * 0.7

    distance_df['start_day'] = distance_df['start_day'].str.capitalize()
    distance_df['end_day'] = distance_df['start_day'].str.capitalize()

    distance_df['start_time'] = pd.to_datetime(distance_df['start_time'], format='%H:%M:%S').dt.time
    distance_df['end_time'] = pd.to_datetime(distance_df['end_time'], format='%H:%M:%S').dt.time

    distance_df['distance'] = distance_df.apply(apply_discount, axis=1)

    return 


# In[50]:




def calculate_time_based_toll_rates(distance_df):
    # Check if 'start_timestamp' column exists in the DataFrame
    if 'start_timestamp' not in distance_df.columns:
        raise ValueError("Column 'start_timestamp' not found in the DataFrame.")

    # Extract date, time, and day from the existing timestamp
    distance_df['start_date'] = pd.to_datetime(distance_df['start_timestamp']).dt.date
    distance_df['start_time'] = pd.to_datetime(distance_df['start_timestamp']).dt.time
    distance_df['start_day'] = pd.to_datetime(distance_df['start_date']).dt.day_name()

    # Create a mapping for time intervals and discount factors
    time_interval_mapping = {
        (datetime.time(0, 0, 0), datetime.time(10, 0, 0)): 0.8,
        (datetime.time(10, 0, 0), datetime.time(18, 0, 0)): 1.2,
        (datetime.time(18, 0, 0), datetime.time(23, 59, 59)): 0.8,
    }

    # Apply discount factors based on time intervals
    for interval, discount_factor in time_interval_mapping.items():
        mask = (distance_df['start_time'] >= interval[0]) & (distance_df['start_time'] <= interval[1])
        distance_df.loc[mask, 'vehicle_type'] *= discount_factor

    # Apply constant discount factor for weekends
    weekend_mask = (distance_df['start_day'].isin(['Saturday', 'Sunday']))
    distance_df.loc[weekend_mask, 'vehicle_type'] *= 0.7

    # Create end_day and end_time columns
    distance_df['end_day'] = distance_df['start_day']
    distance_df['end_time'] = (pd.to_datetime(distance_df['start_timestamp']) + 
                               pd.to_timedelta(distance_df['duration'], unit='s')).dt.time

    # Drop intermediate columns used for calculation
    distance_df = distance_df.drop(['start_date', 'start_timestamp'], axis=1)

    return distance_df

# Assuming 'selected_ids_result' is a NumPy array with a single column
# You may need to adjust this based on your actual data structure
selected_ids_result_df = pd.DataFrame(selected_ids_result, columns=['single_column_name'])

# Convert the column to string type
selected_ids_result_df['single_column_name'] = selected_ids_result_df['single_column_name'].astype(str)

# Split the single column into multiple columns based on the assumed tab separation
selected_ids_result_df[['start_timestamp', 'duration', 'vehicle_type', 'id_start', 'id_end']] = selected_ids_result_df['single_column_name'].str.split('\t', expand=True)

# Now, apply the function to the DataFrame
result_df = calculate_time_based_toll_rates(selected_ids_result_df)
print(result_df.head())


# In[ ]:





# In[ ]:




