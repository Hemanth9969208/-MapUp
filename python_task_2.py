#!/usr/bin/env python
# coding: utf-8

# # Python Task 1

# In[116]:


import pandas as pd
df = pd.read_csv('dataset-1.csv')
import numpy as np
print(df)


# In[123]:


def generate_car_matrix(df):

    # Write your logic here
    # Create a pivot table with id_1 as index, id_2 as columns, and car as values
    car_matrix = df.pivot_table(index='id_1', columns='id_2', values='car', fill_value=0)

    # Set diagonal values to 0
    np.fill_diagonal(car_matrix.values, 0)

    return car_matrix

result_matrix = generate_car_matrix(df)
print(result_matrix)


# In[124]:


import pandas as pd

def get_type_count(df):
 
    # Write your logic here
  
    # Add a new categorical column 'car_type' based on the values of the column 'car'
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Calculate the count of occurrences for each 'car_type' category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts


print(result)


# In[119]:


def get_bus_indexes(df):
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
  
    # Calculate the mean value of the 'bus' column
    bus_mean = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * bus_mean].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

print(result)


# In[125]:


def filter_routes(df):
 
    
    # Calculate the average value of the 'truck' column for each 'route'
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average 'truck' value is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of selected routes
    selected_routes.sort()

    return selected_routes
print(result)


# In[122]:


def multiply_matrix(df):
  
 
    # Apply the specified logic to modify each value in the DataFrame
    modified_df = df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round the values to 1 decimal place
    modified_df = modified_df.round(1)

    return modified_df

modified_result_df = multiply_matrix(result_matrix)

# Print the modified DataFrame
print(modified_result_df)


# In[ ]:





# In[ ]:




