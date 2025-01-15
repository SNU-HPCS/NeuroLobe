import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

import sys
import os


# Load result from pickle file
result_file = 'profile_result.pkl'

if len(sys.argv) > 1:
    result_path = './runspace/' + sys.argv[1] + '/simulator/'
    # print(result_path)
else:
    print("using default runspace './runspace/debug/simulator/'")
    result_path = './runspace/debug/simulator/'

profile_result_pandas = pd.read_pickle(result_path + result_file)

# print(profile_result_pandas)

profile_result = profile_result_pandas.drop(columns=['dt', 'latency', 'cyc_period', 'num_tasks', 'task_list', 'event_list'])
cyc_period = int(profile_result_pandas['cyc_period'].iloc[0])
num_tasks = int(profile_result_pandas['num_tasks'].iloc[0])
dt_list = profile_result_pandas['dt'].iloc[:num_tasks].to_list()
latency_list = profile_result_pandas['latency'].iloc[:num_tasks].to_list()
task_list = profile_result_pandas['task_list'].iloc[:num_tasks].to_list()
event_list = profile_result_pandas['event_list'].iloc[:num_tasks].to_list()

# graph design configuration
line_width = 1
line_alpha = 0.8 # line transparency
marker_size = 3

# for debugging
def print_result(df = profile_result):
    print(df)
    print(df.columns) # timestep
    print(df.index) # task / event / core / profile
    print(cyc_period, num_tasks)
    print(dt_list, latency_list)

# plot 1
def plot_single_task(df = profile_result, task=0, events=['total'], show_sum=False, cores=None):
    if events == 'all':
        events = event_list[task]
        events.add('total')
        events.add('sum')
        # print(events)

    # Add the cycle sum of all events to df data
    if 'sum' in events:
        # Filter out rows with 'total', 'violation' before calculating the sums
        filtered_profile_result = df[(df.index.get_level_values('event') != 'total')\
                                                & (df.index.get_level_values('event') != 'violation')]

        # Calculate the sum of all 'event' values for each 'task', 'core', and 'profile' combination
        sums = filtered_profile_result.groupby(['task', 'core', 'profile']).sum()
        sums.columns = pd.Index(filtered_profile_result.columns.tolist(), name='event')

        # Create a new MultiIndex level for 'sum' for each 'task', 'core', and 'profile'
        task_levels = filtered_profile_result.index.get_level_values('task').unique()
        core_levels = filtered_profile_result.index.get_level_values('core').unique()
        profile_levels = filtered_profile_result.index.get_level_values('profile').unique()

        sum_levels = pd.MultiIndex.from_product([task_levels, core_levels, profile_levels, ['sum']], names=['task', 'core', 'profile', 'event'])

        # Create a new DataFrame with the 'sum' values
        sum_values = pd.DataFrame(index=sum_levels, columns=filtered_profile_result.columns, data=None)

        # Fill in the 'sum' values with the correct sums for each 'task', 'core', and 'profile'
        for task_id in task_levels:
            for core in core_levels:
                for profile in profile_levels:
                    sums_row = sums.xs((task_id, core, profile))
                    sum_values.loc[(task_id, core, profile, 'sum')] = sums_row.tolist()

        # Reorder the levels to have 'event' as the second level
        sum_values = sum_values.reorder_levels(['task', 'event', 'core', 'profile'])

        # Concatenate the original DataFrame and the DataFrame with 'sum' values
        updated_profile_result = pd.concat([df, sum_values])

        # Sort the index for better readability
        updated_profile_result = updated_profile_result.sort_index()

        # Resulting DataFrame with the added 'sum' value in the 'event' level
        df = updated_profile_result.copy()

    # Plot separately for specified cores
    if cores:
        if type(cores) == int:
            cores = list(range(cores))
        assert(type(cores) == list)

        # Calculate the number of rows and columns for subplots
        num_cores = len(cores)
        num_cols = int(np.sqrt(num_cores))
        num_rows = int(np.ceil(num_cores / num_cols))

        # Create a figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))

        # Flatten the axes array to make it easier to work with
        axes = axes.flatten()

        if type(cores) == int:
            cores = list(range(cores))
        assert(type(cores) == list)

        for core, ax in zip(cores, axes):
            for event in events:
                # Extract the target dataset for the specified core
                cyc_data_core = df.loc[
                    (df.index.get_level_values('task') == task) &
                    (df.index.get_level_values('event') == event) &
                    (df.index.get_level_values('profile') == 'cyc') &
                    (df.index.get_level_values('core') == core)
                ] * cyc_period

                cyc_data_core = cyc_data_core.mean(axis=0) # does not change any value, making into a consistent type

                # Plot the data for each core with labels
                label_core = f'Event: {event}, Core: {core}'
                ax.plot(
                    cyc_data_core.index.tolist(),
                    cyc_data_core,
                    label=label_core,
                    linestyle='--',  # Use a dashed line for core data
                    marker='o',
                    markersize=marker_size, 
                    linewidth=line_width,
                    alpha=line_alpha
                )
                ax.set_title(f'Core {core}')  # Set the title for each subplot
                ax.legend()  # Add a legend for each subplot

        # Adjust spacing between subplots
        plt.tight_layout()

    else:
        # Create a plot
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        
        if show_sum:
            average_cyc_data = pd.DataFrame()
            max_cyc_data = pd.DataFrame()
            # Loop through each event combination
            for i, event in enumerate(events):
                # total has value only for the external core
                if event == 'total':
                    # Extract the target dataset for the specified core
                    cyc_data = df.loc[
                        (df.index.get_level_values('task') == task) &
                        (df.index.get_level_values('event') == event) &
                        (df.index.get_level_values('profile') == 'cyc') &
                        (df.index.get_level_values('core') == df.index.get_level_values('core').max())
                    ] * cyc_period

                    cyc_data = cyc_data.mean(axis=0) # does not change any value, making into a consistent type

                    label_max = f'Event: {event}'
                    plt.plot(
                        cyc_data.index.tolist(),
                        cyc_data,
                        label=label_max,
                        marker='o',  # Add a marker (dot) on the data points
                        markersize=marker_size,
                        linewidth=line_width,
                        alpha=line_alpha
                    )
                else:
                    # Extract the target dataset
                    cyc_data = df.loc[
                        (df.index.get_level_values('task') == task) &
                        (df.index.get_level_values('event') == event) &
                        (df.index.get_level_values('profile') == 'cyc')
                    ] * cyc_period
                    
                    # Calculate the average of cyc_data across rows
                    average_cyc_data[i] = cyc_data.mean(axis=0)

                    # Calculate the max of cyc_data across rows
                    max_cyc_data[i] = cyc_data.max(axis=0)
            
            average_cyc_data = average_cyc_data.sum(axis=1)
            max_cyc_data = max_cyc_data.sum(axis=1)
            print(average_cyc_data)
            print(max_cyc_data)

            # Plot the average and max data with labels
            label_average = f'Event: {events} (Average)'
            plt.plot(
                average_cyc_data.index.tolist(),
                average_cyc_data,
                label=label_average,
                marker='o',  # Add a marker on the data points
                markersize=marker_size,
                linewidth=line_width,
                alpha=line_alpha
            )
        
            label_max = f'Event: {events} (Max)'
            plt.plot(
                max_cyc_data.index.tolist(),
                max_cyc_data,
                label=label_max,
                marker='o',  # Add a marker (dot) on the data points
                markersize=marker_size,
                linewidth=line_width,
                alpha=line_alpha
            )

            # Show the legend
            plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.8))

            # Add a horizontal dashed line at the value 50
            # plt.axhline(y=dt_list[task], color='gray', linestyle='--', label='dt')

            plt.xlabel('Timestep')
            plt.ylabel('Time (ns)')
            plt.title(f'Task {task} ({task_list[task]}, dt = {dt_list[task]})')
            plt.grid(True)
        else:
            # Loop through each event combination
            for event in events:
                # total has value only for the external core
                if event == 'total':
                    # Extract the target dataset for the specified core
                    cyc_data = df.loc[
                        (df.index.get_level_values('task') == task) &
                        (df.index.get_level_values('event') == event) &
                        (df.index.get_level_values('profile') == 'cyc') &
                        (df.index.get_level_values('core') == df.index.get_level_values('core').max())
                    ] * cyc_period

                    cyc_data = cyc_data.mean(axis=0) # does not change any value, making into a consistent type

                    label_max = f'Event: {event}'
                    plt.plot(
                        cyc_data.index.tolist(),
                        cyc_data,
                        label=label_max,
                        marker='o',  # Add a marker (dot) on the data points
                        markersize=marker_size,
                        linewidth=line_width,
                        alpha=line_alpha
                    )
                else:
                    # Extract the target dataset
                    cyc_data = df.loc[
                        (df.index.get_level_values('task') == task) &
                        (df.index.get_level_values('event') == event) &
                        (df.index.get_level_values('profile') == 'cyc')
                    ] * cyc_period
                    
                    # Calculate the average of cyc_data across rows
                    average_cyc_data = cyc_data.mean(axis=0)

                    # Calculate the max of cyc_data across rows
                    max_cyc_data = cyc_data.max(axis=0)

                    # Plot the average and max data with labels
                    label_average = f'Event: {event} (Average)'
                    plt.plot(
                        average_cyc_data.index.tolist(),
                        average_cyc_data,
                        label=label_average,
                        marker='o',  # Add a marker on the data points
                        markersize=marker_size,
                        linewidth=line_width,
                        alpha=line_alpha
                    )
                
                    label_max = f'Event: {event} (Max)'
                    plt.plot(
                        max_cyc_data.index.tolist(),
                        max_cyc_data,
                        label=label_max,
                        marker='o',  # Add a marker (dot) on the data points
                        markersize=marker_size,
                        linewidth=line_width,
                        alpha=line_alpha
                    )

                # Show the legend
                plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.8))

                # Add a horizontal dashed line at the value 50
                # plt.axhline(y=dt_list[task], color='gray', linestyle='--', label='dt')

                plt.xlabel('Timestep')
                plt.ylabel('Time (ns)')
                plt.title(f'Task {task} ({task_list[task]}, dt = {dt_list[task]})')
                plt.grid(True)

    # Save the plot as a PNG file
    file_name = f'plot1_single_task_{task}_{task_list[task]}_{events}.png'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')

    # Show the plot
    # plt.show()

# plot 2
def plot_multi_tasks(df = profile_result, tasks='all', plot_type='max'):
    event = 'sum'

    # Create a plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    if tasks == 'all':
        tasks = list(range(num_tasks))

    # Find the GCD of all 'dt's
    dts = [dt_list[i] for i in tasks]
    gcd = np.gcd.reduce(np.array(dts))
    # print(gcd)

    # Add the cycle sum of all events to df data
    # Filter out rows with 'total', 'violation' before calculating the sums
    filtered_profile_result = df[(df.index.get_level_values('event') != 'total')\
                                            & (df.index.get_level_values('event') != 'violation')]

    # Calculate the sum of all 'event' values for each 'task', 'core', and 'profile' combination
    sums = filtered_profile_result.groupby(['task', 'core', 'profile']).sum()
    sums.columns = pd.Index(filtered_profile_result.columns.tolist(), name='event')

    # Create a new MultiIndex level for 'sum' for each 'task', 'core', and 'profile'
    task_levels = filtered_profile_result.index.get_level_values('task').unique()
    core_levels = filtered_profile_result.index.get_level_values('core').unique()
    profile_levels = filtered_profile_result.index.get_level_values('profile').unique()

    sum_levels = pd.MultiIndex.from_product([task_levels, core_levels, profile_levels, ['sum']], names=['task', 'core', 'profile', 'event'])

    # Create a new DataFrame with the 'sum' values
    sum_values = pd.DataFrame(index=sum_levels, columns=filtered_profile_result.columns, data=None)

    # Fill in the 'sum' values with the correct sums for each 'task', 'core', and 'profile'
    for task in task_levels:
        for core in core_levels:
            for profile in profile_levels:
                sums_row = sums.xs((task, core, profile))
                sum_values.loc[(task, core, profile, 'sum')] = sums_row.tolist()

    # Reorder the levels to have 'event' as the second level
    sum_values = sum_values.reorder_levels(['task', 'event', 'core', 'profile'])

    # Concatenate the original DataFrame and the DataFrame with 'sum' values
    updated_profile_result = pd.concat([df, sum_values])

    # Sort the index for better readability
    updated_profile_result = updated_profile_result.sort_index()

    # Resulting DataFrame with the added 'sum' value in the 'event' level
    df = updated_profile_result.copy()

    # Loop through each task
    for task in tasks:
        scale = dt_list[task] / gcd
        # Extract the target dataset
        cyc_data = df.loc[
            (df.index.get_level_values('task') == task) &
            (df.index.get_level_values('event') == event) &
            (df.index.get_level_values('profile') == 'cyc')
        ] * cyc_period / scale
        
        if plot_type == 'avg':
            # Calculate the average of cyc_data across rows
            cyc_data = cyc_data.mean(axis=0)
        elif plot_type == 'max':
            # Calculate the max of cyc_data across rows
            cyc_data = cyc_data.max(axis=0)
        else:
            assert(0) # plot_type = 'max' or 'avg'

        # Plot the data with labels
        label = f'Task {task} ({task_list[task]})'
        plt.plot(
            (cyc_data.index * scale).tolist(),
            cyc_data,
            label=label,
            marker='o',  # Add a marker on the data points
            markersize=marker_size,
            linewidth=line_width,
            alpha=line_alpha
        )

    # Add a horizontal dashed line at the value 50
    # plt.axhline(y=dt_list[task], color='gray', linestyle='--', label='dt')

    plt.xlabel(f'Timestep (dt = {gcd} ns)')
    plt.ylabel('Time (ns)')
    plt.title(f'Tasks {tasks} ({plot_type})')
    plt.grid(True)
    
    # Show the legend
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.8))

    # Save the plot as a PNG file
    file_name = f'plot2_multi_tasks_{tasks}_{[task_list[i] for i in tasks]}_{plot_type}.png'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')

    # Show the plot
    # plt.show()

# plot 3
def plot_check_violation(df = profile_result, task=0):
    event = 'violation'
    core = df.index.get_level_values('core').max() # external core

    # Create a plot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    # Extract the target dataset for the specified core
    cyc_data = df.loc[
        (df.index.get_level_values('task') == task) &
        (df.index.get_level_values('event') == event) &
        (df.index.get_level_values('profile') == 'cyc') &
        (df.index.get_level_values('core') == core)
    ] * cyc_period

    cyc_data = cyc_data.mean(axis=0) # does not change any value, making into a consistent type

    label_max = f'Event: {event}'
    plt.plot(
        cyc_data.index.tolist(),
        cyc_data,
        label=label_max,
        marker='o',  # Add a marker (dot) on the data points
        markersize=marker_size,
        linewidth=line_width,
        alpha=line_alpha
    )

    # Add a horizontal dashed line at the value 50
    # plt.axhline(y=latency_list[task], color='gray', linestyle='--', label='dt')

    plt.xlabel('Timestep')
    plt.ylabel('Time (ns)')
    plt.title(f'Task {task} ({task_list[task]}, latency = {latency_list[task]})')
    plt.grid(True)
    
    # Show the legend
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.8))

    # Save the plot as a PNG file
    file_name = f'plot3_violation_{task}_{task_list[task]}.png'
    plt.savefig(file_name, dpi=300, bbox_inches='tight')

    # Show the plot
    # plt.show()

# calculate under utilization
def under_util(df = profile_result, task=0, cores=1):
    # Add the cycle sum of all events to df data
    # Filter out rows with 'total', 'violation' before calculating the sums
    filtered_profile_result = df[(df.index.get_level_values('event') != 'total')\
                                            & (df.index.get_level_values('event') != 'violation')]

    # Calculate the sum of all 'event' values for each 'task', 'core', and 'profile' combination
    sums = filtered_profile_result.groupby(['task', 'core', 'profile']).sum()
    sums.columns = pd.Index(filtered_profile_result.columns.tolist(), name='event')

    # Create a new MultiIndex level for 'sum' for each 'task', 'core', and 'profile'
    task_levels = filtered_profile_result.index.get_level_values('task').unique()
    core_levels = filtered_profile_result.index.get_level_values('core').unique()
    profile_levels = filtered_profile_result.index.get_level_values('profile').unique()

    sum_levels = pd.MultiIndex.from_product([task_levels, core_levels, profile_levels, ['sum']], names=['task', 'core', 'profile', 'event'])

    # Create a new DataFrame with the 'sum' values
    sum_values = pd.DataFrame(index=sum_levels, columns=filtered_profile_result.columns, data=None)

    # Fill in the 'sum' values with the correct sums for each 'task', 'core', and 'profile'
    for task_id in task_levels:
        for core in core_levels:
            for profile in profile_levels:
                sums_row = sums.xs((task_id, core, profile))
                sum_values.loc[(task_id, core, profile, 'sum')] = sums_row.tolist()

    # Reorder the levels to have 'event' as the second level
    sum_values = sum_values.reorder_levels(['task', 'event', 'core', 'profile'])

    # Calculate the total cycle
    cyc_data = df.loc[
        (df.index.get_level_values('task') == task) &
        (df.index.get_level_values('event') == 'total') &
        (df.index.get_level_values('profile') == 'cyc')
    ]
    total_cycle = cyc_data.sum(axis=1).max()
    # total_cycle = cyc_data.max(axis=0).loc[:29].sum()
    # print(total_cycle)

    # Calculate separately for specified cores
    if type(cores) == int:
        cores = list(range(cores))
    assert(type(cores) == list)

    for core in cores:
        # Extract the target dataset for the specified core
        cyc_data_core = sum_values.loc[
            (sum_values.index.get_level_values('task') == task) &
            (sum_values.index.get_level_values('event') == 'sum') &
            (sum_values.index.get_level_values('profile') == 'cyc') &
            (sum_values.index.get_level_values('core') == core)
        ]
        total_cycles_core = cyc_data_core.sum(axis=1)
        # print(total_cycles_core)
        # total_cycles_core = cyc_data_core.iloc[:, :30].sum(axis=1)
        print("Core", core, ":", 100 - total_cycles_core.loc[task, 'sum', core, 'cyc'] / total_cycle * 100)

def save_csv(df = profile_result, path = None):

    # Add the cycle sum of all events to df data
    # Filter out rows with 'total', 'violation' before calculating the sums
    filtered_profile_result = df[(df.index.get_level_values('event') != 'total')\
                                            & (df.index.get_level_values('event') != 'violation')]

    # Calculate the sum of all 'event' values for each 'task', 'core', and 'profile' combination
    sums = filtered_profile_result.groupby(['task', 'core', 'profile']).sum()
    sums.columns = pd.Index(filtered_profile_result.columns.tolist(), name='event')

    # Create a new MultiIndex level for 'sum' for each 'task', 'core', and 'profile'
    task_levels = filtered_profile_result.index.get_level_values('task').unique()
    core_levels = filtered_profile_result.index.get_level_values('core').unique()
    profile_levels = filtered_profile_result.index.get_level_values('profile').unique()

    sum_levels = pd.MultiIndex.from_product([task_levels, core_levels, profile_levels, ['sum']], names=['task', 'core', 'profile', 'event'])

    # Create a new DataFrame with the 'sum' values
    sum_values = pd.DataFrame(index=sum_levels, columns=filtered_profile_result.columns, data=None)

    # Fill in the 'sum' values with the correct sums for each 'task', 'core', and 'profile'
    for task_id in task_levels:
        for core in core_levels:
            for profile in profile_levels:
                sums_row = sums.xs((task_id, core, profile))
                sum_values.loc[(task_id, core, profile, 'sum')] = sums_row.tolist()

    # Reorder the levels to have 'event' as the second level
    sum_values = sum_values.reorder_levels(['task', 'event', 'core', 'profile'])

    # Concatenate the original DataFrame and the DataFrame with 'sum' values
    updated_profile_result = pd.concat([df, sum_values])


    # Add the average cycle over all cores
    filtered_profile_result = updated_profile_result
    # Calculate the average of all 'core' values for each 'task', 'event', and 'profile' combination
    avgs = filtered_profile_result.groupby(['task', 'event', 'profile']).mean()
    avgs.columns = pd.Index(filtered_profile_result.columns.tolist(), name='core')

    # Create a new MultiIndex level for 'avg' for each 'task', 'core', and 'profile'
    task_levels = filtered_profile_result.index.get_level_values('task').unique()
    event_levels = filtered_profile_result.index.get_level_values('event').unique()
    profile_levels = filtered_profile_result.index.get_level_values('profile').unique()

    avg_levels = pd.MultiIndex.from_product([task_levels, event_levels, profile_levels, ['avg']], names=['task', 'event', 'profile', 'core'])

    # Create a new DataFrame with the 'avg' values
    avg_values = pd.DataFrame(index=avg_levels, columns=filtered_profile_result.columns, data=None)

    # Fill in the 'avg' values with the correct avgs for each 'task', 'event', and 'profile'
    for task_id in task_levels:
        for event in event_levels:
            for profile in profile_levels:
                avgs_row = avgs.xs((task_id, event, profile))
                avg_values.loc[(task_id, event, profile, 'avg')] = avgs_row.tolist()

    # Reorder the levels to have 'event' as the second level
    avg_values = avg_values.reorder_levels(['task', 'event', 'core', 'profile'])

    # Concatenate the original DataFrame and the DataFrame with 'avg' values
    updated_profile_result = pd.concat([df, avg_values])

    # Sort the index for better readability
    updated_profile_result = updated_profile_result.sort_index()

    # Resulting DataFrame with the added 'sum' value in the 'event' level and 'avg' value in the 'core' level
    df = updated_profile_result.copy()

    df.to_csv(path)

def calc_avg_latency(df=profile_result, task=0):
    cyc_data = df.loc[
        (df.index.get_level_values('task') == task) &
        (df.index.get_level_values('event') == 'total') &
        (df.index.get_level_values('profile') == 'cyc')
    ] * cyc_period
    cyc_data = cyc_data.max(axis=0)[:-1]
    return cyc_data.mean()

def count_violation(df = profile_result, task=0):
    # Extract the target dataset for the specified core
    cyc_data = df.loc[
        (df.index.get_level_values('task') == task) &
        (df.index.get_level_values('event') == 'violation') &
        (df.index.get_level_values('profile') == 'cyc')
    ] * cyc_period
    cyc_data = cyc_data.max(axis=0)[:-1] # does not change any value, making into a consistent type
    return len(cyc_data)-1, (cyc_data > latency_list[task]).sum(axis=0)


if __name__ == "__main__":
    # Save results into csv files
    # if len(sys.argv) > 1:
    #     mapping_folder_name = "./csv_results/"
    #     if not os.path.isdir(mapping_folder_name):
    #         os.mkdir(mapping_folder_name)
    #     file_name = mapping_folder_name + sys.argv[1] + ".csv"
    #     save_csv(path=file_name)

    # Calcualte under utilization
    # if len(sys.argv) > 2:    
    #     under_util(cores=int(sys.argv[2]))
    
    # Calculate average latency
    avg_latency = calc_avg_latency(task=0)

    # Count violation timesteps
    timesteps, violation_cnt =count_violation(task=0)

    print("avg latency: %d ns, violation: %d / %d ts (%.2f)" %(avg_latency, violation_cnt, timesteps, violation_cnt / timesteps))