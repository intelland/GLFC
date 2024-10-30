import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def plot_boxplots_from_file(root_dir, experiments_list, file_name, metrics_list, output_dir=None):
    """
    Plot the boxplots of the metrics of multiple experiments and print statistical information.
    
    Parameters:
        root_dir (str): Root directory of all the experiments.
        experiments_list (list): A list of names of the experiments.
        file_name (str): Name of the metrics file (.csv or .xlsx).
        metrics_list (list): A list of metrics to be plotted.
        output_dir (str): Directory to save the output plots. Default is current directory.
    """
    matplotlib.use('Agg')
    if output_dir is None:
        output_dir = os.getcwd()
    
    # Dictionary to hold all metrics data
    all_data = {metric: [] for metric in metrics_list}
    
    # Read data from all experiments
    for experiment in experiments_list:
        file_path = os.path.join(root_dir, experiment, 'test_latest', file_name)
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Please provide a .csv or .xlsx file.")
        
        # Remove the last two rows
        df = df[:-2]
        
        # Append relevant data for each metric
        for metric in metrics_list:
            temp_data = df.iloc[1::2][metric].tolist()[:-2] # Select odd rows
            relevant_data = temp_data[:-4] + temp_data[-3:-2] + temp_data[-1:]
            all_data[metric].append(relevant_data)
    
    # Plot boxplots for each metric
    for metric, data in all_data.items():
        # Combine data for boxplot
        combined_data = [item for sublist in data for item in sublist]

        # Check if data is empty
        if not combined_data:
            print(f"No data available for {metric}. Skipping.")
            continue

        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=experiments_list)
        plt.title(f'Boxplot of {metric}')
        plt.xlabel('Experiments')
        plt.ylabel(metric)

        # Save the plot
        output_path = os.path.join(output_dir, f'boxplot_{metric}.png')
        plt.savefig(output_path)
        plt.close()
        
        print(f"Boxplot for {metric} saved to {output_path}")
        
        # Calculate and print statistical information
        for i, experiment_data in enumerate(data):
            if len(experiment_data) == 0:
                print(f"No data for {metric} in {experiments_list[i]}. Skipping statistics.")
                continue
            mean = pd.Series(experiment_data).mean()
            std_dev = pd.Series(experiment_data).std()
            min_val = pd.Series(experiment_data).min()
            max_val = pd.Series(experiment_data).max()
            print(f"Statistics for {metric} in {experiments_list[i]}:")
            print(f"  Mean: {mean}")
            print(f"  Std Dev: {std_dev}")
            print(f"  Min: {min_val}")
            print(f"  Max: {max_val}")
        
        # Calculate and print p-values for each pair of experiments
        for i in range(len(experiments_list)):
            for j in range(i + 1, len(experiments_list)):
                if len(data[i]) == 0 or len(data[j]) == 0:
                    print(f"Skipping p-value calculation for {metric} between {experiments_list[i]} and {experiments_list[j]} due to insufficient data.")
                    continue
                t_stat, p_value = ttest_ind(data[i], data[j], equal_var=False)
                print(f"p-value for {metric} between {experiments_list[i]} and {experiments_list[j]}: {p_value}")

# Example usage
# plot_boxplots_from_file('/path/to/root_dir', ['exp1', 'exp2'],
