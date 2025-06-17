import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import re


sns.set_theme(style="whitegrid")


def clean_data(data):
    """
    Cleans the data by removing unwanted characters and converting to float.

    Args:
        data (pd.Series): The data to clean.

    Returns:
        pd.Series: Cleaned data.
    """
    # Remove unwanted characters and convert to float
    data = data.astype(str).str.replace('%', '', regex=False)
    data = data.astype(str).str.replace(',', '.', regex=False)  # Replace comma with dot
    data = data.str.replace(' ', '', regex=False)  # Remove spaces
    data = data.str.replace('-', '', regex=False)  # Remove dashes
    data = data.str.replace('(', '', regex=False)  # Remove opening parentheses
    data = data.str.replace('0', '', regex=False)  # Remove leading zeros

    return pd.to_numeric(data, errors='coerce')    # Convert to float





def plot_scatter_trend(
    x_data,
    y_data,
    dot_color='blue',
    dot_size=50,
    show_regression_line=True,
    x_label="X",
    y_label="Y",
    title="X-Y Scatter Plot",
    fig_name="scatter_plot.png",
    show_fig=False
):
    """
    Generates a scatter plot with optional trendlines, similar to the provided image.

    Args:
        x_data (array-like): Data for the x-axis. Expected as values between 0 and 1 if representing percentages (0% to 100%).
        y_data (array-like): Data for the y-axis.
        dot_color (str, optional): Color of the scatter points. Defaults to 'blue'.
        dot_size (int, optional): Size of the scatter points (area in points^2). Defaults to 50.
        show_regression_line (bool, optional): Whether to calculate and plot a linear regression line based on the data. Defaults to True.
        x_label (str, optional): Label for the x-axis. Defaults to "GLC M".
        y_label (str, optional): Label for the y-axis. Defaults to "Prüfungsnote ZAP M".
        title (str, optional): Title of the plot. Defaults to "Grundlagencheck M x Prüfungsnote ZAP M".

    Returns:
        None: Displays the plot.
    """
    if len(x_data) != len(y_data) or len(x_data) == 0:
        print("Error: x_data and y_data must be non-empty and have the same length.")
        return

    x_data = clean_data(x_data)
    y_data = clean_data(y_data)

    # Ensure data are numpy arrays for easier handling
    x = np.array(x_data)
    y = np.array(y_data)
    for i in range(len(x)-1, -1, -1):
        if np.isnan(x[i]) or np.isnan(y[i]):
            x = np.delete(x, i)
            y = np.delete(y, i)

    # --- Plotting Setup ---
    fig, ax = plt.subplots(figsize=(8, 6)) # Use 'ax' for object-oriented interface

    # --- Scatter Plot ---
    ax.scatter(x, y, color=dot_color, s=dot_size, zorder=3, label='Data Points') # zorder=3 puts dots on top

    # --- Trendlines ---
    # Define x range for plotting lines (0% to 100%)
    line_x = np.array([0, x.max()])

    # 1. Regression Line (Calculated - Light Blue)
    if show_regression_line:
        # Calculate linear regression coefficients (slope and intercept)
        slope_regr, intercept_regr = np.polyfit(x, y, 1)
        # print(f"Regression line: y = {slope_regr:.2f}x + {intercept_regr:.2f}")
        # Calculate y-values for the regression line
        line_y_regr = slope_regr * line_x + intercept_regr
        # Plot the regression line
        ax.plot(line_x, line_y_regr, color='skyblue', linestyle='-', linewidth=2,
                label=f'Regression (y={slope_regr:.2f}x+{intercept_regr:.2f})', zorder=2) # zorder=2 puts line below dots

    # --- Statistics ---
    correlation_r, p_value = pearsonr(x, y)
    num_points = len(x)
    stats_text = f"Points (n): {num_points}\n" \
                 f"R-squared (R²): {correlation_r**2:.3f}\n" \
                 f"Reg. line: y = {slope_regr:.2f}x + {intercept_regr:.2f}"
    
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))


    # --- Formatting ---
    # Title and Labels
    ax.set_title(title, fontsize=16, pad=15) # Add padding to title
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)

    # Grid
    ax.grid(True, which='major', color='lightgray', linestyle='-', linewidth=0.7, zorder=0) # zorder=0 puts grid behind everything
    
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.savefig(fig_name, dpi=300, bbox_inches='tight') # Save the figure as a PNG file
    if show_fig:
        plt.show()



def plot_correlation_heatmap(df, column_names):
    """
    Cleans specified columns of a DataFrame, calculates their cross-correlation,
    and plots the result as a heatmap.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_names (list): A list of column names to include in the correlation plot.
    """
    print(f"Processing columns: {column_names}")

    if column_names is None or len(column_names) == 0:
        print("No columns provided for correlation heatmap.")
        return
    
    if column_names == 'all':
        column_names = df.columns.tolist()
        print(f"Using all columns: {column_names}")
    

    # Select and Copy Columns to avoid modifying the original DataFrame
    selected_df = df[column_names].copy()

    # Clean selected columns using the provided clean_data function
    cleaned_df = pd.DataFrame()
    for col in column_names:
        cleaned_df[col] = clean_data(selected_df[col])


    # Calculate Correlation Matrix
    correlation_matrix = cleaned_df.corr(method='pearson') # Or 'kendall', 'spearman'

    # Plot Heatmap
    print("Plotting heatmap...")
    plt.figure(figsize=(max(8, len(column_names)*0.8), max(6, len(column_names)*0.7))) # Adjust size dynamically
    sns.heatmap(correlation_matrix,
                annot=True,         # Show correlation values on the map
                cmap='coolwarm',    # Color scheme (good for correlations: red=positive, blue=negative)
                fmt=".2f",          # Format annotations to 2 decimal places
                linewidths=.5,      # Add lines between cells
                linecolor='lightgray',
                cbar=True)          # Show the color bar legend
    plt.title("Cross-Correlation Matrix", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=10) # Rotate x-axis labels if they overlap
    plt.yticks(rotation=0, fontsize=10)              # Ensure y-axis labels are readable
    plt.tight_layout()              # Adjust plot to prevent labels overlapping
    plt.show()




def calc_stats_table(df, df_names, sn, en):

    columns = [en] + sn
    df = df[columns].copy()
    df_legend = {}


    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].mask(df[col] < 1, np.nan, inplace=True)

    df_order = pd.DataFrame()
    for i, s in enumerate(sn):
        name1 = s + "_" + en
        name2 = en + "_" + f"{i+1}"
        df_order[s] = df[s]
        df_order[name1] = np.where(np.isnan(df[en]), np.nan, df[s])
        df_order[name2] = np.where(np.isnan(df[s]), np.nan, df[en])

        df_names[s] = f"PN{i+1} (Alle)"
        df_names[name1] = f"PN{i+1} (mit EN)"
        df_names[name2] = f"EN{i+1}"

        df_legend[s] = f"Prüfungsnote SimPr {en[-1]} (Alle SuS Serie {i+1})"
        df_legend[name1] = f"Prüfungsnote SimPr {en[-1]} (SuS mit Erfolgsnote Serie {i+1})"
        df_legend[name2] = f"Erfolgsnote ZAP (SuS mit Erfolgsnote Serie {i+1})"


    # Add an column for all series
    df_order["SPall"] = df[sn].mean(axis=1)
    df_names["SPall"] = f"PNall (Alle)"
    df_legend["SPall"] = f"Prüfungsnote alle SimPr (Alle SuS aller Serien)"
    all_en = "SPall_" + en
    df_order[all_en] = np.where(np.isnan(df[en]), np.nan, df_order["SPall"])
    df_names[all_en] = "PNall (mit EN)"
    df_legend[all_en] = "Prüfungsnote aller SimPr (SuS mit Erfolgsnote in einer der Serien)"
    en_all = en + "_all"
    df_order[en_all] = np.where(np.isnan(df_order["SPall"]), np.nan, df[en])
    df_names[en_all] = "ENall"
    df_legend[en_all] = "Erfolgsnote ZAP (SuS mit Note in einer der Serien)"

    df_order[en] = df[en]
    df_names[en] = f"EN"
    df_legend[en] = f"Erfolgsnote {en}"

    # df.info()
    df_stats_full = df_order.describe()
    df_stats = df_stats_full.loc[['count', 'mean', 'min', 'max']]
    df_stats.rename(index={'count': 'n', 'mean': 'Average', 'min': 'Minimum', 'max': 'Maximum'}, inplace=True)

    for col in df_order.columns:
        df_stats.loc['n bestanden (>= 4.5)', col] = (df_order[col] >= 4.5).sum()
        df_stats.loc['% bestanden (>= 4.5)', col] = (df_order[col] >= 4.5).sum() / df_order[col].count() * 100
        df_stats.loc['% bestanden (>= 4.5)', col] = f"{df_stats.loc['% bestanden (>= 4.5)', col]:.2f}%"

    
    sel_col = [col for col in df_order.columns if col.endswith(f"_{en}")]
    for col in sel_col:
        df_stats.loc['n Verbesserung', col] = (df_order[col] < df_order[en]).sum()
        df_stats.loc['% Verbesserung', col] = (df_order[col] < df_order[en]).sum() / df_order[col].count() * 100
        df_stats.loc['% Verbesserung', col] = f"{df_stats.loc['% Verbesserung', col]:.2f}%"
        df_stats.loc['Durchschn. Verbesserung', col] = (df_order[en] - df_order[col]).mean()
        df_stats.loc['r Korrelation', col] = df_order[col].corr(df_order[en])



    # Convert these rows to integer type
    count_rows = ['n', 'n bestanden (>= 4.5)', 'n Verbesserung']
    for row in count_rows:
        df_stats.loc[row] = df_stats.loc[row].astype(int, errors='ignore')



    # Save the stats table to a png file
    df_stats_plot = df_stats.copy()
    # df_stats_plot = pd.DataFrame(df_stats, index=['Beschreibung', 'n', 'Average', 'Minimum', 'Maximum', 'n bestanden (>= 4.5)', 
    #                                               '% Verbesserung', 'Durchschnittliche Verbesserung', 'r Korrelation'])
    df_stats_plot.rename(columns=df_names, inplace=True)

    #legend_text = "\n".join([f"{key}: {value}" for key, value in df_legend.items()])

    return df_order, df_stats_plot.style.format(precision=2, na_rep='')



def calc_stats_table_glc(df, df_names, sn, en):

    columns = [en] + sn
    df = df[columns].copy()        

    for col in df.columns:
        if col.startswith("GLC"):
            df[col] = df[col].astype(str).str.replace('%', '', regex=False)
            df[col] = pd.to_numeric((df[col]), errors='coerce')
            df[col] = df[col]/100
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].mask(df[col] < 1, np.nan, inplace=True)

    df_order = pd.DataFrame()
    for i, s in enumerate(sn):
        name1 = s + "_" + en
        name2 = en + "_" + f"{i+1}"
        df_order[s] = df[s]
        df_order[name1] = np.where(np.isnan(df[en]), np.nan, df[s])
        df_order[name2] = np.where(np.isnan(df[s]), np.nan, df[en])
        if s.startswith("GLC"):
            df_names[s] = f"Prüfungsnote GLC {en[-1]} (Alle SuS)"
            df_names[name1] = f"Prüfungsnote GLC {en[-1]} (SuS mit Erfolgsnote)"
            df_names[name2] = f"Erfolgsnote ZAP (SuS mit GLC)"
        else:
            df_names[s] = f"Prüfungsnote SimPr {en[-1]} (Alle SuS Serie {i+1})"
            df_names[name1] = f"Prüfungsnote SimPr {en[-1]} (SuS mit Erfolgsnot Serie {i+1})"
            df_names[name2] = f"Erfolgsnote ZAP (SuS mit Note in Serie {i+1})"


    # Add an column for all series
    if len(sn) > 1:
        df_order["SPall"] = df[sn].mean(axis=1)
        df_names["SPall"] = f"Prüfungsnote alle SimPr (Alle SuS aller Serien)"
        all_en = "SPall_" + en
        df_order[all_en] = np.where(np.isnan(df[en]), np.nan, df_order["SPall"])
        df_names[all_en] = "Prüfungsnote aller SimPr (SuS mit Erfolgsnote in einer der Serien)"
        en_all = en + "_all"
        df_order[en_all] = np.where(np.isnan(df_order["SPall"]), np.nan, df[en])
        df_names[en_all] = "Erfolgsnote ZAP (SuS mit Note in einer der Serien)"
    
    df_names[en] = f"Erfolgsnote {en}"

    df_order[en] = df[en]

    # df.info()
    df_stats_full = df_order.describe()
    df_stats = df_stats_full.loc[['count', 'mean', 'min', 'max']]
    df_stats.rename(index={'count': 'n', 'mean': 'Average', 'min': 'Minimum', 'max': 'Maximum'}, inplace=True)

    for col in df_order.columns:
        if col.startswith("GLC"):
            df_stats.loc['n bestanden (>= 0.5)', col] = (df_order[col] >= 0.5).sum()
            df_stats.loc['% bestanden (>= 0.5)', col] = (df_order[col] >= 0.5).sum() / df_order[col].count() * 100
            df_stats.loc['% bestanden (>= 0.5)', col] = f"{df_stats.loc['% bestanden (>= 0.5)', col]:.2f}%"
        else:
            df_stats.loc['n bestanden (>= 4.5)', col] = (df_order[col] >= 4.5).sum()
            df_stats.loc['% bestanden (>= 4.5)', col] = (df_order[col] >= 4.5).sum() / df_order[col].count() * 100
            df_stats.loc['% bestanden (>= 4.5)', col] = f"{df_stats.loc['% bestanden (>= 4.5)', col]:.2f}%"


    sel_col = [col for col in df_order.columns if col.endswith(f"_{en}")]
    for col in sel_col:
        if col.startswith("GLC"):
            df_stats.loc['n Verbesserung', col] = (df_order[col]*6 < df_order[en]).sum()
            df_stats.loc['% Verbesserung', col] = (df_order[col]*6 < df_order[en]).sum() / df_order[col].count() * 100
            df_stats.loc['% Verbesserung', col] = f"{df_stats.loc['% Verbesserung', col]:.2f}%"
            df_stats.loc['Durchschn. Verbesserung', col] = (df_order[en] - df_order[col]*6).mean()
            df_stats.loc['r Korrelation', col] = df_order[col].corr(df_order[en])
        else:
            df_stats.loc['n Verbesserung', col] = (df_order[col] < df_order[en]).sum()
            df_stats.loc['% Verbesserung', col] = (df_order[col] < df_order[en]).sum() / df_order[col].count() * 100
            df_stats.loc['% Verbesserung', col] = f"{df_stats.loc['% Verbesserung', col]:.2f}%"
            df_stats.loc['Durchschn. Verbesserung', col] = (df_order[en] - df_order[col]).mean()
            df_stats.loc['r Korrelation', col] = df_order[col].corr(df_order[en])



    # Convert these rows to integer type
    if s.startswith("GLC"):
        count_rows = ['n', 'n bestanden (>= 0.5)', 'n Verbesserung']
    else:
        count_rows = ['n', 'n bestanden (>= 4.5)', 'n Verbesserung']
    for row in count_rows:
        df_stats.loc[row] = df_stats.loc[row].astype(int, errors='ignore')



    # Save the stats table to a png file
    df_stats_plot = df_stats.copy()
    df_stats_plot.rename(columns=df_names, inplace=True)


    return df_order, df_stats_plot.style.format(precision=2, na_rep='')



def create_line_plot(df, serien, en, fig_name, x_labels = ['Serie 1', 'Serie 2', 'Serie 3', 'Serie 4', 'Serie 5']):
    """
    Creates a line plot from the given data, showing the development of averages
    across Serie 1, Serie 2, Serie 3, and Erfolgsnote.

    Args:
        data (dict): A dictionary containing the data, with keys
                      "Serie 1", "Serie 2", "Serie 3", and "Erfolgsnote",
                      and values representing the average values for each category
                      across time.
                      Example:
                      {
                          "Serie 1": [average_value_for_serie1],
                          "Serie 2": [average_value_for_serie2],
                          "Serie 3": [average_value_for_serie3],
                          "Erfolgsnote": [average_value_for_erfolgsnote],
                      }
    """

    not_all = [item for item in df.columns if not re.search(r"all", item)]
    df = df[not_all].copy()
    
    df_s = df[serien].copy()
    df_s.rename(columns={0: 'Serie 1', 1: 'Serie 2', 2: 'Serie 3', 3: 'Serie 4', 4: 'Serie 5'}, inplace=True)

    s_sen = [item for item in df.columns if re.search(r"_ZAP", item)]
    df_temp = df[df[en] >= 1].copy()
    df_sen = df_temp[s_sen].copy()
    df_sen.rename(columns={0: 'Serie 1', 1: 'Serie 2', 2: 'Serie 3', 3: 'Serie 4', 4: 'Serie 5'}, inplace=True)


    s_en = [item for item in df.columns if re.search(r"^ZAP", item)]
    df_en = df[s_en].copy()
    df_en = df_en.iloc[:, 0:len(serien)]  # Ensure it has the same number of series as df_s
    df_en.rename(columns={0: 'Serie 1', 1: 'Serie 2', 2: 'Serie 3', 3: 'Serie 4', 4: 'Serie 5'}, inplace=True)
    if serien[0].startswith("GLC"):
        df_en = df_en/6

    y_s = df_s.mean()
    s_err = df_s.std()
    y_sen = df_sen.mean()
    sen_err = df_sen.std()
    y_en = df_en.mean()
    en_err = df_en.std()

    x = x_labels  # x-axis labels for the series
    
        
    x_pos = np.arange(len(x))

    # Adjust x-positions for each series
    x_s_offset = x_pos - 0.05  # Shift "All SuS" to the left
    x_sen_offset = x_pos      # Keep "SuS mit Erfolgsnote" at the center
    x_en_offset = x_pos + 0.05  # Shift "ZAP Note" to the right

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot the lines.  Use the extracted data.
    plt.errorbar(x_s_offset, y_s, yerr=s_err, fmt='o', color='#4c72b0', capsize=5, label='Durchschnitt aller SuS dieser Serie')
    plt.errorbar(x_sen_offset, y_sen, yerr=sen_err, fmt='s', color='#81a8ca', capsize=5, label='Durchschnitt der SuS mit Erfolgsnote')
    plt.errorbar(x_en_offset, y_en, yerr=en_err, fmt='^', color="#2C755E", capsize=5, label='ZAP Note der SuS dieser Serie')


    # Add labels and title
    plt.ylabel("Durchschnittsnote mit Std. Abw.")
    plt.title("Durchschnittsnoten in allen Serien")
    plt.grid(True)
    plt.legend()

    # Rotate x-axis labels
    plt.xticks(x_pos, x, rotation=45, ha="right")  # Use original x labels
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')  # Save the figure as a PNG file
    plt.show()



def create_violin_plot(df, serien, en, xtic, fig_name):
    """
    Creates a violin plot to visualize the distribution of grades
    across Serie 1, Serie 2, Serie 3, and Erfolgsnote.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        serien (list): A list of column names representing the series
                       (e.g., ['Serie_1', 'Serie_2', 'Serie_3']).
        en (str): The column name representing the English grade
                    (e.g., 'English_Grade').
    """

    not_all = [item for item in df.columns if not re.search(r"all", item, re.IGNORECASE)]
    df_filtered = df[not_all].copy()

    df_s = df_filtered[serien].copy()
    df_s = df_s.rename(columns={i: f'Serie {i+1}' for i in range(len(serien))})
    df_s_melted = df_s.melt(var_name='Serie', value_name='Note')
    df_s_melted['Gruppe'] = 'Alle SuS'

    df_temp = df_filtered[df_filtered[en] >= 1].copy()
    s_sen = [item for item in df_temp.columns if any(s in item for s in serien) and "_ZAP" in item]
    df_sen = df_temp[s_sen].copy()
    df_sen = df_sen.rename(columns={i: f'Serie {i+1}' for i in range(len(serien))})
    df_sen_melted = df_sen.melt(var_name='Serie', value_name='Note')
    df_sen_melted['Gruppe'] = 'SuS mit Erfolgsnote'


    s_en = [item for item in df_filtered.columns if item.startswith("ZAP") and not any(s in item for s in ['all', '_ZAP'])]
    df_en = df_filtered[s_en].copy()
    df_en = df_en.iloc[:, :len(serien)]
    df_en = df_en.rename(columns={i: f'Serie {i+1}' for i in range(len(serien))})
    if serien[0].startswith("GLC"):
        df_en = df_en/6
    df_en_melted = df_en.melt(var_name='Serie', value_name='Note')
    df_en_melted['Gruppe'] = 'ZAP Note'

    plt.figure(figsize=(10, 6))
    """
    # Combine the data for plotting
    df_plot = pd.concat([df_s_melted, df_sen_melted, df_en_melted], ignore_index=True)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Serie', y='Note', hue='Gruppe', data=df_plot, dodge=True, palette={'Alle SuS': '#4c72b0',
                                                                                       'SuS mit Erfolgsnote': '#81a8ca',
                                                                                       'ZAP Note': "#2C755E"})
    """
    
    
    sns.violinplot(x='Serie', y='Note', hue='Gruppe', data=df_s_melted, dodge=True, palette={'Alle SuS': '#4c72b0',
                                                                                       'SuS mit Erfolgsnote': '#81a8ca',
                                                                                       'ZAP Note': "#2C755E"})                                                                                
    sns.violinplot(x='Serie', y='Note', hue='Gruppe', data=df_sen_melted, dodge=True, palette={'Alle SuS': '#4c72b0',
                                                                                       'SuS mit Erfolgsnote': '#81a8ca',
                                                                                       'ZAP Note': "#2C755E"})
    sns.violinplot(x='Serie', y='Note', hue='Gruppe', data=df_en_melted, dodge=True, palette={'Alle SuS': '#4c72b0',
                                                                                       'SuS mit Erfolgsnote': '#81a8ca',
                                                                                       'ZAP Note': "#2C755E"})

    # Add labels and title
    plt.ylabel("Note")
    plt.xlabel("Serie")
    plt.title("Notenverteilung in allen Serien")
    plt.grid(axis='y', linestyle='--')
    plt.legend(title='Gruppe', loc='lower right')
    plt.xticks(np.arange(len(xtic)), xtic)  # Set x-axis labels
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')  # Save the figure as a PNG file
    plt.show()



def create_pass_fail_bar_plot(df, threshold, serien, en, fig_name):
    """
    Creates stacked bar plots to illustrate pass/fail rates for different series
    and time points, given a passing threshold.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        threshold (float): The passing threshold.
        serien (list): List of series names (e.g., ["Serie 1", "Serie 2", "Serie 3"]).
        en (str):  Name of the Erfolgsnote column (e.g., "Erfolgsnote").
    """

    # 1. Filter out columns containing "all"
    not_all = [item for item in df.columns if not re.search(r"all", item, re.IGNORECASE)]
    df = df[not_all].copy()

    # 2. Create DataFrames for analysis
    df_s = df[serien + [en]].copy()
    df_s.rename(columns={df_s.columns[i]: f'Serie {i+1}' if i < len(serien) else 'Erfolgsnote' for i in range(len(df_s.columns))}, inplace=True)
    df_temp = df[df[en] >= 1].copy()
    df_sen = df_temp[[col for col in df.columns if any(s in col for s in serien) and "_ZAP" in col] + [en]].copy()
    df_sen.rename(columns={df_sen.columns[i]: f'Serie {i+1}' if i < len(serien) else 'Erfolgsnote' for i in range(len(df_sen.columns))}, inplace=True)
    df_en = df[[col for col in df.columns if col.startswith("ZAP") and not any(s in col for s in ['all', '_ZAP'])]].copy()
    if serien[0].startswith("GLC"):
        df_en = df_en/6
    df_en.rename(columns={df_en.columns[i]: f'Serie {i+1}' if i < len(serien) else 'Erfolgsnote' for i in range(len(df_en.columns))}, inplace=True)

    df_s = df_s.iloc[:, 0:len(serien)]
    df_sen = df_sen.iloc[:, 0:len(serien)]
    df_en = df_en.iloc[:, 0:len(serien)]

    # 3. Calculate pass/fail counts for each group
    def calculate_pass_fail(data, threshold):
        """Calculates pass/fail counts for a given DataFrame."""
        passed = (data >= threshold).sum(axis=0) # sum over rows
        failed = (data < threshold).sum(axis=0)
        total = passed + failed
        # Normalize to get percentages
        passed = (passed / total * 100).fillna(0).round(1)
        failed = (failed / total * 100).fillna(0).round(1)
        return pd.DataFrame({'Passed': passed, 'Failed': failed}, index=data.columns)

    pf_s = calculate_pass_fail(df_s, threshold)
    pf_sen = calculate_pass_fail(df_sen, threshold)
    pf_en = calculate_pass_fail(df_en, threshold)

    time_steps = ['Serie 1', 'Serie 2', 'Serie 3', 'Serie 4', 'Serie 5', 'Serie 6', 'Serie 7', 'Serie 8', 'Serie 9']
    time_steps = time_steps[:len(serien)]  # Adjust the length of time_steps based on the number of series
    df = pd.DataFrame({'Durchschnitt aller SuS dieser Serie': pf_s['Passed'],
                       'Durchschnitt der SuS mit Erfolgsnote': pf_sen['Passed'],
                       'ZAP Note der SuS dieser Serie': pf_en['Passed']})
    
    fig, ax = plt.subplots(layout='constrained', figsize=(10, 6))

    x = np.arange(len(time_steps))  # the label locations
    width = 0.25  # the width of the bars
    multiplier = 0
    colors = ['#4c72b0', '#81a8ca',  "#2C755E"]

    for i, (condition, grade) in enumerate(df.items()):
        # Create a bar plot for each condition
        offset = width * multiplier
        color = colors[i]
        rects = ax.bar(x + offset, grade, width, label=condition, color=color, alpha=0.8)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('Bestehensquote in allen Serien')
    ax.set_ylabel('Bestehensquote in %')
    ax.set_xticks(x + width, time_steps, rotation=45)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    stats_text = f"Threshold : {threshold}"
    
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    
    fig.savefig(fig_name, dpi=300, bbox_inches='tight')  # Save the figure as a PNG file



def calc_corr_table(df, df_names, ex_prefix, sn, en):
    """
    Calculates the correlation table for a DataFrame and returns a styled DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
        df_names (dict): A dictionary mapping column names to their display names.
        ex_prefix (str): The prefix for exercise columns.
        sn (str): The name of the first score column.
        en (str): The name of the second score column.
    """

    exercises = [col for col in df.columns if col.startswith(ex_prefix)]
    columns = [sn, en] + exercises
    df_corr = df[columns].copy()

    for col in df_corr.columns:
        df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')

    for col in [sn, en]:
        df_corr[col].mask(df_corr[col] < 1, np.nan, inplace=True)


    df_stats_full = df_corr.describe()
    df_stats = df_stats_full.loc[['count', 'mean', 'min', 'max']]
    df_stats.rename(index={'count': 'n', 'mean': 'Average', 'min': 'Minimum', 'max': 'Maximum'}, inplace=True)
    #df_stats.loc['Beschreibung'] = df_names[exercises]



    for i, col in enumerate(exercises):
        df_stats.loc['Aufgabennummer', col] = i + 1
        df_stats.loc['r mit SimPr Note', col] = df_corr[col].corr(df_corr[sn])
        df_stats.loc['r mit Erfolgsnote', col] = df_corr[col].corr(df_corr[en])
        df_stats.loc['r2 mit SimPr Note', col] = df_corr[col].corr(df_corr[sn])**2
        df_stats.loc['r2 mit Erfolgsnote', col] = df_corr[col].corr(df_corr[en])**2


    df_plot = df_stats[exercises].copy()
    df_plot = df_plot.iloc[[4, 0, 1, 2, 3, 5, 6, 7, 8]]  # Insert the new row at the top
    # df_plot.rename(columns=df_names, inplace=True)
    df_top = df_plot[:2].style.format(precision=0, na_rep='-')  # Top two rows
    df_bottom = df_plot[2:].style.format(precision=2, na_rep='-')
    df_final = df_top.concat(df_bottom)


    return df_final



def calc_corr_table_glc(df, df_names, ex_prefix, sn, en):
    """
    Calculates the correlation table for a DataFrame and returns a styled DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
        df_names (dict): A dictionary mapping column names to their display names.
        ex_prefix (str): The prefix for exercise columns.
        sn (str): The name of the first score column.
        en (str): The name of the second score column.
    """

    exercises = [col for col in df.columns if col.startswith(ex_prefix)]
    columns = [sn, en] + exercises
    df_corr = df[columns].copy()

    for col in df_corr.columns:
        if col.startswith("GLC"):
            df_corr[col] = df_corr[col].astype(str).str.replace('%', '', regex=False)  # Apply .str to the column
            df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')
            df_corr[col] = df_corr[col]/100
        else:
            df_corr[col] = pd.to_numeric(df_corr[col], errors='coerce')
            df_corr[col] = df_corr[col].mask(df_corr[col] < 1, np.nan) # corrected mask


    df_stats_full = df_corr.describe()
    df_stats = df_stats_full.loc[['count', 'mean', 'min', 'max']]
    df_stats.rename(index={'count': 'n', 'mean': 'Average', 'min': 'Minimum', 'max': 'Maximum'}, inplace=True)
    df_stats.loc['Beschreibung'] = df_names[exercises]


    for col in exercises:
        df_stats.loc['r mit GLC Note', col] = df_corr[col].corr(df_corr[sn])
        df_stats.loc['r mit Erfolgsnote', col] = df_corr[col].corr(df_corr[en])
        df_stats.loc['r2 mit GLC Note', col] = df_corr[col].corr(df_corr[sn])**2
        df_stats.loc['r2 mit Erfolgsnote', col] = df_corr[col].corr(df_corr[en])**2



    df_plot = df_stats[exercises].copy()
    df_plot = df_plot.iloc[[4, 0, 1, 2, 3, 5, 6, 7, 8]]
    # df_plot.rename(columns=df_names, inplace=True)


    return df_plot.style.format(precision=2, na_rep='')








def plot_histograms(df, serie, serie_en, serie_en_good, hist_title, title_en, title_good, title_all, fig_name, show_fig=False):
    plt.figure(figsize=(10, 6))
    plt.hist(df[serie], range=[1, 6], histtype='barstacked', color='#4c72b0', alpha=1, label=title_all)
    plt.hist(df[serie_en], range=[1, 6], histtype='barstacked', color='#55a868', alpha=0.7, label=title_en)
    plt.hist(df[serie_en_good], range=[1, 6], histtype='barstacked', color='#81a8ca', alpha=0.7, label=title_good)
    plt.title(hist_title)
    plt.xlabel('Note')
    plt.ylabel('Anzahl')
    plt.xlim(1, 6)
    # Total N to the plot
    N = len(df[serie].dropna())
    plt.text(0.05, 0.95, f'N={N}', transform=plt.gca().transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    plt.legend(framealpha=0.7, fontsize='x-small')
    plt.savefig(fig_name, dpi=300, bbox_inches='tight') # Save the figure as a PNG file
    if show_fig:
        plt.show()



def plot_histograms_glc(df, serie, serie_en, serie_en_good, hist_title, title_en, title_good, title_all, fig_name, show_fig=False):
    plt.figure(figsize=(10, 6))
    plt.hist(df[serie], range=[0, 1], histtype='barstacked', color='#4c72b0', alpha=1, label=title_all)
    plt.hist(df[serie_en], range=[0, 1], histtype='barstacked', color='#55a868', alpha=0.7, label=title_en)
    plt.hist(df[serie_en_good], range=[0, 1], histtype='barstacked', color='#81a8ca', alpha=0.7, label=title_good)
    plt.title(hist_title)
    plt.xlabel('Note')
    plt.ylabel('Anzahl')
    plt.xlim(0, 1)
    N = len(df[serie].dropna())
    plt.text(0.05, 0.95, f'N={N}', transform=plt.gca().transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    plt.legend(framealpha=0.7, fontsize='x-small')
    plt.savefig(fig_name, dpi=300, bbox_inches='tight') # Save the figure as a PNG file
    if show_fig:
        plt.show()



def plot_success_chance(df, sim_exam_col, serie_nr, final_exam_col, final_pass_threshold, fig_name='success_chance.png', bin_size=0.5, variability='stderr', show_fig=False):
    """
    Visualizes the chance of succeeding in the final exam based on the simulation exam grade,
    with grades grouped into bins to smooth the curve and an area showing variability.

    Args:
        df (pd.DataFrame): DataFrame containing simulation exam grades and final exam results.
        sim_exam_col (str): Name of the column containing simulation exam grades.
        final_exam_col (str): Name of the column containing final exam grades.
        final_pass_threshold (float): The passing grade threshold for the final exam.
        sim_pass_threshold (float, optional): The passing grade threshold for the simulation exam.
                                              If provided, a vertical line will be added to the plot.
        bin_size (float): The size of the grade bins for grouping.
        variability (str): The measure of variability to show ('stderr' for standard error,
                           'std' for standard deviation). Defaults to 'stderr'.
    """

    df = df.dropna(subset=[final_exam_col]).copy() # .copy() to avoid SettingWithCopyWarning

    # Create a binary 'Passed Final' column
    df['Passed Final'] = df[final_exam_col] >= final_pass_threshold

    # Create bins for the simulation exam grades
    # min_grade = math.floor(df[sim_exam_col].min())
    min_grade = 1
    # max_grade = math.ceil(df[sim_exam_col].max())
    max_grade = 6
    
    custom_bins = np.arange(min_grade, max_grade + bin_size, bin_size)
    custom_bins[-1] += np.finfo(float).eps
    bins = pd.cut(df[sim_exam_col], bins=custom_bins, include_lowest=True, right=False)


    #bins = pd.cut(df[sim_exam_col], bins=pd.interval_range(start=min_grade, end=max_grade, freq=bin_size), include_lowest=True, right=False)

    # Calculate the probability of passing and variability for each bin
    grouped = df.groupby(bins)['Passed Final'].agg(['mean', 'sem', 'std', 'count']).reset_index()
    grouped = grouped.dropna(subset=['mean'])

    bin_centers = [(interval.left + interval.right) / 2 for interval in grouped[sim_exam_col]]

    # Determine the upper and lower bounds for the variability area
    if variability == 'stderr':
        upper_bound = grouped['mean'] + grouped['sem']
        lower_bound = grouped['mean'] - grouped['sem']
        y_label_variability = 'Standard Error'
    elif variability == 'std':
        upper_bound = grouped['mean'] + grouped['std']
        lower_bound = grouped['mean'] - grouped['std']
        y_label_variability = 'Standard Deviation'
    else:
        raise ValueError("Invalid value for 'variability'. Choose 'stderr' or 'std'.")

    # Create the line plot with variability area
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, grouped['mean'], marker='o', label='Bestehensquote')
    plt.fill_between(bin_centers, lower_bound, upper_bound, alpha=0.3, label=f'Variability ({y_label_variability})')

    # --- Add N to the plot ---
    for i, N in enumerate(grouped['count']):
        plt.text(bin_centers[i], grouped['mean'].iloc[i]+0.03, f'N={int(N)}', ha='center', va='bottom', fontsize=9, color='black')
    # -------------------------

    # --- Add Trendline ---
    # Calculate the trendline (e.g., a linear fit)
    # You can change the degree of the polynomial (e.g., 2 for quadratic)
    z = np.polyfit(bin_centers, grouped['mean'], 1) # 1 for a linear trendline
    p = np.poly1d(z)

    # Plot the trendline
    plt.plot(bin_centers, p(bin_centers), "r--", label='Trendline') # Red dashed line for the trend
    # -------------------

    # Add labels and title
    plt.xlabel(f"Simulations Prüfung Serie {serie_nr} \n"
               f"(Bestanden: EN >= {final_pass_threshold}, bucket-size: {bin_size})")
    plt.ylabel("Bestehensquote in der ZAP Prüfung")
    plt.title("Erfolgsquote in der ZAP Prüfung basierend auf der SimPrüfung")
    plt.grid(True)
    plt.ylim(0, 1.05) # Set y-axis limits for probability
    N = sum(grouped['count'])
    plt.text(0.05, 0.95, f'N={N}', transform=plt.gca().transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')  # Save the figure as a PNG file
    if show_fig:
        plt.show()



def plot_success_chance_glc(df, sim_exam_col, final_exam_col, final_pass_threshold, fig_name='success_chance.png', bin_size=0.1, variability='stderr', show_fig=False):
    """
    Visualizes the chance of succeeding in the final exam based on the simulation exam grade,
    with grades grouped into bins to smooth the curve and an area showing variability.

    Args:
        df (pd.DataFrame): DataFrame containing simulation exam grades and final exam results.
        sim_exam_col (str): Name of the column containing simulation exam grades.
        final_exam_col (str): Name of the column containing final exam grades.
        final_pass_threshold (float): The passing grade threshold for the final exam.
        sim_pass_threshold (float, optional): The passing grade threshold for the simulation exam.
                                              If provided, a vertical line will be added to the plot.
        bin_size (float): The size of the grade bins for grouping.
        variability (str): The measure of variability to show ('stderr' for standard error,
                           'std' for standard deviation). Defaults to 'stderr'.
    """

    df = df.dropna(subset=[final_exam_col]).copy()

    # Create a binary 'Passed Final' column
    df['Passed Final'] = df[final_exam_col] >= final_pass_threshold

    # Create bins for the simulation exam grades
    # min_grade = math.floor(df[sim_exam_col].min())
    min_grade = 0
    # max_grade = math.ceil(df[sim_exam_col].max())
    max_grade = 1
    
    custom_bins = np.arange(min_grade, max_grade + bin_size, bin_size)
    custom_bins[-1] += np.finfo(float).eps
    bins = pd.cut(df[sim_exam_col], bins=custom_bins, include_lowest=True, right=False)


    #bins = pd.cut(df[sim_exam_col], bins=pd.interval_range(start=min_grade, end=max_grade, freq=bin_size), include_lowest=True, right=False)

    # Calculate the probability of passing and variability for each bin
    grouped = df.groupby(bins)['Passed Final'].agg(['mean', 'sem', 'std', 'count']).reset_index()
    grouped = grouped.dropna(subset=['mean'])

    bin_centers = [(interval.left + interval.right) / 2 for interval in grouped[sim_exam_col]]

    # Determine the upper and lower bounds for the variability area
    if variability == 'stderr':
        upper_bound = grouped['mean'] + grouped['sem']
        lower_bound = grouped['mean'] - grouped['sem']
        y_label_variability = 'Standard Error'
    elif variability == 'std':
        upper_bound = grouped['mean'] + grouped['std']
        lower_bound = grouped['mean'] - grouped['std']
        y_label_variability = 'Standard Deviation'
    else:
        raise ValueError("Invalid value for 'variability'. Choose 'stderr' or 'std'.")

    # Create the line plot with variability area
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, grouped['mean'], marker='o', label='Bestehensquote')
    plt.fill_between(bin_centers, lower_bound, upper_bound, alpha=0.3, label=f'Variability ({y_label_variability})')

    # --- Add N to the plot ---
    for i, N in enumerate(grouped['count']):
        plt.text(bin_centers[i], grouped['mean'].iloc[i]+0.03, f'N={int(N)}', ha='center', va='bottom', fontsize=9, color='black')
    # -------------------------

    # --- Add Trendline ---
    # Calculate the trendline (e.g., a linear fit)
    # You can change the degree of the polynomial (e.g., 2 for quadratic)
    z = np.polyfit(bin_centers, grouped['mean'], 1) # 1 for a linear trendline
    p = np.poly1d(z)

    # Plot the trendline
    plt.plot(bin_centers, p(bin_centers), "r--", label='Trendline') # Red dashed line for the trend
    # -------------------

    # Add labels and title
    plt.xlabel(f"GLC Prüfung \n"
               f"(Bestanden: EN >= {final_pass_threshold}, bucket-size: {bin_size})")
    plt.ylabel("Bestehensquote in der ZAP Prüfung")
    plt.title("Erfolgsquote in der ZAP Prüfung basierend auf dem GLC")
    plt.grid(True, which='both')
    plt.ylim(0, 1.05) # Set y-axis limits for probability
    N = sum(grouped['count'])
    plt.text(0.05, 0.95, f'N={N}', transform=plt.gca().transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_name, dpi=300, bbox_inches='tight')  # Save the figure as a PNG file
    if show_fig:
        plt.show()



def generate_plots(df, sn, en, xtic, x_labels_lineplot, show_fig=False):
    """
    Generates scatter plots and histograms for the given DataFrame and series.
    Args:
        df (pd.DataFrame): The input DataFrame with the extra columns from calc_stats_table.
        sn (list): List of series names.
        en (str): The name of the exam score column.
    """
    fach = en.split("-")[1]


    create_line_plot(df, sn, en, fig_name=f"{fach.lower()}_line_plot.png", x_labels=x_labels_lineplot)
    create_violin_plot(df, sn, en, xtic, fig_name=f"{fach.lower()}_violin_plot.png")
    create_pass_fail_bar_plot(df, threshold=4.5, serien=sn, en=en, fig_name=f"{fach.lower()}_pass_fail_bar_plot.png")

    for i, s in enumerate(sn):
        df[f"{s}_{en}_good"] = np.where(df[en] >= 4.5, df[f"{s}_{en}"], np.nan)
        if s.startswith("GLC"):
            plot_scatter_trend(x_data=df[s], y_data=df[en], dot_size=30, show_regression_line=True, x_label="GLC", y_label="Erfolgsnote", 
                               title=f"Prüfungsnote GLC {fach} vs. {en}", fig_name=f"{fach.lower()}_glc_vs_{en.lower()}.png", show_fig=show_fig)
            plot_histograms_glc(df = df, serie=s, serie_en=f"{s}_{en}", serie_en_good=f"{s}_{en}_good",
                                hist_title=f"Histogramme GLC {fach}",
                                title_en=f"Prüfungsnote GLC {fach} (Erfolgsnote vorhanden)",
                                title_good=f"Prüfungsnote GLC {fach} (Erfolgsnote >= 4.5)",
                                title_all=f"Prüfungsnote GLC {fach} (Alle)",
                                fig_name=f"{fach.lower()}_glc_histograms.png",
                                show_fig=show_fig)
        
        else:
            plot_scatter_trend(x_data=df[s], y_data=df[en], dot_size=30, show_regression_line=True, x_label="SimPr", y_label="Erfolgsnote", 
                               title=f"Prüfungsnote SimPr fach Serie {i+1} vs. {en}", fig_name=f"{fach.lower()}_simpr{i+1}_vs_{en.lower()}.png", show_fig=show_fig)
            plot_histograms(df = df, serie=s, serie_en=f"{s}_{en}", serie_en_good=f"{s}_{en}_good",
                            hist_title=f"Histogramme SimPr {fach}",
                            title_en=f"Prüfungsnote SimPr {fach} Serie {i+1} (Erfolgsnote vorhanden)",
                            title_good=f"Prüfungsnote SimPr {fach} Serie {i+1} (Erfolgsnote >= 4.5)",
                            title_all=f"Prüfungsnote SimPr {fach} Serie {i+1} (Alle)",
                            fig_name=f"{fach.lower()}_simpr{i+1}_histograms.png",
                            show_fig=show_fig)
            plot_success_chance(df, sim_exam_col=s, serie_nr=i+1, final_exam_col=en, final_pass_threshold=4.5, 
                                fig_name=f"{fach.lower()}_simpr{i+1}_success_chance.png", show_fig=show_fig)
    print("Plots generated successfully!")



def generate_plots_tryexcept(df, sn, en, xtic, x_labels_lineplot, show_fig=False):
    """
    Generates scatter plots and histograms for the given DataFrame and series.
    Args:
        df (pd.DataFrame): The input DataFrame with the extra columns from calc_stats_table.
        sn (list): List of series names.
        en (str): The name of the exam score column.
    """
    fach = en.split("-")[1]

    
    # Try creating main plots
    try:
        create_line_plot(df, sn, en, fig_name=f"{fach.lower()}_line_plot.png", x_labels=x_labels_lineplot)
    except Exception as e:
        print(f"Warning: Failed to create line plot ({fach.lower()}_line_plot.png). Error: {e}")

    try:
        create_violin_plot(df, sn, en, xtic, fig_name=f"{fach.lower()}_violin_plot.png")
    except Exception as e:
        print(f"Warning: Failed to create violin plot ({fach.lower()}_violin_plot.png). Error: {e}")

    try:
        create_pass_fail_bar_plot(df, threshold=4.5, serien=sn, en=en, fig_name=f"{fach.lower()}_pass_fail_bar_plot.png")
    except Exception as e:
        print(f"Warning: Failed to create pass/fail bar plot ({fach.lower()}_pass_fail_bar_plot.png). Error: {e}")
    
    try:
        plot_success_chance(df, "SPall", "Durchschnitt", final_exam_col=en, final_pass_threshold=4.5,
                            fig_name=f"{fach.lower()}_all_success_chance.png", show_fig=show_fig)
    except Exception as e:
        print(f"Warning: Failed to success chance plot (all_success_chance.png). Error: {e}")

    for i, s in enumerate(sn):
        df[f"{s}_{en}_good"] = np.where(df[en] >= 4.5, df[f"{s}_{en}"], np.nan)

        if s.startswith("GLC"):
            try:
                plot_scatter_trend(x_data=df[s], y_data=df[en], dot_size=30, show_regression_line=True, x_label="GLC", y_label="Erfolgsnote",
                                   title=f"Prüfungsnote GLC {fach} vs. {en}", fig_name=f"{fach.lower()}_glc_vs_{en.lower()}.png", show_fig=show_fig)
            except Exception as e:
                print(f"Warning: Failed to create GLC scatter trend plot for {s}. Error: {e}")

            try:
                plot_histograms_glc(df = df, serie=s, serie_en=f"{s}_{en}", serie_en_good=f"{s}_{en}_good",
                                    hist_title=f"Histogramme GLC {fach}",
                                    title_en=f"Prüfungsnote GLC {fach} (Erfolgsnote vorhanden)",
                                    title_good=f"Prüfungsnote GLC {fach} (Erfolgsnote >= 4.5)",
                                    title_all=f"Prüfungsnote GLC {fach} (Alle)",
                                    fig_name=f"{fach.lower()}_glc_histograms.png",
                                    show_fig=show_fig)
            except Exception as e:
                print(f"Warning: Failed to create GLC histograms for {s}. Error: {e}")

            try:
                plot_success_chance_glc(df, sim_exam_col=s, final_exam_col=en, final_pass_threshold=4.5,
                                    fig_name=f"{fach.lower()}_glc_success_chance.png", show_fig=show_fig)
            except Exception as e:
                print(f"Warning: Failed to create success chance plot for {s}. Error: {e}")

        else: # For other series (presumably SimPr)
            try:
                plot_scatter_trend(x_data=df[s], y_data=df[en], dot_size=30, show_regression_line=True, x_label="Prüfungsnote", y_label="Erfolgsnote",
                                   title=f"Prüfungsnote SimPr {fach} Serie {i+1} vs. {en}", fig_name=f"{fach.lower()}_simpr{i+1}_vs_{en.lower()}.png", show_fig=show_fig)
            except Exception as e:
                print(f"Warning: Failed to create SimPr scatter trend plot for {s}. Error: {e}")

            try:
                plot_histograms(df = df, serie=s, serie_en=f"{s}_{en}", serie_en_good=f"{s}_{en}_good",
                                hist_title=f"Histogramme SimPr {fach}",
                                title_en=f"Prüfungsnote SimPr {fach} Serie {i+1} (Erfolgsnote vorhanden)",
                                title_good=f"Prüfungsnote SimPr {fach} Serie {i+1} (Erfolgsnote >= 4.5)",
                                title_all=f"Prüfungsnote SimPr {fach} Serie {i+1} (Alle)",
                                fig_name=f"{fach.lower()}_simpr{i+1}_histograms.png",
                                show_fig=show_fig)
            except Exception as e:
                print(f"Warning: Failed to create SimPr histograms for {s}. Error: {e}")

            try:
                plot_success_chance(df, sim_exam_col=s, serie_nr=i+1, final_exam_col=en, final_pass_threshold=4.5,
                                    fig_name=f"{fach.lower()}_simpr{i+1}_success_chance.png", show_fig=show_fig)
            except Exception as e:
                print(f"Warning: Failed to create success chance plot for {s}. Error: {e}")

    print("Plots generated successfully!")