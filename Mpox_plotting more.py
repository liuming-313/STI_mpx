import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import sciris as sc
import pylab as pl
import pandas as pd
import datetime as dt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
# Read the data from 'STI_rate.xlsx'
data = pd.read_excel('STI_rate.xlsx', sheet_name='Sheet1')
def format_ax(ax, key=None,start=-5, end=530):
    ''' Format the axes nicely '''
    @ticker.FuncFormatter
    def date_formatter(x, pos):
        date_obj = dt.date(2022, 5, 11) + dt.timedelta(days=x)
        return date_obj.strftime('%b-%d-%Y')

    ax.xaxis.set_major_formatter(date_formatter)
    if key != 'r_eff':
        sc.commaticks()
    pl.xlim([start, end])
    sc.boxoff()
    return




def STI_male_MSM():
    # Select the columns you want to use
    selected_cols = ['Disease', 'Male ratio', 'MSM ratio', 'Sex contact ratio', 'Region', 'Year']
    df = data[selected_cols]

    # Create the figure with figsize specified
    plt.figure(figsize=(12, 8))

    # Set X and Y limits
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    # Set X and Y ticks
    plt.xticks(np.arange(0, 1.1, step=0.2), [str(i)  for i in np.arange(0, 110, step=20)])
    plt.yticks(np.arange(0.2, 1.1, step=0.2), [str(i)  for i in np.arange(20, 110, step=20)])
    # Turn off the right and top axes
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Get unique disease categories and assign a color to each category
    unique_diseases = df['Disease'].unique()
    n_unique_diseases = len(unique_diseases)
    colors = sns.color_palette("tab20", n_colors=n_unique_diseases)

    # Create a color dictionary for mapping disease to color
    color_dict = dict(zip(unique_diseases, colors))

    # Map disease names to colors in the DataFrame
    df['Color'] = df['Disease'].map(color_dict)
    markers={}
    region_legend_labels = list(df['Region'].unique()) # Sort the unique regions
    marker_shapes= ["d",'*','^','p','P','X']  # Specify the marker shapes for each region
    for i,region in enumerate(region_legend_labels):
        markers[region]=marker_shapes[i]
    # Create the scatter plot with custom colors
    scatter = sns.scatterplot(data=df, x="MSM ratio", y="Male ratio", hue="Disease", palette=color_dict, style="Region", size="Year", sizes=(50, 200),markers=markers)

    # Set labels and title
    plt.xlabel("MSM ratio (%)")
    plt.ylabel("Male ratio (%)")
    #plt.title("Disease Data")

    # Create a legend for diseases with custom colors and labels
    legend_labels = unique_diseases
    legend_colors = [color_dict[label] for label in unique_diseases]
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=5, markerfacecolor=color) for color in legend_colors]
    legend1=plt.legend(legend_handles, legend_labels,  loc='upper left',frameon=False, markerscale=2, labelspacing=0.2, handlelength=1, handletextpad=0.5, borderpad=0.5, title_fontsize='x-large', fontsize='large')

    # Manually specify legend labels for Year and Region
    year_legend_labels = sorted(df['Year'].unique())  # Sort the unique years


    year_marker_sizes =np.linspace(5,9,10,endpoint=True)  # Specify the marker sizes for each year

    # Create a custom legend for "Year" and "Region" (lower right)
    legend2_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markersize=marker_size, markerfacecolor='grey', markeredgecolor='white',label=str(year))
        for year,marker_size in zip(year_legend_labels,year_marker_sizes)
    ] + [
        plt.Line2D([0], [0], marker=marker_shape, color='w', markersize=6, markerfacecolor='white', markeredgecolor='black', label=region)
        for region,marker_shape in zip(region_legend_labels,marker_shapes)
    ]

    legend2_labels = year_legend_labels + region_legend_labels
    legend2 = plt.legend(
        handles=legend2_handles,
        labels=legend2_labels,
        loc='lower right',
        frameon=False,
        markerscale=1.5,
        labelspacing=0.2,
        handlelength=1,
        handletextpad=0.5,
        borderpad=0.5,
        title_fontsize='x-large',
        fontsize='large',
    )

    # Add both legends to the plot
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.savefig(f"../results/Mpox/STI_male_MSM.svg", format='svg', dpi=1200)
    # Show the plot
    plt.show()
    return
#STI_male_MSM()
def disease_HIV_MSM():
    # Select the columns you want to use
    selected_cols = ['Disease',  'MSM ratio', 'Sex contact ratio', 'Region', 'Year','HIV+ in all',"HIV+ in MSM","HIV+ in non-MSM"]
    df = data[selected_cols]
    #just leave df with 'HIV+ in all' not None
    df=df[df['HIV+ in all'].notna()]
    # Create the figure with figsize specified
    plt.figure(figsize=(12, 8))

    # Set X and Y limits
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)


    # Set X and Y ticks
    plt.xticks(np.arange(0, 1.1, step=0.2), [str(i)  for i in np.arange(0, 110, step=20)])
    plt.yticks(np.arange(0.2, 1.1, step=0.2), [str(i)  for i in np.arange(20, 110, step=20)])

    # Turn off the right and top axes
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # Get unique disease categories and assign a color to each category
    unique_diseases = df['Disease'].unique()
    n_unique_diseases = len(unique_diseases)
    colors = sns.color_palette("tab20", n_colors=n_unique_diseases)

    # Create a color dictionary for mapping disease to color
    color_dict = dict(zip(unique_diseases, colors))

    # Map disease names to colors in the DataFrame
    df['Color'] = df['Disease'].map(color_dict)
    markers = {}
    markers_MSM ={}
    region_legend_labels = list(df['Region'].unique())  # Sort the unique regions
    marker_shapes = ["d", '*', '^', '$f$', 'P', 'X']  # Specify the marker shapes for each region
    for i, region in enumerate(region_legend_labels):
        markers[region] = marker_shapes[i]
        markers_MSM[region]=marker_shapes[i]+'/'
    # Create the scatter plot with custom colors
    scatter = sns.scatterplot(data=df, x="MSM ratio", y="HIV+ in all", hue="Disease", palette=color_dict, style="Region",
                              size="Year", sizes=(50, 200), markers=markers,edgecolor='black')

    plt.xlabel("MSM ratio (%)")
    plt.ylabel("HIV positive rate (%)")
    plt.title("")
    sns.scatterplot(data=df, x="MSM ratio", y="HIV+ in MSM", hue="Disease", palette=color_dict, style="Region",
                              size="Year", sizes=(50, 200), markers=markers, edgecolor='red')
    sns.scatterplot(data=df, x="MSM ratio", y="HIV+ in non-MSM", hue="Disease", palette=color_dict, style="Region",
                              size="Year", sizes=(50, 200), markers=markers, edgecolor='green')
    # Set labels and title
    plt.ylim(0, 1.1)
    # Turn off the right and top axes
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)


    # Create a legend for diseases with custom colors and labels
    legend_labels = unique_diseases
    legend_colors = [color_dict[label] for label in unique_diseases]
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markersize=5, markerfacecolor=color) for color in
                      legend_colors]
    legend1 = plt.legend(legend_handles, legend_labels, loc='upper left', frameon=False, markerscale=2,
                         labelspacing=0.2, handlelength=1, handletextpad=0.5, borderpad=0.5, title_fontsize='x-large',
                         fontsize='large')

    # Manually specify legend labels for Year and Region
    year_legend_labels = sorted(df['Year'].unique())  # Sort the unique years

    year_marker_sizes = np.linspace(5, 9, 10, endpoint=True)  # Specify the marker sizes for each year

    # Create a custom legend for "Year" and "Region" (lower right)
    legend2_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markersize=marker_size, markerfacecolor='grey',
                   markeredgecolor='white', label=str(year))
        for year, marker_size in zip(year_legend_labels, year_marker_sizes)
    ]

    legend2_handles.extend([
        plt.Line2D([0], [0], marker='o', color='w', markersize=6, markerfacecolor='white',
                   markeredgecolor=markeredge_color, label=label)
        for markeredge_color, label in zip(['red', 'black','green'], ['HIV positive rate in MSM','HIV positive rate in all', 'HIV positive rate in non-MSM'])
    ])

    legend2_handles.extend([
        plt.Line2D([0], [0], marker=marker_shape, color='w', markersize=6, markerfacecolor='white',
                   markeredgecolor='black', label=region)
        for region, marker_shape in zip(region_legend_labels, marker_shapes)
    ])

    legend2_labels = year_legend_labels +['HIV positive rate in MSM','HIV positive rate in all', 'HIV positive rate in non-MSM']+ region_legend_labels
    legend2 = plt.legend(
        handles=legend2_handles,
        labels=legend2_labels,
        loc='lower left',
        frameon=False,
        markerscale=1.5,
        labelspacing=0.2,
        handlelength=1,
        handletextpad=0.5,
        borderpad=0.5,
        title_fontsize='x-large',
        fontsize='large',
    )

    # Add both legends to the plot
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.savefig(f"../results/Mpox/HIV_MSM.svg", format='svg', dpi=1200)
    # Show the plot
    plt.show()
    return
#disease_HIV_MSM()


def male_ratio():
    """
    the data is in the Mpox_male_MSM_data.xlsx, Male sheet, which contain Region	Male ratio	X_pos
    make a dot line plot,different color for different region
    :return:

    """
    # Read the data from 'Mpox_male_MSM_data.xlsx'
    data = pd.read_excel('Mpox_male_MSM_data.xlsx', sheet_name='Male')
    # plot the figure
    plt.figure(figsize=(12, 8))
    # Set X and Y limits
    plt.xlim(0, 1.1)
    plt.ylim(0.75, 1.03)
    # Set X and Y ticks

    plt.xticks(np.arange(0.2, 1.1, step=0.2), [str(i) for i in np.arange(20, 110, step=20)])
    plt.yticks(np.arange(0.75, 1.02, step=0.05), [str(i) for i in np.arange(75, 102, step=5)])
    # Turn off the right and top axes
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # Get unique region categories and assign a color to each category
    unique_region = data['Region'].unique()
    n_unique_region = len(unique_region)
    colors = sns.color_palette("tab20", n_colors=n_unique_region)
    # Create a color dictionary for mapping region to color
    color_dict = dict(zip(unique_region, colors))
    # Map region names to colors in the DataFrame
    data['Color'] = data['Region'].map(color_dict)
    # Create the scatter plot with custom colors
    for region, group in data.groupby('Region'):
        plt.scatter(group['X_pos'],group['Male ratio'], label=region, c=group['Color'], s=100, alpha=0.7)

        # Connect the dots with lines, specifying the color directly
        color = color_dict[region]
        plt.plot(group['X_pos'],group['Male ratio'],  color=color, alpha=0.7, linestyle='-', linewidth=2)

    # Add labels and legend
    plt.xlabel('Relative Proportion in Each Region')
    plt.ylabel('Proportion of Male Cases in All Cases')
    plt.legend(title='Region')
    plt.savefig(f"../results/Mpox/male_ratio.svg", format='svg', dpi=1200)
    # Show the plot
    plt.show()





##Region	MSM ratio	X_pos
def MSM_ratio():
    """
    the data is in the Mpox_male_MSM_data.xlsx, MSM sheet, which contain Region	MSM ratio	X_pos

    make a dot line plot,different color for different region
    :return:

    """
    # Read the data from 'Mpox_male_MSM_data.xlsx'
    data = pd.read_excel('Mpox_male_MSM_data.xlsx', sheet_name='MSM')
    # plot the figure
    plt.figure(figsize=(12, 8))
    # Set X and Y limits
    plt.xlim(0, 1.1)
    plt.ylim(0.6, 1.03)
    # Set X and Y ticks

    plt.xticks(np.arange(0.2, 1.1, step=0.2), [str(i) for i in np.arange(20, 110, step=20)])
    plt.yticks(np.arange(0.6, 1.02, step=0.05), [str(i) for i in np.arange(60, 102, step=5)])
    # Turn off the right and top axes
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # Get unique region categories and assign a color to each category
    unique_region = data['Region'].unique()
    n_unique_region = len(unique_region)
    colors = sns.color_palette("tab20", n_colors=n_unique_region)
    # Create a color dictionary for mapping region to color
    color_dict = dict(zip(unique_region, colors))
    # Map region names to colors in the DataFrame
    data['Color'] = data['Region'].map(color_dict)
    # Create the scatter plot with custom colors
    for region, group in data.groupby('Region'):
        plt.scatter(group['X_pos'],group['MSM ratio'], label=region, c=group['Color'], s=100, alpha=0.7)

        # Connect the dots with lines, specifying the color directly
        color = color_dict[region]
        plt.plot(group['X_pos'],group['MSM ratio'],  color=color, alpha=0.7, linestyle='-', linewidth=2)

    # Add labels and legend
    plt.xlabel('Relative proportion in each region')
    plt.ylabel('Proportion of MSM cases ')
    plt.legend(title='Region')
    plt.savefig(f"../results/Mpox/MSM_ratio.svg", format='svg', dpi=1200)

    # Show the plot
    plt.show()




def plot_trans():
    """
    this function will plot the mpox transmission from Gay, bisexual males, MSM, Heterosexual Transmission and Lesbian.
    data will be in two shapes:
    if the proportion of gay is None, just plot the MSM and Heterosexual Transmission,WSW
    if the proportion of gay is not None, plot Gay, bisexual males, Heterosexual Transmission,WSW
    it is a horizontal stacked bar plot with different color for different transmission
    """
    locations = [
        "China",
        "City and County of San Francisco, United States",
        "County of San Diego, United States",
        "City of Chicago, United States",
        "England, United Kingdom",
        "King County, United States",
        "District of Columbia (D.C.), United States",
        "County of Los Angeles, United States",
        "County of Alameda, United States",
        "State of California, United States",
        "City of New York, United States",
        "State of Oregon, United States",
        "State of Michigan, United States",
        "Brazil"
    ]
    Gay=["NA", 91.3, 82.2, 85.6, "NA", "NA", 82.4, 77.4, 72.0, 78.6, "NA", 79.3, "NA", 73.7]
    Bisexual_male=["NA", 4.9, 10.6, 8.6, "NA", "NA", 9.8, 12.6, 16.9, 11.0, "NA", 13.6, "NA", 10.8]
    MSM= [94.1, 96.2, 92.8, 94.2, 96.8, 96.4, 92.2, 90.0, 88.9, 89.6, 88.5, 92.9, 82.1, 84.5]
    Heterosexual_Transmission= [5.9, 3.8, 7.2, 5.8, 3.2, 3.6, 7.8, 10.0, 11.1, 10.3, 11.5, 7.1, 17.9, 15.3]
    WSW= [0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0, 0,0]
    Male_ratio=[99.3,99.3,99.0,98.9,98.6,98.3,98.1,98.1,97.9,97.4,97.2,96.7,95.5,92.0]
    #the RGB color for Gay, bisexual males, MSM, Heterosexual Transmission and Lesbian
    colors_array = np.array([[132, 60, 12],[197,90,17],[248,203,173],[251,229,214]])/255
    # plot the figure
    plt.figure(figsize=(14, 8))
    # Set X and Y limits
    plt.xlim(0, 100)
    plt.ylim(-1, 14)
    # Set X and Y ticks
    plt.xticks(np.arange(0, 110, step=10), [str(i) for i in np.arange(0, 110, step=10)])
    plt.yticks(np.arange(0,len(locations)), [str(i) for i in locations])
    # Turn off the right and top axes
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    # plot the horizontal stacked bar plot
    # Define transmission categories

    y_pos=0
    for i in range(len(locations)):
    # Loop through each region

        if Gay[i] == "NA":
        # Define the y positions for bars for the current region
            transmissions=['MSM','Other transmission']
            # Plot the horizontal stacked bar for the current transmission categor
            plt.barh(y_pos, MSM[i], color=(colors_array[0]+colors_array[1])/2, label="MSM")
            plt.barh(y_pos, Heterosexual_Transmission[i], color=(colors_array[2]+colors_array[3])/2,left=MSM[i], label=transmissions[1],alpha=.5)
        else:
            transmissions = ['Homosexual males', 'Bisexual males', 'Heterosexual males and females', 'Homosexual females']
            data=np.array([Gay[i],Bisexual_male[i],Heterosexual_Transmission[i],WSW[i]])
            for j in range(len(transmissions)):
                if transmissions[j]!='Homosexual females':
                    plt.barh(y_pos,data[j],color=colors_array[j],left=np.sum(data[:j]),label=transmissions[j])
                else:
                    plt.barh(y_pos, data[j], color='grey', left=np.sum(data[:j]), label=transmissions[j])

        plt.vlines(x=Male_ratio[i],ymin=y_pos-0.4, ymax=y_pos+0.4, color='blue', linestyle='-', linewidth=2, label='Male Ratio')
        plt.text(Male_ratio[i]+0.5, y_pos, str(Male_ratio[i]), color='blue', va='center', ha='left', fontweight='bold')
        y_pos +=1


    # Customize the plot
    # plt.yticks(np.arange(len(locations)) * (bar_height + bar_gaps) + (len(transmissions) - 1) * (bar_height + bar_gaps) / 2,
    #            locations)
    # Adjust the space on the left side of the subplots
    plt.subplots_adjust(left=0.25)
    plt.xlabel('Proportion')
    #plt.title('Sexual orientation distribution and male ratio of Mpox cases in different countries and regions',y=1.05)

    # Displaying a custom legend
    custom_legend = [Line2D([0], [0], color=colors_array[i], lw=6) for i in range(3)]+[Line2D([0], [0], color='grey', lw=6) ]
    # Add MSM legend with a red border
    custom_legend_MSM = Line2D([0], [0], color=(colors_array[0]+colors_array[1])/2, lw=6,label="MSM")
    custom_legend_Other = Line2D([0], [0], color=(colors_array[2]+colors_array[3])/2,lw=6, label=transmissions[1],alpha=.5)
    custom_legend = custom_legend + [custom_legend_MSM, custom_legend_Other]
    custom_legend_male_ratio = Line2D([0, 0], [1, 1], color='blue', linestyle='-', linewidth=2, label='Male Ratio')
    # custom_legend_male_ratio.set_color("blue")
    custom_legend.append(custom_legend_male_ratio)


    transmissions = ['Homosexual males', 'Bisexual males', 'Heterosexual males and females',
                     'Homosexual females', 'MSM', 'Other transmission', "Male ratio"]

    plt.legend(custom_legend, transmissions, bbox_to_anchor=(-0.3, 1.02), loc='center left', frameon=False, ncol=7)
    plt.xticks(np.arange(0, 101, step=10),[str(i) + '%' for i in np.arange(0, 101, step=10)] )

    plt.gca().invert_yaxis()
    plt.savefig(f"../results/Mpox/Sex_orientation.svg", format='svg', dpi=1200)
    # Show the plot
    plt.show()


def region_male():
    """
    this function will plot male ratio in different regions by date
    plot dot line plot
    """
    global_dates = [1, 8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85, 92, 99, 106, 113, 120, 127, 134, 141, 148, 155, 162, 169, 176, 183, 190, 197, 204, 211, 218, 225, 232, 239, 246, 253, 260, 267, 274, 281, 288, 295, 302, 309, 316, 323, 330, 337, 344, 351, 358, 365, 372, 379, 386, 393, 400, 407, 414, 421, 428, 435, 442, 449, 456, 463, 470, 477]

    global_male_ratio = [
    0.982, 0.990, 0.983, 0.983, 0.994, 0.989, 0.988, 0.989, 0.989, 0.980,
    0.974, 0.972, 0.961, 0.963, 0.949, 0.959, 0.952, 0.959, 0.961, 0.954,
    0.937, 0.940, 0.937, 0.947, 0.924, 0.933, 0.920, 0.871, 0.920, 0.943,
    0.932, 0.942, 0.946, 0.843, 0.849, 0.857, 0.865, 0.909, 0.947, 0.838,
    0.874, 0.977, 0.963, 0.944, 0.971, 0.951, 0.953, 0.981, 0.984, 1.000,
    0.966, 0.981, 0.943, 0.988, 1.000, 0.957, 0.988, 0.987, 1.000, 1.000,
    0.994, 1.000, 0.994, 1.000, 1.000, 1.000, 0.983, 1.000, 1.000,
]
    us_dates=[0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161, 169.5, 182, 220.5, 273, 329, 392, 476.5]

    us_male_ratio=[
    0.750, 1.000, 1.000, 0.971, 1.000, 0.993, 0.983, 0.997, 0.993, 0.992,
    0.993, 0.988, 0.985, 0.978, 0.969, 0.965, 0.964, 0.955, 0.950, 0.957,
    0.948, 0.937, 0.925, 0.909, 0.873, 0.921, 0.946, 0.923, 0.968, 0.979,
    0.978
]
    uk_dates=[17.5,43,50,57.5,64.5,71.5,78.5,85.5,96,110.5,123,136.5,152,190.5]
    uk_male_ratio=[0.993, 1.000, 0.996, 0.989, 0.989, 0.986, 0.984, 0.961, 0.978, 0.980,0.960, 0.985, 0.971, 0.952]
    china_dates=[400,430.5,461.5,492,522.5]
    china_male_ratio=[1, 1, 0.988, 0.990, 0.984]

    # Plot the figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the global male ratio
    ax.plot(global_dates, global_male_ratio, marker='o', label='Global', linestyle='-', color='blue')

    # Plot the US male ratio
    ax.plot(us_dates, us_male_ratio, marker='o', label='United States', linestyle='-', color='green')

    # Plot the UK male ratio
    ax.plot(uk_dates, uk_male_ratio, marker='o', label='England, United Kingdom', linestyle='-', color='orange')

    # Plot the China male ratio
    ax.plot(china_dates, china_male_ratio, marker='o', label='China', linestyle='-', color='red')

    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Male ratio')
    # Set the y axis ticks as %


    #ax.set_title('Male Ratio in Different Regions Over Time')
    plt.legend(frameon=False, loc='lower left', fontsize='large')
    # Format the axes
    format_ax(ax)
    ax.set_yticks(np.arange(0, 1.01, step=0.1) )
    ax.set_yticklabels([str(i) + '%' for i in np.arange(0, 101, step=10)])

    plt.ylim(0,1.1)
    # Display the plot
    plt.tight_layout()
    plt.savefig(f"../results/Mpox/Region_male_ratio.svg", format='svg', dpi=1200)
    plt.show()




def region_MSM():
    """
    this function will plot MSM ratio in different regions by date
    plot dot line plot
    """
    global_dates = [1, 8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85, 92, 99, 106, 113, 120, 127, 134, 141, 148, 155, 162, 169, 176, 183, 190, 197, 204, 211, 218, 225, 232, 239, 246, 253, 260, 267, 274, 281, 288, 295, 302, 309, 316, 323, 330, 337, 344, 351, 358, 365, 372, 379, 386, 393, 400, 407, 414, 421, 428, 435, 442, 449, 456, 463, 470]

    global_MSM_ratio =  [
    0.995, 0.971, 0.982, 0.953, 0.990, 0.987, 0.979, 0.958, 0.965, 0.947,
    0.921, 0.916, 0.893, 0.852, 0.826, 0.851, 0.828, 0.803, 0.821, 0.827,
    0.820, 0.793, 0.769, 0.800, 0.745, 0.748, 0.745, 0.745, 0.730, 0.780,
    0.721, 0.727, 0.717, 0.747, 0.720, 0.762, 0.741, 0.793, 0.817, 0.838,
    0.686, 0.875, 0.863, 0.800, 0.889, 0.750, 0.813, 0.929, 0.956, 0.929,
    0.939, 0.900, 0.953, 0.974, 1.000, 0.955, 0.944, 0.924, 0.927, 0.967,
    0.987, 0.994, 0.985, 0.980, 1.000, 0.971, 0.978, 0.968
]
    uk_dates=[11.5, 35, 49, 75.5, 89, 103, 120]
    uk_MSM_ratio=[0.993, 0.929, 0.987, 0.892, 1.000, 0.955, 0.984]
    china_dates=[400,430.5,461.5,492,522.5]
    china_MSM_ratio=[1.000, 0.963, 0.925, 0.929, 0.900]
    #plot the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    # Plot the global MSM ratio
    ax.plot(global_dates, global_MSM_ratio, marker='o', label='Global', linestyle='-', color='blue')
    # Plot the UK MSM ratio
    ax.plot(uk_dates, uk_MSM_ratio, marker='o', label='England, United Kingdom', linestyle='-', color='orange')
    # Plot the China MSM ratio
    ax.plot(china_dates, china_MSM_ratio, marker='o', label='China', linestyle='-', color='red')
    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('MSM ratio')
    #ax.set_title('MSM Ratio in Different Regions Over Time')
    plt.legend(frameon=False, loc='lower left', fontsize='large')
    # Format the axes
    format_ax(ax)
    ax.set_yticks(np.arange(0, 1.01, step=0.1) )
    ax.set_yticklabels([str(i) + '%' for i in np.arange(0, 101, step=10)])
    plt.ylim(0,1.1)
    # Display the plot
    plt.tight_layout()
    plt.savefig(f"../results/Mpox/Region_MSM_ratio.svg", format='svg', dpi=1200)
    plt.show()



def us_trans():
    """
    this function will plot population segementation by sexual orientation ratio in Amrica by date
    plot stacked bar pplot
    """
    colors_array = np.array([[132, 60, 12], [197, 90, 17], [248, 203, 173], [251, 229, 214]]) / 255
    us_date=[0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112, 119, 126, 133, 140, 147, 154, 161]
    Including_men_other_genders = [0.000, 9.090, 0.000, 11.770, 10.770, 9.930, 7.710, 3.400, 2.660, 1.480, 2.110, 1.850, 2.240, 2.370, 2.360, 2.140, 2.430, 2.840, 2.600, 4.440, 2.230, 1.370, 2.640, 4.660]
    No_recent_sexual_partners=[50.00, 9.09, 33.33, 5.89, 3.08, 6.11, 6.26, 8.01, 10.89, 12.72, 17.66, 18.13, 19.40, 21.11, 20.98, 23.07, 22.23, 21.33, 22.44, 19.32, 24.48, 18.29, 17.62, 11.66]
    Exclusively_men=[25.00, 81.82, 66.67, 79.48, 86.15, 83.26, 82.87, 87.06, 83.98, 83.25, 77.11, 75.70, 72.89, 69.67, 68.18, 63.62, 65.32, 62.79, 61.14, 63.72, 60.36, 64.00, 66.96, 62.94]
    Exclusively_women_other_genders=[0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.45, 1.20, 1.77, 1.79, 2.39, 3.11, 3.96, 4.68, 5.37, 7.72, 6.46, 8.51, 8.78, 8.22, 7.79, 10.06, 5.29, 11.66]
    Women=[25.00, 0.00, 0.00, 2.86, 0.00, 0.69, 1.72, 0.33, 0.69, 0.76, 0.73, 1.21, 1.51, 2.16, 3.11, 3.45, 3.56, 4.54, 5.04, 4.30, 5.16, 6.28, 7.49, 9.09]
    # Convert percentages to proportions
    total = np.array(Including_men_other_genders) + np.array(No_recent_sexual_partners) + np.array(Exclusively_men) + np.array(Exclusively_women_other_genders) + np.array(Women)
    Including_men_other_genders = np.array(Including_men_other_genders) / total
    No_recent_sexual_partners = np.array(No_recent_sexual_partners) / total
    Exclusively_men = np.array(Exclusively_men) / total
    Exclusively_women_other_genders = np.array(Exclusively_women_other_genders) / total
    Women = np.array(Women) / total
    #plot the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    # Plot the stacked bars
    ax.bar(us_date, Exclusively_men, color=colors_array[0], label='Male, Recent partners exclusively men', width=5)
    ax.bar(us_date, Including_men_other_genders, bottom=Exclusively_men,color=colors_array[1], label='Male, Recent partners include men and other genders',width=5)
    ax.bar(us_date, Exclusively_women_other_genders,bottom=Exclusively_men + Including_men_other_genders, color=colors_array[2],label='Male, Recent partners exclusively women and other genders', width=5)
    ax.bar(us_date, No_recent_sexual_partners, bottom=Exclusively_men + Including_men_other_genders+Exclusively_women_other_genders, color='grey', label='Male, No recent sexual partners',width=5)
    ax.bar(us_date, Women, bottom=Exclusively_women_other_genders + Exclusively_men + No_recent_sexual_partners + Including_men_other_genders, color='blue', label='Female',width=5)

    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Proportion')
    plt.legend(loc='lower right',frameon=True, fontsize='large')

    # Format the axes
    format_ax(ax, start=-5, end=165)
    ax.set_yticks(np.arange(0, 1.01, step=0.1))
    ax.set_yticklabels([str(i) + '%' for i in np.arange(0, 101, step=10)])
    plt.ylim(0, 1.1)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"../results/Mpox/us_trans_stacked_bar.svg", format='svg', dpi=1200)
    plt.show()



# # Call the function to create the scatter plot
# male_ratio()
# # Call the function to create the scatter plot
# MSM_ratio()
# # Call the function to plot the transmission data
plot_trans()
# Call the function
region_male()
# Call the function
region_MSM()
# Call the function
us_trans()
