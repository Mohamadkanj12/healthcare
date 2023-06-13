# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 18:37:27 2023

@author: student
"""


# import libraries
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Navigation bar
    navigation = st.sidebar.radio("Navigation", ["Home", "Incidences","Mortality","Risk Factors", "Contact"])
    
    # Handle navigation
    if navigation == "Home":
        home_page()
    elif navigation == "Incidences":
        incidences()
    elif navigation == "Mortality":
        mortality()
    elif navigation == "Risk Factors":
        Risk_Factors()
    elif navigation == "Contact":
        contact_page()

def home_page():
    st.image("./AUB-logo.png", width=150)
    st.markdown("<h1 style='text-align: center; font-size: 28px;'>Fighting Cancer: Cancer Statistics Dashboard for Pennsylvania</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 16px;'>In Memory of Kate from 'My Sister's Keeper' and Every Other Fighter in This World</p>", unsafe_allow_html=True)

    st.markdown("##### Incidences", unsafe_allow_html=True)
    incidences_data = {
        "Number of New Cases in 2022": 85110,
        "The AVG Number of New cases in other States": 38360,
        "Incidence rates, 2015-2019": 476.8,
        "Incidence rates, 2022": 656.71,
    }
    display_statistics(incidences_data)

    st.markdown("##### Mortality", unsafe_allow_html=True)
    mortality_data = {
        "Number of Deaths": 27260,
        "The AVG Number of Deaths in other states": 12187,
    }
    display_statistics(mortality_data)

def display_statistics(data):
    cols = st.columns(2)  # Divide the screen into 2 columns
    items = list(data.items())
    num_items = len(items)
    max_height = max([len(title) for title, _ in items])  # Get the maximum title length
    for i in range(num_items):
        with cols[i % 2]:
            box_height = max_height + 3  # Adjust the height based on the maximum title length
            fill_color = get_fill_color(i+1)  # Get the fill color based on the box index
            box_style = f"height: {box_height * 0.1}rem; background-color: {fill_color};"  # Set the box height and fill color using CSS
            st.markdown(f'<div style="{box_style}">**{items[i][0]}:** {items[i][1]}</div>', unsafe_allow_html=True)
            
            if i == 0 or i == 1:  # Check if the current index is 0 (Box 1) or 1 (Box 2)
                st.markdown("<br>", unsafe_allow_html=True)  # Add an empty line using HTML <br> tag

# Fill color for boxes
def get_fill_color(box_index):
    if box_index in [1, 2]:
        return "lightblue"  # Light red for box 1 and 2
    elif box_index in [3, 4]:
        return "lightgray"  # Light blue for box 2 and 3
    elif box_index in [1, 2]:
        return "lightgray"  # Light gray for box 5 and 6
    else:
        return "white"  # Default color for other boxes






def incidences():
    st.title("Incidences")

    # Import the Excel files
    Estimateddeathes = pd.read_csv(r'data_projection.csv')
    df = Estimateddeathes

    # Filter Title
    st.sidebar.title('Cancer Incidences in Pennsylvania by Type')

    # Filter options
    year_options = df['year'].unique().tolist()
    selected_year = st.sidebar.selectbox('Select a year', year_options)
    sex_options = df['sex'].unique().tolist()
    selected_sex = st.sidebar.selectbox('Select a sex', sex_options)
    
    # Filter the dataset based on selected year and sex
    filtered_df = df[(df['year'] == selected_year) & (df['sex'] == selected_sex)]
    
    # Sort the dataset based on inc_count or death_count
    selected_count = st.selectbox('Select count type', ['inc_count', 'death_count'])
    sorted_df = filtered_df.sort_values(by=selected_count, ascending=False)
    
    # Reverse the order of the sorted_df to make the chart appear from top to lowest
    sorted_df = sorted_df.iloc[::-1]
    
    # Create the horizontal bar chart
    fig = go.Figure()
    
    # Assign color to top 3 bars as black, rest as gray
    colors = ['lightblue' if i > 13 else 'lightgray' for i in range(len(sorted_df))]
    
    fig.add_trace(go.Bar(
        y=sorted_df['cancer'],
        x=sorted_df[selected_count],
        orientation='h',
        marker=dict(color=colors),
        text=sorted_df[selected_count],
        textposition='auto',
        texttemplate='%{text:.2s}'
    ))
    
    # Define the title and graph title
    title = "<b>Cancer Incidence in Pennsylvania by Type</b>"

    
    # Add the title and graph title to the layout
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=20)),
        width=870,
        height=600,
    )

    # Move the x-axis to the top
    fig.update_layout(xaxis=dict(side='top'))
    
    # Adjust spacing between y-axis labels and bars
    fig.update_layout(
        yaxis=dict(tickmode='linear', tick0=0.5, dtick=1)
    )
    
    # Move the x-axis to the top
    fig.update_layout(xaxis=dict(side='top'))
    
    # Adjust spacing between y-axis labels and bars
    fig.update_layout(
        yaxis=dict(tickmode='linear', tick0=0.5, dtick=1)
    )
    
    # Remove gridlines
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig)



    # Import the Excel files
    df2 = pd.read_csv(r'data_median_age.csv')
         
    # Display the title with increased font size
    st.markdown("<h3 style='font-size:20px; font-weight:bold;'>Median Age for Developing Cancer</h3>", unsafe_allow_html=True)

    # Create filters using multiselect widget
    selected_sex = st.selectbox("Select Sex", df2['sex'].unique())
    
    # Filter the DataFrame based on the selected sex
    filtered_median_df = df2[(df2['data_type'] == 'incidence') & (df2['sex'] == selected_sex)]
    
    # Set the color for the bars
    chart_color = 'lightblue'
    
    # Create a horizontal bar chart with custom color using Altair
    if not filtered_median_df.empty:
        chart = alt.Chart(filtered_median_df).mark_bar(color=chart_color).encode(
            x=alt.X('median_age', axis=alt.Axis(title=None, orient='top', grid=False)),
            y=alt.Y('cancer', sort='x', axis=alt.Axis(orient='left', grid=False))
        ).configure_axis(grid=False).configure_view(strokeWidth=0)  # Remove gridlines and borders
        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("No data available for the selected filters.")
    
    
    # Graph 3: How Cancer Changed over Time
    # Import the Excel files
    df3 = pd.read_csv(r'data_joinpoint_aapc.csv')

    # Display the title with increased font size
    st.markdown("<h3 style='font-size:20px; font-weight:bold;'>How Cancer Changed over Time</h3>", unsafe_allow_html=True)

    # Filter the dataframe based on the specified criteria
    filtered_df = df3[(df3['first_year'] == 2015) & (df3['data_type'] == 'incidence')]
    
    # Sidebar filters title
    st.sidebar.title('How Cancer Changed over Time')

    # Sidebar filters
    sex_filter = st.sidebar.selectbox('Select Sex', df3['sex'].unique())
    race_filter = st.sidebar.selectbox('Select Race', df3['race'].unique())
    stage_filter = st.sidebar.selectbox('Select Stage', df3['stage'].unique())
    age_filter = st.sidebar.selectbox('Select Age', df3['age'].unique())
    geography_filter = st.sidebar.selectbox('Select Geography', df3['geography'].unique())
    
    # Apply additional filters based on the selected values
    filtered_df = filtered_df[(filtered_df['sex'] == sex_filter) &
                              (filtered_df['race'] == race_filter) &
                              (filtered_df['stage'] == stage_filter) &
                              (filtered_df['age'] == age_filter) &
                              (filtered_df['geography'] == geography_filter)]
    
    # Sort the filtered DataFrame by AAPC values in descending order
    filtered_df = filtered_df.sort_values('aapc', ascending=True)
    
    # Convert AAPC column to numeric values
    filtered_df['aapc'] = pd.to_numeric(filtered_df['aapc'])
    
    # Create a list of colors for the bars based on positive or negative values
    colors = np.where(filtered_df['aapc'] >= 0, 'lightblue', 'lightgray')
    
    # Remove the borders
    plt.box(False)
    
    # Create the horizontal bar chart with custom colors
    plt.barh(filtered_df['cancer'], filtered_df['aapc'], color=colors)
    plt.xlabel('AAPC')
    plt.ylabel('Cancer')
    plt.title('AAPC by Cancer Type')
    
    # Display the chart using Streamlit
    st.pyplot(plt)

    








    
    
    
def mortality():
    st.title("Mortality & Survival Rates")
    
    # Graph 4: How Cancer Changed over Time
    # Import the Excel files
    df4 = pd.read_csv(r'survival_by_years_since_dx.csv')
    
    # Display the title with increased font size
    st.markdown("<h3 style='font-size:20px; font-weight:bold;'>Net Cancer Survival, by Number of Years Since Diagnosis, by Cancer, Pennsylvania Residents, 2000-2018</h3>", unsafe_allow_html=True)

    
    # Convert 'summary_interval' column from months to years
    df4['summary_interval'] = df4['summary_interval'].apply(lambda x: int(x.split()[0]) / 12)

    # Filter options
    selected_cancers = st.multiselect('Select cancer(s)', df4['cancer'].unique())
    selected_sexes = st.multiselect('Select sex(es)', df4['sex'].unique())
    selected_races = st.multiselect('Select race(s)', df4['race'].unique())
    selected_stages = st.multiselect('Select stage(s)', df4['stage'].unique())
    selected_ages = st.multiselect('Select age(s)', df4['age'].unique())
    
    # Apply filters
    filtered_df = df4[(df4['cancer'].isin(selected_cancers)) &
                      (df4['sex'].isin(selected_sexes)) &
                      (df4['race'].isin(selected_races)) &
                      (df4['stage'].isin(selected_stages)) &
                      (df4['age'].isin(selected_ages))]
    
    # Create a light color palette
    num_colors = len(selected_cancers)
    light_palette = sns.color_palette("pastel", n_colors=num_colors)
    
    # Create the line graph with light colors
    fig, ax = plt.subplots()
    for i, cancer_type in enumerate(selected_cancers):
        ax.plot(filtered_df[filtered_df['cancer'] == cancer_type]['summary_interval'],
                filtered_df[filtered_df['cancer'] == cancer_type]['net_relative'],
                label=cancer_type, color=light_palette[i])
    
    ax.set_xlabel('Years Since Diagnosis')
    ax.set_ylabel('Percent Surviving')
    ax.set_title('Survival Rate Over Time')
    
    # Adjust legend properties
    legend = ax.legend(ncol=2, fontsize='small')
    legend.get_frame().set_linewidth(0)
    
    st.pyplot(fig)
    
    # Display the filtered dataset
    st.write(filtered_df)

    
    
    
    
    
    
    
    # Import the Excel files
    Estimateddeathes = pd.read_csv(r'data_projection.csv')
    df = Estimateddeathes
    
    # Filter Title
    st.sidebar.title('Cancer Deaths in Pennsylvania by Type')
    
    # Filter options
    year_options = df['year'].unique().tolist()
    selected_year = st.sidebar.selectbox('Select a year', year_options, key='year_select')
    sex_options = df['sex'].unique().tolist()
    selected_sex = st.sidebar.selectbox('Select a sex', sex_options, key='sex_select')
    
    # Filter the dataset based on selected year and sex
    filtered_df = df[(df['year'] == selected_year) & (df['sex'] == selected_sex)]
    
    # Sort the dataset based on inc_count or death_count
    selected_count = st.selectbox('Select count type', ['death_count'], key='count_select')
    sorted_df = filtered_df.sort_values(by=selected_count, ascending=False)
    
    # Reverse the order of the sorted_df to make the chart appear from top to lowest
    sorted_df = sorted_df.iloc[::-1]
    
    # Create the horizontal bar chart
    fig = go.Figure()
    
    # Assign color to top 3 bars as black, rest as gray
    colors = ['lightblue' if i > 13 else 'lightgray' for i in range(len(sorted_df))]
    
    fig.add_trace(go.Bar(
        y=sorted_df['cancer'],
        x=sorted_df[selected_count],
        orientation='h',
        marker=dict(color=colors),
        text=sorted_df[selected_count],
        textposition='auto',
        texttemplate='%{text:.2s}'
    ))
    
    # Define the title and graph title
    title = "<b>Cancer Deaths in Pennsylvania by Type</b>"
    
    # Add the title and graph title to the layout
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=20)),
        width=870,
        height=600,
    )
    
    # Move the x-axis to the top
    fig.update_layout(xaxis=dict(side='top'))
    
    # Adjust spacing between y-axis labels and bars
    fig.update_layout(
        yaxis=dict(tickmode='linear', tick0=0.5, dtick=1)
    )
    
    # Remove gridlines
    fig.update_layout(
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig)
    
    
    
    
    
    
def Risk_Factors():
    # Insert empty lines or line breaks before and after the Markdown content
    st.markdown("r>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:black'>CANCER SCREENING AND RISK FACTOR PREVALENCE ONLY FOR THE MOST FREQUENT CANCER TYPES</h3>", unsafe_allow_html=True)

    # Write a description
    st.write("<h3 style='color:#808080; font-size:14px'>The information provided highlights key cancer screening rates, cancer risk factors in Pennsylvania compared to the national average. This data can assist decision-making in the fight against cancer by identifying areas that require improvement or reinforcement. </h3>", unsafe_allow_html=True)
    
    # Read the Excel file
    file_path = r'cancer_screaning.xlsx'
    df1 = pd.read_excel(file_path)
    
    # Create the table with the first 2 rows
    table_data = df1.head(2)
    
    # Sort the table based on the "National Rank" column
    table_data = table_data.sort_values(by='National Rank')
    
    # Display the table without the index column
    st.write(table_data.to_html(index=False, justify='left', escape=False), unsafe_allow_html=True)
    
    # Customize the table style
    table_style = """
    <style>
        table {
            font-family: Arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
            color: black;
        }
    
        th, td {
            padding: 8px;
            text-align: left;
        }
    
        tr:nth-child(even) {
            background-color: lightgray;
        }
    
        td {
            color: black;
            background-color: lightgray; /* Set fill color to lightgray */
        }
    </style>
    """
    st.write(table_style, unsafe_allow_html=True)
    
    # Create the table with the last 10 rows
    table_data = df1.tail(11)
    
    # Exclude the first row from the DataFrame
    table_data = df1.iloc[3:]

    # Remove the index and column titles
    table_data_html = table_data.to_html(index=False, justify='left', escape=False)
    table_data_html = table_data_html.replace('<th>','<th style="font-weight:bold;">')
    
    # Display the modified table without column titles and with the first row as the title
    st.write(table_data_html, unsafe_allow_html=True)
    
    # Customize the table style
    table_style = """
    <style>
        table {
            font-family: Arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
        }
    
        th, td {
            padding: 8px;
            text-align: left;
            width: 150px;  /* Adjust the width as per your requirement */
        }
    
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
    </style>
    """
    st.write(table_style, unsafe_allow_html=True)








def contact_page():
    st.title("Contact")
    st.write("This is the Contact page.")

def main():
    # Navigation bar
    navigation = st.sidebar.radio("Navigation", ["Home", "Incidences","Mortality & Survival Rates","Risk Factors", "Contact"])
    
    # Handle navigation
    if navigation == "Home":
        home_page()
    elif navigation == "Incidences":
        incidences()
    elif navigation == "Mortality & Survival Rates":
        mortality()
    elif navigation == "Risk Factors":
        Risk_Factors()
    elif navigation == "Contact":
        contact_page()

if __name__ == '__main__':
    main()
