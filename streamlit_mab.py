import pandas as pd
import numpy as np
import streamlit as st
import mab
import sys
import itertools
import altair as alt
import time

##################################################
## Parameters:                                  ##
## Use Max vs. Prob assignment                  ##
## Show distribution of campaign assignment     ##
## Learning rate (between 0.5 and 1.5)          ##
##################################################

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)

st.markdown("<h2 style='text-align: center; color: grey;'>Multi-Armed Bandit Campaign Simulator</h2>", unsafe_allow_html=True)
st.markdown(":gray[placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text placeholder text]")


with st.sidebar:
    st.radio(
        "Select Optimization Method 👉",
        options=["Best Variant Assignment", "Probabilistic Assignment"],
        captions=[
        "***Assigns the variant with the highest probability (optimizes exploitation)***",
        "***Assigns variants proportional to expected performance (sacrifices exploitation for more exploration)***"]
    )

    st.divider()

    learn_rate = st.slider("Optimization Rate", 0, 100, 50, label_visibility="visible", help='The higher the value, the faster the optimization occurs, but the chances of truly optimizing is lower')
    learn_rate = (learn_rate + 50) / 100

    st.divider()

    st.write("Display Output")

    data_df = pd.DataFrame(
        {
            "Show": ["Performance by Variant", "Uplift by Variant", "Variant Assignment by Cohort"],
            "Include": [True, False, False],
        }
    )

    st.data_editor(
        data_df,
        column_config={
            "Include": st.column_config.CheckboxColumn(
                "Include",
                help="Check if you want to see this output",
                default=False,
            )
        },
        hide_index=True,
    )

    st.divider()



## Parameters for simulation
steps = 50
include_variant_chart = True
include_variant_uplift = True
include_cohort_tables = True

if include_cohort_tables:
    tab1, tab2 = st.tabs(["Optimization Charts", "Variant Assignment"])
    with tab2:
        placeholder = st.empty()
else:
    tab1 = st.tabs(["Charts"])

if st.button('Run Simulation'):
    with st.spinner('Running Simulation...'):

        row_count = 100000
        seg_cols = ['gender', 'age',
                    'income', 'buyer',
                    'region', 'area',
                    'parent']

        segments = ["Male", "Female",
                    "Young", "Middle Age", "Older",
                    "Low Income", "Medium Income", "High Income",
                    "Prior Buyer", "First-Time Buyer",
                    "North", "West", "South", "East",
                    "Urban", "Suburban",
                    "Non-Parent", "Parent"]

        combo_weights = mab.create_all_combo_weights()
        Segment_df = mab.create_synthetic_sample(row_count=row_count)

        ## Full random assignment for Variant
        Segment_df = mab.add_conversion_rates(df=Segment_df,
                                              seg_cols=seg_cols,
                                              segments=segments,
                                              all_combos_weights=combo_weights,
                                              print_diagnostics=False)

        for i in range(steps):

            if i == 0:

                Segment_df = mab.create_synthetic_sample(row_count=row_count)
                Segment_df = mab.add_conversion_rates(df=Segment_df, seg_cols=seg_cols, segments=segments, all_combos_weights=combo_weights, print_diagnostics=False)

                ## Store results
                overall_performance = Segment_df['converted'].mean()
                org_target_performance = Segment_df.loc[Segment_df['variant_assignment'] != 'Control', 'converted'].mean()
                overall_target_performance = Segment_df.loc[Segment_df['variant_assignment'] != 'Control', 'converted'].mean()

                ## Adding overall target performance
                overall_target_performance_variant_a = Segment_df.loc[(Segment_df['variant_assignment'] != 'Control') & (Segment_df['variant_assignment'] == 'Variant A'), 'converted'].mean()
                overall_target_performance_variant_b = Segment_df.loc[(Segment_df['variant_assignment'] != 'Control') & (Segment_df['variant_assignment'] == 'Variant B'), 'converted'].mean()
                overall_target_performance_variant_c = Segment_df.loc[(Segment_df['variant_assignment'] != 'Control') & (Segment_df['variant_assignment'] == 'Variant C'), 'converted'].mean()

                ## Adding organic performance only by variant
                organic_target_performance_variant_a = Segment_df.loc[(Segment_df['variant_assignment'] != 'Control') & (Segment_df['variant_assignment'] == 'Variant A'), 'converted'].mean()
                organic_target_performance_variant_b = Segment_df.loc[(Segment_df['variant_assignment'] != 'Control') & (Segment_df['variant_assignment'] == 'Variant B'), 'converted'].mean()
                organic_target_performance_variant_c = Segment_df.loc[(Segment_df['variant_assignment'] != 'Control') & (Segment_df['variant_assignment'] == 'Variant C'), 'converted'].mean()

                ## Uplift
                variant_a_uplift = overall_target_performance_variant_a - organic_target_performance_variant_a
                variant_b_uplift = overall_target_performance_variant_b - organic_target_performance_variant_b
                variant_c_uplift = overall_target_performance_variant_c - organic_target_performance_variant_c

                ##################
                ## INSERT CHART ##
                ##################

                with tab1:

                    # Set up data
                    chart_data = pd.DataFrame({'Metric': ['Organic Target Only','Organic + Optimized'],
                                               'itr': [1,1],
                                               'Value': [org_target_performance, overall_target_performance]})

                    line_chart = alt.Chart(chart_data, title="Performance by Target Audience").mark_line(
                                    point=alt.OverlayMarkDef(filled=True, size=15)
                                    ).encode(
                                        alt.X('itr:N', scale=alt.Scale(domain=list(range(0,51))), title="Round"),
                                        alt.Y('Value:Q',scale=alt.Scale(domainMin=0.20), title = 'Performance').axis(format='%'),
                                        alt.Color('Metric:N',
                                                 legend=alt.Legend(title="Targeted Audience", titleFontSize=16))
                                    ).properties(
                                        height=300
                                    ).configure_title(fontSize=16, orient='top', anchor='middle'
                                    ).interactive()

                    my_chart = st.altair_chart(line_chart, use_container_width=True)

                    if include_variant_chart:

                        chart_data_variant = pd.DataFrame({'Metric': ['Variant A', 'Variant A', 'Variant B', 'Variant B', 'Variant C', 'Variant C'],
                                                           'Type':['Organic', 'Organic + Optimized', 'Organic', 'Organic + Optimized','Organic', 'Organic + Optimized'],
                                                           'itr': [1,1,1,1,1,1],
                                                           'Value': [organic_target_performance_variant_a, overall_target_performance_variant_a,
                                                                     organic_target_performance_variant_b, overall_target_performance_variant_b,
                                                                     organic_target_performance_variant_c, overall_target_performance_variant_c]})

                        st.divider()

                        line_chart_variant = alt.Chart(chart_data_variant).mark_line(
                                                ).encode(
                                                    alt.X('itr:N', scale=alt.Scale(domain=list(range(0,51))), title="Round"),
                                                    alt.Y('Value:Q',scale=alt.Scale(domainMin=0.20), title = 'Performance').axis(format='%'),
                                                    alt.Color('Metric:N',
                                                             legend=alt.Legend(title="Variant", titleFontSize=16)),
                                                    alt.StrokeDash('Type:N',
                                                                   legend=alt.Legend(title="Optimized", titleFontSize=16)
                                                        )
                                                ).properties(
                                                    height=300,
                                                    title={
                                                        'text': ["Performance by Variant"]
                                                    }
                                                ).configure_title(fontSize=16, orient='top', anchor='middle'
                                                ).interactive()

                        my_chart_variant = st.altair_chart(line_chart_variant, use_container_width=True)

                    if include_variant_uplift:

                        st.divider()
                        chart_data_uplift = pd.DataFrame({'Metric': ['Variant A                   ‎ ', 'Variant B', 'Variant C'],
                                                           'itr': [1,1,1],
                                                           'Value': [variant_a_uplift, variant_b_uplift, variant_c_uplift]})

                        line_chart_uplift = alt.Chart(chart_data_uplift).mark_line(
                                                    ).encode(
                                                    alt.X('itr:N', scale=alt.Scale(domain=list(range(0,51))), title="Round"),
                                                    alt.Y('Value:Q',scale=alt.Scale(domainMin=0), title = 'Performance').axis(format='+%'),
                                                    alt.Color('Metric:N',
                                                             legend=alt.Legend(title="Variant", titleFontSize=16))
                                                    ).properties(
                                                    height=300,
                                                    title={
                                                        'text': ["Optimization Uplift"]
                                                    }
                                                    ).configure_title(fontSize=16, orient='top', anchor='middle'
                                                    ).interactive()

                        my_chart_uplift = st.altair_chart(line_chart_uplift, use_container_width=True)

                    ### For Next Iteration ###

                    # Take the Target group and build an optimization score to determine how ads should be allocated
                    Segment_df.loc[Segment_df['variant_assignment'] == 'Variant A', 'Variant_a_performance'] = Segment_df['converted']
                    Segment_df.loc[Segment_df['variant_assignment'] == 'Variant B', 'Variant_b_performance'] = Segment_df['converted']
                    Segment_df.loc[Segment_df['variant_assignment'] == 'Variant C', 'Variant_c_performance'] = Segment_df['converted']

                    ## Performance Scores all interactions
                    perf_scores_all_interactions = Segment_df.groupby(seg_cols).agg({'Variant_a_performance': ['mean'],
                                                                                     'Variant_b_performance': ['mean'],
                                                                                     'Variant_c_performance': ['mean']}).reset_index().droplevel(1, axis = 1)

            if i > 0:

                Segment_df_step2 = mab.create_synthetic_sample(row_count=row_count)
                Segment_df_step2 = mab.assignment_with_optimization(df=Segment_df_step2, prior_performance_scores=perf_scores_all_interactions,seg_cols=seg_cols,method='max', opt_target_size=i/100, learning_weight=2)
                Segment_df_step2 = mab.add_conversion_rates(df=Segment_df_step2, seg_cols=seg_cols, segments=segments, all_combos_weights=combo_weights, print_diagnostics=False, assign_variant=False)
                Segment_df_step2 = Segment_df_step2.reset_index(drop=True)

                ## Store Results
                overall_performance = Segment_df_step2['converted'].mean()
                overall_target_performance = Segment_df_step2.loc[Segment_df_step2['target_control'] != 'control', 'converted'].mean()
                org_target_performance = Segment_df_step2.loc[Segment_df_step2['target_control'] == 'target_org', 'converted'].mean()
                opt_target_performance = Segment_df_step2.loc[Segment_df_step2['target_control'] == 'target_opt', 'converted'].mean()

                ## Adding overall target performance
                overall_target_performance_variant_a = Segment_df_step2.loc[(Segment_df_step2['target_control'] != 'Control') & (Segment_df_step2['variant_assignment'] == 'Variant A'), 'converted'].mean()
                overall_target_performance_variant_b = Segment_df_step2.loc[(Segment_df_step2['target_control'] != 'Control') & (Segment_df_step2['variant_assignment'] == 'Variant B'), 'converted'].mean()
                overall_target_performance_variant_c = Segment_df_step2.loc[(Segment_df_step2['target_control'] != 'Control') & (Segment_df_step2['variant_assignment'] == 'Variant C'), 'converted'].mean()

                ## Adding organic performance only by variant
                organic_target_performance_variant_a = Segment_df_step2.loc[(Segment_df_step2['target_control'] == 'target_org') & (Segment_df_step2['variant_assignment'] == 'Variant A'), 'converted'].mean()
                organic_target_performance_variant_b = Segment_df_step2.loc[(Segment_df_step2['target_control'] == 'target_org') & (Segment_df_step2['variant_assignment'] == 'Variant B'), 'converted'].mean()
                organic_target_performance_variant_c = Segment_df_step2.loc[(Segment_df_step2['target_control'] == 'target_org') & (Segment_df_step2['variant_assignment'] == 'Variant C'), 'converted'].mean()

                ## Uplift
                variant_a_uplift = overall_target_performance_variant_a - organic_target_performance_variant_a
                variant_b_uplift = overall_target_performance_variant_b - organic_target_performance_variant_b
                variant_c_uplift = overall_target_performance_variant_c - organic_target_performance_variant_c

                ## For Next Iteration ##

                # Take the Target group and build an optimization score to determine how ads should be allocated
                Segment_df_step2.loc[Segment_df_step2['variant_assignment'] == 'Variant A', 'Variant_a_performance'] = Segment_df_step2['converted']
                Segment_df_step2.loc[Segment_df_step2['variant_assignment'] == 'Variant B', 'Variant_b_performance'] = Segment_df_step2['converted']
                Segment_df_step2.loc[Segment_df_step2['variant_assignment'] == 'Variant C', 'Variant_c_performance'] = Segment_df_step2['converted']

                ## Performance Scores all interactions
                perf_scores_all_interactions = Segment_df_step2[Segment_df_step2['target_control'] == 'target_org'].groupby(seg_cols).agg({'Variant_a_performance': ['mean'],
                                                                                                                                           'Variant_b_performance': ['mean'],
                                                                                                                                           'Variant_c_performance': ['mean']}).reset_index().droplevel(1, axis = 1)
                new_data = pd.DataFrame({'Metric': ['Organic Target Only','Organic + Optimized'],
                                         'itr': [i+1,i+1],
                                         'Value': [org_target_performance, overall_target_performance]})

                my_chart.add_rows(new_data)

                if include_variant_chart:
                    new_chart_data_variant = pd.DataFrame({'Metric': ['Variant A', 'Variant A', 'Variant B', 'Variant B', 'Variant C', 'Variant C'],
                                                           'Type':['Organic', 'Organic + Optimized', 'Organic', 'Organic + Optimized','Organic', 'Organic + Optimized'],
                                                           'itr': [i+1,i+1,i+1,i+1,i+1,i+1],
                                                           'Value': [organic_target_performance_variant_a, overall_target_performance_variant_a,
                                                                     organic_target_performance_variant_b, overall_target_performance_variant_b,
                                                                     organic_target_performance_variant_c, overall_target_performance_variant_c]})

                    my_chart_variant.add_rows(new_chart_data_variant)

                if include_variant_uplift:
                    new_chart_data_uplift = pd.DataFrame({'Metric': ['Variant A                   ‎ ', 'Variant B', 'Variant C'],
                                                          'itr': [i+1,i+1,i+1],
                                                          'Value': [variant_a_uplift, variant_b_uplift, variant_c_uplift]})

                    my_chart_uplift.add_rows(new_chart_data_uplift)

                with tab2:

                    if include_cohort_tables:

                        org_table = mab.get_variant_assignment_counts(df = Segment_df_step2[Segment_df_step2['target_control'] != 'target_org'], table_name='Optimized', seg_cols=seg_cols)
                        opt_table = mab.get_variant_assignment_counts(df = Segment_df_step2[Segment_df_step2['target_control'] != 'target_opt'], table_name='Organic', seg_cols=seg_cols)
                        curr_table = pd.concat([org_table, opt_table], axis =1 )

                        with placeholder.container():
                            curr_message = "On iteration " + str(i+1) + " out of " + str(steps)
                            st.write(curr_message)
                            st.dataframe(curr_table,
                                         height=650,
                                         use_container_width=True,
                                         column_config = {
                                             "Optimized": st.column_config.NumberColumn(
                                                 "Optimized,
                                                 help="The price of the product in USD"
                                                 format="$%d"
                                             )
                                         )
