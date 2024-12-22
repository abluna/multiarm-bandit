import pandas as pd
import numpy as np
import streamlit as st
import mab
import sys
import itertools
import altair as alt

import warnings
warnings.simplefilter("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)

st.write("""
# Multi-Arm Bandit Campaign Simulator
""")

test_message = mab.mab_test()
if st.button('Test App'):
    with st.spinner('Wait for it...'):
        
        for i in [1,2,3,4]:
            # Create a categorical column with 3 categories
            categories = ['Category A', 'Category B', 'Category C']
            categorical_column = np.random.choice(categories, size=50)

            # Create 4 numeric columns with random numbers
            ind = list(range(50))
            numeric_column_1 = np.random.rand(50) * 100 + 100  # Random numbers between 0 and 100
            numeric_column_2 = np.random.rand(50) * 50 + 100   # Random numbers between 0 and 50
            numeric_column_3 = np.random.rand(50) * 10 + 100    # Random numbers between 0 and 10
            numeric_column_4 = np.random.rand(50) * 500 + 100   # Random numbers between 0 and 500

            # Create a DataFrame
            data = pd.DataFrame({
                'Category': categorical_column,
                'ind':ind,
                'Numeric_1': numeric_column_1,
                'Numeric_2': numeric_column_2,
                'Numeric_3': numeric_column_3,
                'Numeric_4': numeric_column_4
            })

            data_long = pd.wide_to_long(data, stubnames='Numeric_', i=['Category', 'ind'], j='new_ind').reset_index()
            
            if i == 1:
            
                data_long_chart1 = data_long[data_long["new_ind"]==i].copy().reset_index()
                line_chart = alt.Chart(data_long_chart1).mark_line().encode(
                        alt.X('ind:N', scale=alt.Scale(domain=list(range(100))), title="Round"),
                        alt.Y('Numeric_:Q', scale=alt.Scale(domainMin=100), title = 'Value'),
                        alt.Color('new_ind:N',
                                 legend=alt.Legend(title="Poop"))
                    ).properties(
                        height=200
                    ).interactive()
                
                my_chart = st.altair_chart(line_chart, use_container_width=True)
                
            if i > 1:
                
                new_data = data_long[data_long["new_ind"]==i].copy().reset_index()
                my_chart.add_rows(new_data)
            

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

if st.button('Run Performance Simulation'):
    with st.spinner('Running Simulation...'):

        combo_weights = mab.create_all_combo_weights()
        Segment_df = mab.create_synthetic_sample(row_count=row_count)

        ## Full random assignment for Variant
        Segment_df = mab.add_conversion_rates(df=Segment_df, 
                                              seg_cols=seg_cols, 
                                              segments=segments, 
                                              all_combos_weights=combo_weights,
                                              print_diagnostics=False)


        schema = {'ind_num':pd.Series(dtype='int'),
                  'Overall Performance':pd.Series(dtype='float'), 
                  'All Target Performance':pd.Series(dtype='float'), 
                  'Organic Target Performance':pd.Series(dtype='float'), 
                  'Organic Variant A Performance':pd.Series(dtype='float'),
                  'All Variant A Performance':pd.Series(dtype='float'),
                  'Organic Variant B Performance':pd.Series(dtype='float'),
                  'All Variant B Performance':pd.Series(dtype='float'),
                  'Organic Variant C Performance':pd.Series(dtype='float'),
                  'All Variant C Performance':pd.Series(dtype='float')}
                  
              
                                              
        df_to_display = pd.DataFrame(schema)
                                
        steps = 10
        row_count = 100000

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


                ## For Next Iteration ##
                
                # Take the Target group and build an optimization score to determine how ads should be allocated
                Segment_df_step2.loc[Segment_df_step2['variant_assignment'] == 'Variant A', 'Variant_a_performance'] = Segment_df_step2['converted']
                Segment_df_step2.loc[Segment_df_step2['variant_assignment'] == 'Variant B', 'Variant_b_performance'] = Segment_df_step2['converted']
                Segment_df_step2.loc[Segment_df_step2['variant_assignment'] == 'Variant C', 'Variant_c_performance'] = Segment_df_step2['converted']
                
                ## Performance Scores all interactions
                perf_scores_all_interactions = Segment_df_step2[Segment_df_step2['target_control'] == 'target_org'].groupby(seg_cols).agg({'Variant_a_performance': ['mean'],
                                                                                                                                           'Variant_b_performance': ['mean'],
                                                                                                                                           'Variant_c_performance': ['mean']}).reset_index().droplevel(1, axis = 1)

            curr_table = pd.DataFrame([{'ind_num':i+1,
                                        'Overall Performance':overall_performance, 
                                        'All Target Performance':overall_target_performance, 
                                        'Organic Target Performance':org_target_performance,
                                        'Organic Variant A Performance':organic_target_performance_variant_a,
                                        'All Variant A Performance':overall_target_performance_variant_a,
                                        'Organic Variant B Performance':organic_target_performance_variant_b,
                                        'All Variant B Performance':overall_target_performance_variant_b,
                                        'Organic Variant C Performance':organic_target_performance_variant_c,
                                        'All Variant C Performance':overall_target_performance_variant_c}])
            

        