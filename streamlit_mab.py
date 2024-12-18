import pandas as pd
import numpy as np
import streamlit as st
import mab
import sys
import itertools

st.write("""
# Multi-Arm Bandit Campaign Simulator
""")

test_message = mab.mab_test()
if st.button('Test App'):
    with st.spinner('Wait for it...'):
        st.write(test_message)


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
                                              
        df_to_display = pd.DataFrame(columns=['Iteration', 
                                              'Overall Performance', 
                                              'Organic Target Performance', 
                                              'Organic Variant A Performance',
                                              'All Variant A Performance',
                                              'Organic Variant B Performance',
                                              'All Variant B Performance',
                                              'Organic Variant C Performance',
                                              'All Variant C Performance'])
        
        initial_df_st = st.table(df_to_display)
        

        ## Let's do 50 Loops, each time increasing the optimized target sample by 1
        overall_performance = []
        overall_target_performance = []
        org_target_performance = []
        opt_target_performance = []
        overall_target_performance_variant_a = []
        overall_target_performance_variant_b = []
        overall_target_performance_variant_c = []
        organic_target_performance_variant_a = []
        organic_target_performance_variant_b = []
        organic_target_performance_variant_c = []

        steps = 75
        row_count = 100000

        for i in range(steps):

            if i == 0:
                
                Segment_df = mab.create_synthetic_sample(row_count=row_count)        
                Segment_df = mab.add_conversion_rates(df=Segment_df, seg_cols=seg_cols, segments=segments, all_combos_weights=combo_weights, print_diagnostics=False)

                ## Store results
                overall_performance = Segment_df['converted'].mean()
                org_target_performance = Segment_df.loc[Segment_df['variant_assignment'] != 'Control', 'converted'].mean()
                overall_target_performance = Segment_df.loc[Segment_df['variant_assignment'] != 'Control', 'converted'].mean()
                opt_target_performance = None
                
                ## Adding overall target performance
                overall_target_performance_variant_a = Segment_df.loc[(Segment_df['variant_assignment'] != 'Control') & (Segment_df['variant_assignment'] == 'Variant A'), 'converted'].mean()
                overall_target_performance_variant_b = Segment_df.loc[(Segment_df['variant_assignment'] != 'Control') & (Segment_df['variant_assignment'] == 'Variant B'), 'converted'].mean()
                overall_target_performance_variant_c = Segment_df.loc[(Segment_df['variant_assignment'] != 'Control') & (Segment_df['variant_assignment'] == 'Variant C'), 'converted'].mean()

                ## Adding organic performance only by variant
                organic_target_performance_variant_a = Segment_df.loc[(Segment_df['variant_assignment'] == 'target_org') & (Segment_df['variant_assignment'] == 'Variant A'), 'converted'].mean()
                organic_target_performance_variant_b = Segment_df.loc[(Segment_df['variant_assignment'] == 'target_org') & (Segment_df['variant_assignment'] == 'Variant B'), 'converted'].mean()
                organic_target_performance_variant_c = Segment_df.loc[(Segment_df['variant_assignment'] == 'target_org') & (Segment_df['variant_assignment'] == 'Variant C'), 'converted'].mean()


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
                overall_target_performance_variant_a = Segment_df_step2.loc[(Segment_df_step2['variant_assignment'] != 'Control') & (Segment_df_step2['variant_assignment'] == 'Variant A'), 'converted'].mean()
                overall_target_performance_variant_b = Segment_df_step2.loc[(Segment_df_step2['variant_assignment'] != 'Control') & (Segment_df_step2['variant_assignment'] == 'Variant B'), 'converted'].mean()
                overall_target_performance_variant_c = Segment_df_step2.loc[(Segment_df_step2['variant_assignment'] != 'Control') & (Segment_df_step2['variant_assignment'] == 'Variant C'), 'converted'].mean()

                ## Adding organic performance only by variant
                organic_target_performance_variant_a = Segment_df_step2.loc[(Segment_df_step2['variant_assignment'] == 'target_org') & (Segment_df_step2['variant_assignment'] == 'Variant A'), 'converted'].mean()
                organic_target_performance_variant_b = Segment_df_step2.loc[(Segment_df_step2['variant_assignment'] == 'target_org') & (Segment_df_step2['variant_assignment'] == 'Variant B'), 'converted'].mean()
                organic_target_performance_variant_c = Segment_df_step2.loc[(Segment_df_step2['variant_assignment'] == 'target_org') & (Segment_df_step2['variant_assignment'] == 'Variant C'), 'converted'].mean()


                ## For Next Iteration ##
                
                # Take the Target group and build an optimization score to determine how ads should be allocated
                Segment_df_step2.loc[Segment_df_step2['variant_assignment'] == 'Variant A', 'Variant_a_performance'] = Segment_df_step2['converted']
                Segment_df_step2.loc[Segment_df_step2['variant_assignment'] == 'Variant B', 'Variant_b_performance'] = Segment_df_step2['converted']
                Segment_df_step2.loc[Segment_df_step2['variant_assignment'] == 'Variant C', 'Variant_c_performance'] = Segment_df_step2['converted']
                
                ## Performance Scores all interactions
                perf_scores_all_interactions = Segment_df_step2[Segment_df_step2['target_control'] != 'control'].groupby(seg_cols).agg({'Variant_a_performance': ['mean'],
                                                                                                                                        'Variant_b_performance': ['mean'],
                                                                                                                                        'Variant_c_performance': ['mean']}).reset_index().droplevel(1, axis = 1)

            curr_table = pd.DataFrame({'Iteration': i, 
                                        'Overall Performance':overall_performance, 
                                        'All Target Performance':overall_target_performance, 
                                        'Organic Variant A Performance':organic_target_performance_variant_a,
                                        'All Variant A Performance':overall_target_performance_variant_a,
                                        'Organic Variant B Performance':organic_target_performance_variant_b,
                                        'All Variant B Performance':overall_target_performance_variant_b,
                                        'Organic Variant C Performance':organic_target_performance_variant_c,
                                        'All Variant C Performance':overall_target_performance_variant_c})
            
            initial_df_st.add_rows(curr_table)