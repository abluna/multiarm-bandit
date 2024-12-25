def mab_test():
    return 'Good - Abraham'

def get_pd_weights(segments):

    import numpy as np
    import pandas as pd
    
    ## Take Conversion Weights by group
    performance_weights =  [[0.072,0.254,0.621,0.424],
                            [0.053,0.477,0.543,0.28],
                            [0.04,0.289,0.546,0.464],
                            [0.123,0.334,0.408,0.557],
                            [0.018,0.622,0.244,0.433],
                            [0.099,0.302,0.302,0.695],
                            [0.075,0.406,0.487,0.406],
                            [0.012,0.475,0.448,0.376],
                            [0.103,0.559,0.197,0.543],
                            [0.035,0.371,0.61,0.318],
                            [0.011,0.251,0.1,0.563],
                            [0.033,0.251,0.587,0.461],
                            [0.088,0.796,0.196,0.307],
                            [0.043,0.408,0.491,0.4],
                            [0.162,0.603,0.388,0.308],
                            [0.114,0.355,0.386,0.558],
                            [0.085,0.35,0.611,0.172],
                            [0.075,0.54,0.472,0.287]]


    pd_weights = pd.DataFrame(data=performance_weights, 
                              columns=['Control', 'Variant A', 'Variant B', 'Variant C'], 
                              index = segments)
                              
    return pd_weights

def get_pd_weights_long(segments):

    import numpy as np
    import pandas as pd
    
    ## Take Conversion Weights by group
    performance_weights =  [[0.072,0.254,0.621,0.424],
                            [0.053,0.477,0.543,0.28],
                            [0.04,0.289,0.546,0.464],
                            [0.123,0.334,0.408,0.557],
                            [0.018,0.622,0.244,0.433],
                            [0.099,0.302,0.302,0.695],
                            [0.075,0.406,0.487,0.406],
                            [0.012,0.475,0.448,0.376],
                            [0.103,0.559,0.197,0.543],
                            [0.035,0.371,0.61,0.318],
                            [0.011,0.251,0.1,0.563],
                            [0.033,0.251,0.587,0.461],
                            [0.088,0.796,0.196,0.307],
                            [0.043,0.408,0.491,0.4],
                            [0.162,0.603,0.388,0.308],
                            [0.114,0.355,0.386,0.558],
                            [0.085,0.35,0.611,0.172],
                            [0.075,0.54,0.472,0.287]]

    pd_weights = pd.DataFrame(data=performance_weights, 
                              columns=['Control', 'Variant A', 'Variant B', 'Variant C'], 
                              index = segments)
                              
    ## Make Long
    pd_weights_control = pd_weights[['Control']].copy()
    pd_weights_control.columns = ['weight']
    pd_weights_control['variant'] = 'Control'
    
    pd_weights_Variant_A = pd_weights[['Variant A']].copy()
    pd_weights_Variant_A.columns = ['weight']
    pd_weights_Variant_A['variant'] = 'Variant A'    
    
    pd_weights_Variant_B = pd_weights[['Variant B']].copy()
    pd_weights_Variant_B.columns = ['weight']
    pd_weights_Variant_B['variant'] = 'Variant B'
    
    pd_weights_Variant_C = pd_weights[['Variant C']].copy()       
    pd_weights_Variant_C.columns = ['weight']
    pd_weights_Variant_C['variant'] = 'Variant C'
    
    final_pd_weights = pd.concat([pd_weights_control, pd_weights_Variant_A, pd_weights_Variant_B, pd_weights_Variant_C])
    
    final_pd_weights['Segments'] = final_pd_weights.index
    return final_pd_weights

def create_synthetic_sample(row_count=10000):
    
    import pandas as pd
    import numpy as np
    
    Segment_df = pd.DataFrame({
    'Rand_01': np.random.uniform(1, 100, row_count) ,
    'Rand_02': np.random.uniform(1, 100, row_count) ,
    'Rand_03': np.random.uniform(1, 100, row_count) ,
    'Rand_04': np.random.uniform(1, 100, row_count) ,
    'Rand_05': np.random.uniform(1, 100, row_count) ,
    'Rand_06': np.random.uniform(1, 100, row_count) ,
    'Rand_07': np.random.uniform(1, 100, row_count) ,
    })

    ## Syntax to Create Segments
    gender_condition = [
            (Segment_df['Rand_01'] < 50),
            (Segment_df['Rand_01'] >= 50)
        ]

    gender_names = [
                'Male',
                'Female',
                ]

    age_condition = [
            (Segment_df['Rand_02'] < 33),
            (Segment_df['Rand_02'] >= 33) & (Segment_df['Rand_02'] < 65),
            (Segment_df['Rand_02'] >= 65)
        ]

    age_names = [
                'Young',
                'Middle Age',
                'Older',
                ]

    income_condition = [
            (Segment_df['Rand_03'] < 33),
            (Segment_df['Rand_03'] >= 33) & (Segment_df['Rand_03'] < 66),
            (Segment_df['Rand_03'] >= 66)
        ]

    income_names = [
                'Low Income',
                'Medium Income',
                'High Income',
                ]

    buyer_condition = [
            (Segment_df['Rand_04'] < 50),
            (Segment_df['Rand_04'] >= 50)
        ]

    buyer_names = [
                'Prior Buyer',
                'First-Time Buyer',
                ]

    region_condition = [
            (Segment_df['Rand_05'] < (1/4)*100 ),
            (Segment_df['Rand_05'] >= (1/4)*100) & (Segment_df['Rand_05'] < (2/4)*100),
            (Segment_df['Rand_05'] >= (2/4)*100) & (Segment_df['Rand_05'] < (3/4)*100),
            (Segment_df['Rand_05'] >= (3/4)*100)
        ]

    region_names = [
                'North',
                'West',
                'South',
                'East'
                ]

    area_condition = [
            (Segment_df['Rand_06'] < 50),
            (Segment_df['Rand_06'] >= 50)
        ]

    area_names = [
                'Urban',
                'Suburban'
                ]

    parent_condition = [
            (Segment_df['Rand_06'] < 50),
            (Segment_df['Rand_06'] >= 50)
        ]

    parent_names = [
                    'Non-Parent',
                    'Parent'
                ]

    Segment_df['gender'] = np.select(gender_condition, gender_names, default='Unknown')
    Segment_df['age'] = np.select(age_condition, age_names, default='Unknown')
    Segment_df['income'] = np.select(income_condition, income_names, default='Unknown')
    Segment_df['buyer'] = np.select(buyer_condition, buyer_names, default='Unknown')
    Segment_df['region'] = np.select(region_condition, region_names, default='Unknown')
    Segment_df['area'] = np.select(area_condition, area_names, default='Unknown')
    Segment_df['parent'] = np.select(parent_condition, parent_names, default='Unknown')
    
    cols_of_interest = ["gender", "age", "income", "buyer", "region", "area", "parent"]

    return Segment_df[cols_of_interest]
    
def add_conversion_rates(df, seg_cols=None, segments=None, all_combos_weights=None, assign_variant=True, print_diagnostics=True):
    
    import numpy as np
    import pandas as pd
    
    pd_weights = get_pd_weights_long(segments=segments)

    variant_names = [
            'Control',
            'Variant A',
            'Variant B',
            'Variant C',
            ]    

    if assign_variant:

        ## Full random assignment for Variant
        df['Rand_08'] = np.random.uniform(1, 100, len(df))

        variant_condition = [
                (df['Rand_08'] <= 5),
                (df['Rand_08'] > 5) & (df['Rand_08'] <= 37),
                (df['Rand_08'] > 37) & (df['Rand_08'] <= 69),
                (df['Rand_08'] > 69)
            ]

        df['variant_assignment'] = np.select(variant_condition, variant_names, default='Unknown')
        
    orig_cols = list(df.columns)
        
    ## Loop through each segment to get the segment weights
    for col in seg_cols:
        cur_pd_weights = pd_weights
        cur_pd_weights['seg_names'] = cur_pd_weights.index
        df = df.merge(cur_pd_weights, how='left', left_on = [col,'variant_assignment'], right_on = ['seg_names','variant'], suffixes = (None, "_" + col))
                
    cols_with_weights = [col for col in df.columns if 'weight' in col]
    df['final_seg_weight'] = df[cols_with_weights].mean(axis=1)
    
    ## Now, let's add the all-combos weights
    df = df.merge(all_combos_weights, how='left', on = ["gender","age","income","buyer","region","area","parent","variant_assignment"])
    df['combined_weight'] = df['combos_weights'] * (1 + df['final_seg_weight'])
    df['converted'] = df.apply(lambda row: np.random.choice([1,0], p=[row['combined_weight'], 1 - row['combined_weight']]), axis=1)

    if print_diagnostics:
        df['was_modified'] = df['variant_assignment'] != 'Control'
        modified_ratios = df.groupby('was_modified').agg({'converted': ['mean', 'count', 'sum']})
        display(modified_ratios)

    cols_to_return = orig_cols + ['final_seg_weight', 'combos_weights', 'combined_weight', 'converted']

    return df[cols_to_return]
     
def assignment_with_optimization(df, prior_performance_scores=None,seg_cols=None,method='max', opt_target_size=0.05, learning_weight=1):

    import numpy as np
    import pandas as pd
    
    ## Create Optimized Target vs Organic Target vs. Control
    size_control = 0.05
    size_optimized_target =  opt_target_size
    size_organic_target = 1 - (size_control + size_optimized_target)

    df['target_control'] = np.random.choice(['target_org', 'target_opt', 'control'], len(df), p=[size_organic_target, size_optimized_target, size_control])

    Segment_df_step2_target_opt = df[df['target_control']=='target_opt'].copy()
    Segment_df_step2_target_org = df[df['target_control']=='target_org'].copy()
    Segment_df_step2_control = df[df['target_control']=='control'].copy()

    Segment_df_step2_target_org['variant_assignment'] = np.random.choice(['Variant A', 'Variant B', 'Variant C'], len(Segment_df_step2_target_org), p=[1/3,1/3,1/3])
    Segment_df_step2_control['variant_assignment'] = "Control"

    ###  Create routine for assigning weights to each variant ###
    Segment_df_step2_target_opt = Segment_df_step2_target_opt.merge(prior_performance_scores, how='left', on=["gender","age","income","buyer","region","area","parent"])

    if method=='prob':
    
        ## Turn it into a share
        variant_col_names = ['Variant_a_performance','Variant_b_performance','Variant_c_performance']
        Segment_df_step2_target_opt[variant_col_names] = Segment_df_step2_target_opt[variant_col_names] ** learning_weight

        Segment_df_step2_target_opt['final_denominator'] = Segment_df_step2_target_opt[variant_col_names].sum(axis=1)
        for i in range(3):
            Segment_df_step2_target_opt[variant_col_names[i] + '_share'] = Segment_df_step2_target_opt[variant_col_names[i]] / Segment_df_step2_target_opt['final_denominator']

        Segment_df_step2_target_opt['variant_assignment'] = Segment_df_step2_target_opt.apply(lambda row: np.random.choice(['Variant A', 'Variant B', 'Variant C'], p=row[['Variant_a_performance_share','Variant_b_performance_share','Variant_c_performance_share']]), axis=1)

        cols_for_concat = ['gender', 'age', 'income', 'buyer', 'region', 'area', 'parent','variant_assignment','target_control']
        
    if method=='max':
        ## Turn it into a share
        variant_col_names = ['Variant_a_performance','Variant_b_performance','Variant_c_performance']
        Segment_df_step2_target_opt[variant_col_names] = Segment_df_step2_target_opt[variant_col_names] ** learning_weight

        Segment_df_step2_target_opt['final_denominator'] = Segment_df_step2_target_opt[variant_col_names].sum(axis=1)
        for i in range(3):
            Segment_df_step2_target_opt[variant_col_names[i] + '_share'] = Segment_df_step2_target_opt[variant_col_names[i]] / Segment_df_step2_target_opt['final_denominator']
        
        variant_col_names_share = ['Variant_a_performance_share','Variant_b_performance_share','Variant_c_performance_share']
        Segment_df_step2_target_opt['variant_assignment'] = Segment_df_step2_target_opt[variant_col_names].idxmax(axis='columns')
        Segment_df_step2_target_opt['max_prob'] = Segment_df_step2_target_opt[variant_col_names_share].max(axis=1)
        Segment_df_step2_target_opt['core_membership'] = np.where(Segment_df_step2_target_opt['max_prob'] > 0.7, 'Yes', 'No')
        
        Segment_df_step2_target_opt = Segment_df_step2_target_opt.replace({'variant_assignment' : {'Variant_a_performance' : 'Variant A', 
                                                                                                   'Variant_b_performance' : 'Variant B', 
                                                                                                   'Variant_c_performance' : 'Variant C'}})
        
        cols_for_concat = ['gender', 'age', 'income', 'buyer', 'region', 'area', 'parent','variant_assignment','target_control','max_prob','core_membership']
    
    ## join the three tables  
    Segment_df_step2_final = pd.concat([Segment_df_step2_target_opt, Segment_df_step2_target_org,Segment_df_step2_control], sort=True)[cols_for_concat]
    Segment_df_step2_final.sort_index()
    
    return Segment_df_step2_final
 
def progress_bar(current, total, bar_length=20):
    fraction = current / total
    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '
    ending = '\n' if current == total else '\r'
    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)
       
def create_all_combo_weights():

    import itertools
    import pandas as pd
    import numpy as np
    
    df = pd.DataFrame({'gender': ["Male", "Female",np.nan,np.nan] ,
                       'age': ["Young", "Middle Age", "Older",np.nan],
                       'income': ["Low Income", "Medium Income", "High Income",np.nan],
                       'buyer': ["Prior Buyer", "First-Time Buyer",np.nan,np.nan],
                       'region': ["North", "West", "South", "East"],
                       'area': ["Urban", "Suburban",np.nan,np.nan],
                       'parent': ["Non-Parent", "Parent",np.nan,np.nan]})
        
    combinations = list(itertools.product(*df.values.T))
    result_df = pd.DataFrame(combinations, columns=df.columns).dropna().reset_index(drop=True)

    ## Add Random Weights for each variant
    variant_a = result_df.copy()
    variant_b = result_df.copy()
    variant_c = result_df.copy()
    variant_control = result_df.copy()
    
    variant_a["combos_weights"] = np.random.chisquare(15, size=len(result_df)) / 100
    variant_a['variant_assignment'] = 'Variant A'

    variant_b["combos_weights"] = np.random.chisquare(15, size=len(result_df)) / 100
    variant_b['variant_assignment'] = 'Variant B'
    
    variant_c["combos_weights"] = np.random.chisquare(15, size=len(result_df)) / 100
    variant_c['variant_assignment'] = 'Variant C'
    
    variant_control["combos_weights"] = np.random.chisquare(3, size=len(result_df)) / 100
    variant_control['variant_assignment'] = 'Control'
    
    result_df = pd.concat([variant_a, variant_b, variant_c, variant_control]).reset_index(drop=True)
    
    return result_df

def draw_histogram(n_array, n_bins=50):
    
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a histogram with 50 bins
    plt.hist(n_array, bins=n_bins)

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram')

    # Display the histogram
    plt.show()


def get_variant_assignment_counts(df, table_name, seg_cols):

    import pandas as pd

    df_list = []
    for j in seg_cols:
        seg_counts = df.groupby(j).agg({'Variant_a_performance': ['count'],
                                        'Variant_b_performance': ['count'],
                                        'Variant_c_performance': ['count']}).astype('float64').reset_index().droplevel(
            1, axis=1)

        denom = seg_counts.iloc[:, [1, 2, 3]].sum(axis=1)
        for i in [1, 2, 3]:
            seg_counts.iloc[:, i] = seg_counts.iloc[:, i] / denom

        ## Create Tuple
        seg_name = seg_counts.columns[0]
        my_tuple_rows = merged_list = [(seg_name, seg) for seg in seg_counts.iloc[:, 0]]
        my_tuple_columns = [(table_name, 'Variant A'), (table_name, 'Variant B'), (table_name, 'Variant C')]

        final_df = pd.DataFrame(data=seg_counts.iloc[:, [1, 2, 3]].values,
                                columns=pd.MultiIndex.from_tuples(my_tuple_columns),
                                index=pd.MultiIndex.from_tuples(my_tuple_rows))

        df_list.append(final_df)

    concat_df = pd.concat(df_list)
    return concat_df