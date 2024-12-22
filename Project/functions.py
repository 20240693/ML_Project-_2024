# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:24:43 2024

"""

# Imports
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.metrics import f1_score
from sklearn.base import clone





#Function to ensure consistency on the carrier names

 # Terms that will be replaced
replacement_map = {
    'INS': 'INSURANCE',
    'ED': 'EDUCATION',
    'HLTH': 'HEALTH',
    'MED': 'MEDICAL',
    ' & ': ' AND ',
    'NY': 'NEW YORK',
    'CENTRAL': 'CSD',
    'UNION': 'UFSD',
}

 # Terms to remove
terms_to_remove = ['INC', 'CO', 'COMPANY', 'CORPORATION', 'COUNTY OF', 'CORP', 'SCHOOL', 'CSD', 'DIST', 'DIS', 'OF', 'DISTRICT', 'SCH', 'SCHL', 'FREE']
 
 #function
def clean_carrier_name(name):
    if isinstance(name, str):
        cleaned_name = re.sub(r"[.,]", "", name)  # Remove periods and commas
        for key, value in replacement_map.items():
            cleaned_name = re.sub(rf"\b{key}\b", value, cleaned_name)  # Replace abbreviations
        for term in terms_to_remove:
            cleaned_name = re.sub(rf"\b{term}\b", "", cleaned_name)  # Remove unwanted terms
        cleaned_name = cleaned_name.replace("  ", " ")  # Remove double spaces
        return cleaned_name.strip()  # Remove spaced from beginning or end
    else: # will handle the issing values
       return name 
   
#function to ensure consistency on zip codes (considering US valid Zip Codes )
   
def invalid_zip_codes(zip_codes):
    invalid_codes = []

    for zip_code in zip_codes:
        if isinstance(zip_code, str) and (not (zip_code.isdigit() and len(zip_code) == 5)):
            invalid_codes.append(zip_code)

    return invalid_codes

#function to replace invalid Zip codes with missing values

def replace_invalid_zip_codes(zip_codes):
    zip_codes = zip_codes.apply(lambda x: x if (isinstance(x, str) and x.isdigit() and len(x) == 5) else np.nan)
    return zip_codes



#function to check coherence on codes
def check_code_description_duplicates(train_data,code_column, description_column):
    # Checking  codes with multiple descriptions
    duplicated_codes = train_data.groupby(code_column)[description_column].nunique()
    codes_with_multiple_descriptions = duplicated_codes[duplicated_codes > 1]
    
    # Checking  descriptions with multiple codes
    duplicated_descriptions = train_data.groupby(description_column)[code_column].nunique()
    descriptions_with_multiple_codes = duplicated_descriptions[duplicated_descriptions > 1]
    
    return codes_with_multiple_descriptions, descriptions_with_multiple_codes 



#to detect outliers
def calculate_percentage_outliers(data, col):
    total_rows = len(data)
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count rows outside the IQR bounds
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    percentage_removed = (len(outliers) / total_rows) * 100
    return percentage_removed

#to replace outliers with the bounds of IQR
def replace_outliers(train,test, col):  #train is fromwhere we want  to take the values and test to replace
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1 -(1.5*IQR)
    upper_whisker = Q3 + (1.5*IQR)
    test[col]=np.where(test[col]>upper_whisker,upper_whisker,np.where(test[col]<lower_whisker,lower_whisker,test[col]))

#to produce the correlation heatmap
def cor_heatmap(cor):
    plt.figure(figsize=(12,10))
    sns.heatmap(data = cor, annot = True, cmap = plt.cm.Reds, fmt='.1')
    plt.show()

#Chi-Square Test
def TestIndependence(X,y,var,alpha=0.05):        
    dfObserved = pd.crosstab(y,X) 
    chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
    if p<alpha:
        result="{0} is IMPORTANT for Prediction".format(var)
    else:
        result="{0} is NOT an important predictor. (Discard {0} from model)".format(var)
    print(result)
    
#compare model f1_score

def compare_model_f1_scores(df, models, train_scores,val_scores):
    count = 0
    for count, model in enumerate(models):
        train_score = train_scores.get(f'f1_train_{model}')
        val_score = val_scores.get(f'f1_val_{model}')
        
        df.iloc[count] = [train_score, val_score]
        count +=1
    
    return df


#preprocessing function
def preprocess_X(train_data,val_data):
    # Treat outliers
    numerical_columns = ['Age at Injury', 'Average Weekly Wage', 'Number of Dependents', 'Age', 'IME-4 Count',
                         'Accident Year', 'Days_from_Assembly Date']
    for col in numerical_columns:
        replace_outliers(train_data,train_data, col)
        replace_outliers(train_data,val_data, col)
    
    # Treat missing values
    columns_to_drop = train_data.columns[train_data.isnull().mean() > 0.8] 
    train_data = train_data.drop(columns=columns_to_drop)
    val_data = val_data.drop(columns=columns_to_drop)
    
    condition = train_data['Age at Injury'].isna() & train_data['Age'].notna()
    train_data.loc[condition, 'Age at Injury'] = np.floor( train_data['Age']-(train_data['Days_from_Assembly Date']/365))

    condition = val_data['Age at Injury'].isna() & val_data['Age'].notna()
    val_data.loc[condition, 'Age at Injury'] = np.floor( val_data['Age']-(val_data['Days_from_Assembly Date']/365))

    
    cat_cols_to_fill = ['Industry Code', 'WCIO Cause of Injury Code', 'WCIO Nature of Injury Code',
                        'WCIO Part Of Body Code', 'Zip Code', 'County of Injury']
    for col in cat_cols_to_fill:
        mode = train_data[col].mode()[0]
        train_data[col].fillna(mode, inplace=True)
        val_data[col].fillna(mode, inplace=True)
 
    mean_cols = ["Age","Age at Injury"]

    for col in mean_cols:
        mean = round(train_data[col].mean())
        train_data[col].fillna(mean, inplace=True)
        val_data[col].fillna(mean, inplace=True)
        
    median_cols_round = ["Accident Year"]

    for col in median_cols_round:
        median = round(train_data[col].median())
        train_data[col].fillna(median, inplace=True)
        val_data[col].fillna(median, inplace=True)

    industry_mean_train = train_data.groupby('Industry Code')['Average Weekly Wage'].median()
    overall_mean_train = train_data['Average Weekly Wage'].median()


    train_data['Average Weekly Wage'] = train_data['Average Weekly Wage'].fillna(train_data['Industry Code'].map(industry_mean_train))
    val_data['Average Weekly Wage'].fillna(val_data['Industry Code'].map(industry_mean_train).fillna(overall_mean_train),inplace=True)
    
    train_data['IME-4 Count'].fillna(0, inplace=True)
    val_data['IME-4 Count'].fillna(0, inplace=True)
    #Encoding
    enc1 = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1) 
    
    columns_enc=['Alternative Dispute Resolution', 'Attorney/Representative',  'Carrier Name', 'Carrier Type', 
                 'County of Injury','Medical Fee Region', 'COVID-19 Indicator', 'District Name', 'Gender', 
                 'Agreement Reached', 'WCB Decision']
    train_data[columns_enc] = enc1.fit_transform(train_data[columns_enc])
    val_data[columns_enc] = enc1.transform(val_data[columns_enc])
    for column in columns_enc:
        mode = train_data[train_data[column] != -1][column].mode()[0] 
        val_data[column] = val_data[column].replace(-1, mode)

    #Scaling 
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(train_data)  
    X_val_scaled = scaler.transform(val_data)  


    train_data = pd.DataFrame(X_train_scaled, columns=train_data.columns).set_index(train_data.index)
    val_data= pd.DataFrame(X_val_scaled, columns=val_data.columns).set_index(val_data.index)
    
    return train_data,val_data

#compare default vs tuned
def compare_model_f1_scores_tuned(df, models, default_scores,opt_scores):
    count = 0
    for count, model in enumerate(models):
        default_score = default_scores.get(f'f1_val_{model}')
        opt_score = opt_scores.get(f'f1_val_tuned_{model}')
        
        df.iloc[count] = [default_score, opt_score]
        count +=1
    
    return df


#plot model performnce
def plot_model_comparison(data, title='Model Performance Comparison', colors=['yellowgreen', 'dimgray']):
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.4
    index_positions = range(len(data))

    # Creating bars
    default_bars = ax.bar(index_positions, data['Macro F1 Score Default'], width=bar_width, label='Default', color=colors[0])
    tuned_bars = ax.bar([i + bar_width for i in index_positions], data['Macro F1 Score Optimized'], width=bar_width, label='Tuned', color=colors[1])

    # Adding values above bars
    for bar in default_bars + tuned_bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height), ha='center', va='bottom')

    # Adding title, labels, and legend
    plt.title(title)
    plt.ylabel('F1 Score')
    plt.xticks([i + bar_width / 2 for i in index_positions], data.index, ha='center')
    plt.legend()

    plt.tight_layout()
    plt.show()

#agr vs withour agr
def show_results(df, *args, X_train_scaled, y_train,X_val_scaled,y_val,X_train_agr,X_val_agr):
    count = 0
    # for each model passed as argument
    for arg in args:
        model_1 = clone(arg)
        model_2 = clone(arg)
 
        # fit the model to the data
        model_1.fit(X_train_scaled,y_train)
        # check the  F1 Score for data without Agreement Reacched
        test_pred = model_1.predict(X_val_scaled)
        value_test = round(f1_score(test_pred,y_val,average='macro'),3)
        
        # fit the model to the data
        model_2.fit(X_train_agr,y_train)
        # check the  F1 Score for data with the prediction of Agreement Reacched
        test_pred = model_2.predict(X_val_agr)
        value_test_agr = round(f1_score(test_pred,y_val,average='macro'),3)
  
        df.iloc[count] = value_test, value_test_agr
    
        count+=1
        
    return df




    
