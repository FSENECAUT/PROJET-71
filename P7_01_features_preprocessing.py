import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import gc

# Séparation des variables selon leurs types
def separation_des_variables_selon_type(df):
    # Regroupement des variables catégorielles et numériques dans 2 listes
    numerical_columns = df.columns[(df.dtypes == 'float64')
                          | (df.dtypes == 'int64')|(df.dtypes == 'int8')]
    categorical_columns = df.columns[df.dtypes == 'object']
    return categorical_columns, numerical_columns

# calcul du top des fréquences des valeurs d'une liste
def calcul_top_frequence(liste):
    cnt = len(list(Counter(liste)))
    top_frequence = Counter(liste).most_common()[cnt-2]
    return top_frequence

# 
def creation_var_top_modalite(df, variable, col_count):
    #variable_1=(variable + "_" + "liste").upper()
    df[variable] = df[variable].dropna().apply(lambda x: calcul_top_frequence(x))
    separer_tuple = pd.DataFrame(df[variable].values.tolist(), index=df.index)
    df[variable + "_" + "classe".upper()] = separer_tuple[0]
    df[variable + "_" + "proportion".upper()] = separer_tuple[1]/df[col_count]
    df = df.drop([variable], axis=1)
    return df

# Preprocess application_train.csv and application_test.csv
def application_train_test():
    # Read data and merge
    df = pd.read_csv('data/application_train.csv', dtype={'SK_ID_CURR':'object'})
    test_df = pd.read_csv('data/application_test.csv', dtype={'SK_ID_CURR':'object'})
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index(drop=True)
    # Remove 4 values with 'XNA' CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect()
    return df


## Bureau et bureau_balance
def bureau_and_balance():
    bureau = pd.read_csv('data/bureau.csv',dtype={'SK_ID_CURR':'object', 'SK_ID_BUREAU':'object'})
    bb = pd.read_csv('data/bureau_balance.csv',dtype={'SK_ID_BUREAU':'object','MONTHS_BALANCE':'object'})
    bb_cat, bb_num = separation_des_variables_selon_type(bb)
    # Mapping pour éliminer les 'X' et 'C'
    bb['STATUS_MOD']=bb['STATUS'].map({'0':'0', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', 'X':'0', 'C':'0'})
    # aggregation avec création variables 'DPD_classe' et 'DPD_proportion'
    bb_agg = bb.groupby(['SK_ID_BUREAU']).agg({'MONTHS_BALANCE':'count', 'STATUS_MOD':list})
    bb_agg['DPD'] = bb_agg['STATUS_MOD'].dropna().apply(
        lambda x: calcul_top_frequence(x))
    DPD = pd.DataFrame(bb_agg['DPD'].values.tolist(), index=bb_agg.index)
    bb_agg['DPD_classe'] = DPD[0]
    bb_agg['DPD_proportion'] = DPD[1]/bb_agg['MONTHS_BALANCE']
    # supression colonnes inutiles
    bb_agg = bb_agg.drop(['STATUS_MOD', 'DPD'], axis=1)
    # jointure entre bb_agg et bureau
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    # bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True) # supression colonne de jointure
    del bb, bb_agg # nettoyage des fichiers inutiles
    gc.collect()
    # Bureau and bureau_balance numeric features
    cat_bureau, num_bureau = separation_des_variables_selon_type(bureau)
    aggregations = { col:['mean'] for col in num_bureau}
    # Bureau and bureau_balance categorical features
    cat_aggregations = {'SK_ID_BUREAU': ['count'],
    'CREDIT_ACTIVE': [list],
    'CREDIT_CURRENCY': [list],
    'CREDIT_TYPE': [list],
    'DPD_classe': [list]}
    aggregations.update(cat_aggregations)
    # Aggregation du dataframe bureau
    bureau_agg = bureau.groupby('SK_ID_CURR').agg(aggregations)
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Création de nouvelles variables catégorielles
    for col in list(cat_bureau)[2:]:
        bureau_agg = creation_var_top_modalite(bureau_agg, 'BURO_'+col+'_LIST', 'BURO_SK_ID_BUREAU_COUNT')
    gc.collect()
    return bureau_agg


### Preprocess previous_applications.csv
def previous_applications():
    prev = pd.read_csv('data/previous_application.csv',dtype={'SK_ID_CURR':'object', 'SK_ID_PREV':'object', 'NFLAG_LAST_APPL_IN_DAY':'object', 'NFLAG_INSURED_ON_APPROVAL':'object'} )
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['AMT_CREDIT'].replace(0, 8.054100e+04, inplace=True)
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    cat_prev, num_prev = separation_des_variables_selon_type(prev)
    aggregations={ col :['mean'] for col in num_prev}
    # Previous applications categorical features
    cat_aggregations={'SK_ID_PREV':['count'], 'NAME_CONTRACT_TYPE': [list],
    'WEEKDAY_APPR_PROCESS_START': [list], 'FLAG_LAST_APPL_PER_CONTRACT': [list], 'NFLAG_LAST_APPL_IN_DAY': [list],
    'NAME_CASH_LOAN_PURPOSE': [list], 'NAME_CONTRACT_STATUS': [list], 'NAME_PAYMENT_TYPE': [list],
    'CODE_REJECT_REASON': [list], 'NAME_TYPE_SUITE': [list], 'NAME_CLIENT_TYPE': [list],
    'NAME_GOODS_CATEGORY': [list], 'NAME_PORTFOLIO': [list], 'NAME_PRODUCT_TYPE': [list],
    'CHANNEL_TYPE': [list], 'NAME_SELLER_INDUSTRY': [list], 'NAME_YIELD_GROUP': [list],
    'PRODUCT_COMBINATION': [list], 'NFLAG_INSURED_ON_APPROVAL': [list]}
    aggregations.update(cat_aggregations)
    # agregation selon 'SK_ID_CURR'    
    prev_agg = prev.groupby('SK_ID_CURR').agg(aggregations)
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Création de nouvelles variables catégorielles
    for col in list(cat_prev)[2:]:
        prev_agg = creation_var_top_modalite(prev_agg, 'PREV_'+col+'_LIST', 'PREV_SK_ID_PREV_COUNT')
    return prev_agg


    # Preprocess POS_CASH_balance.csv
def pos_cash():
    pos = pd.read_csv('data/POS_CASH_balance.csv',  dtype={'SK_ID_CURR':'object', 'SK_ID_PREV':'object','MONTHS_BALANCE':'object'})
    cat_pos, num_pos = separation_des_variables_selon_type(pos)
    # Previous applications numeric features
    aggregations={ col :['mean'] for col in num_pos}
    # Previous applications categorical features
    cat_aggregations={'SK_ID_PREV': ['nunique'], 'MONTHS_BALANCE': ['count'], 'NAME_CONTRACT_STATUS': [list]}
    aggregations.update(cat_aggregations)
    aggregations.update(cat_aggregations)
    # Agregaton selon 'SK_ID_CURR'
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Création de nouvelles variables catégorielles 
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    # 
    pos_agg = creation_var_top_modalite(pos_agg, 'POS_NAME_CONTRACT_STATUS_LIST', 'POS_SK_ID_PREV_NUNIQUE')
    del pos
    gc.collect()
    return pos_agg

# Preprocess installments_payments.csv
def installments_payments():
    ins = pd.read_csv('data/installments_payments.csv',dtype={'SK_ID_PREV':'object', 'SK_ID_CURR':'object'} )
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['AMT_INSTALMENT'].replace(0,1,inplace=True)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Previous applications numeric features
    cat_ins, num_ins = separation_des_variables_selon_type(ins)
    aggregations={'NUM_INSTALMENT_VERSION': ['nunique'],'DAYS_INSTALMENT': ['mean'],
    'DAYS_ENTRY_PAYMENT': ['mean'],'AMT_INSTALMENT': ['mean'],
    'AMT_PAYMENT': ['mean'],'PAYMENT_PERC': ['mean'],
    'PAYMENT_DIFF': ['mean'],'DPD': ['mean'],'DBD': ['mean']}
    # Previous applications categorical features
    aggregations['SK_ID_PREV']=['nunique']
    # Agregaton selon 'SK_ID_CURR'
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance():
    cc = pd.read_csv('data/credit_card_balance.csv',    dtype={'SK_ID_PREV':'object', 'SK_ID_CURR':'object','MONTHS_BALANCE':'object'})
    # Previous applications numeric features
    cat_cc, num_cc = separation_des_variables_selon_type(cc)
    aggregations={ col :['mean'] for col in num_cc}
    # Previous applications categorical features
    cat_aggregations={'SK_ID_PREV': ['nunique'],  'MONTHS_BALANCE': ['count'],
    'NAME_CONTRACT_STATUS': [list]}
    aggregations.update(cat_aggregations)
    cc_agg = cc.groupby('SK_ID_CURR').agg(aggregations)
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Création de nouvelles variables catégorielles
    cc_agg = creation_var_top_modalite(cc_agg, 'CC_NAME_CONTRACT_STATUS_LIST', 'CC_SK_ID_PREV_NUNIQUE')
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

print("before __name__ guard")  # possibilité d'exécuter le script directement (partie ci-dessous)
if __name__ == '__main__':

    ### Création des dataframes 'application_train' et 'aplication_test'
    df = application_train_test()
    df_cat, df_num = separation_des_variables_selon_type(df)
    application_train = df[df.loc[:, 'TARGET'].notna()]
    application_test = df[df.loc[:, 'TARGET'].isna()]
    application_test.drop('TARGET', axis=1, inplace=True)
    application_train.shape # (307507, 248)
    application_test.shape #  (48744, 247)

    ### Regroupement du dataframe df avec bureau_agg : train_1 et test_1
    # Création du dataframe intégrant 'Bureau_balance' et 'bureau' (305811 lignes*116 colonnes - index égal SK_ID_CURR)
    bureau_agg = bureau_and_balance()
    # Jointure entre application_train et  bureau_agg
    application_train['SK_ID_CURR'] = application_train['SK_ID_CURR'].astype('object')
    train_1 = application_train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
    train_1.shape # 307507 lignes , 151 colonnes 
    # Jointure entre application_test et  bureau_agg
    application_test['SK_ID_CURR'] = application_test['SK_ID_CURR'].astype('object')
    test_1 = application_test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
    test_1.shape # 48744 lignes et 150 colonnes

    ### Regroupement du dataframe train_2 (test_2) avec prev_agg et ins_agg
    # Création des dataframes prev_agg et ins_agg
    # previous_applications (338857 lignes*56 colonnes - index égal SK_ID_CURR)
    prev_agg = previous_applications()
    prev_agg.drop(['PREV_RATE_INTEREST_PRIMARY_MEAN', 
        'PREV_RATE_INTEREST_PRIVILEGED_MEAN'], axis=1, inplace=True) # 98% de valeurs manquantes
    # installments_payments (339587  lignes*11 colonnes - index égal SK_ID_CURR)
    ins_agg = installments_payments()
    # Jointure entre prev_agg et ins_agg
    prev_ins_agg = prev_agg.merge(ins_agg, on = 'SK_ID_CURR', how = 'left')
    # Jointure train_1 (test_1) avec prev_ins_agg
    train_2 = train_1.merge(prev_ins_agg, on = 'SK_ID_CURR', how = 'left')
    train_1.shape # 307507 lignes , 151 colonnes
    test_2 = test_1.merge(prev_ins_agg, on = 'SK_ID_CURR', how = 'left')
    test_2.shape # 48744 lignes , 217 colonnes

### Regroupement du dataframe train_3 (test_3) avec pos_agg et cc_agg
    # Création des dataframes pos_agg et cc_agg
    # POS_CASH_balance (337252 lignes*9 colonnes - index égal SK_ID_CURR)
    pos_agg = pos_cash()
    # credit_card_balance (103558 lignes*24 colonnes - index égal SK_ID_CURR)
    # Jointure entre prev_agg et ins_agg
    pos_cc_agg = pos_agg.merge(cc_agg, on = 'SK_ID_CURR', how = 'left')
    # Jointure train_2 (test_2) avec pos_cc_agg
    train_3 = train_2.merge(pos_cc_agg, on = 'SK_ID_CURR', how = 'left')
    train_3.shape # 307507 lignes , 247 colonnes
    test_3 = test_2.merge(pos_cc_agg, on = 'SK_ID_CURR', how = 'left')
    test_3.shape  # 48744 lignes, 246 colonnes
    
    # Déplacer la colonne "Target" en fin de dataframe train_3
    # Obtenir la liste de colonnes
    cols = list(train_3)
    # Déplacer "TARGET" en fin de liste en utilisant index, pop and insert
    cols.insert(247, cols.pop(cols.index('TARGET')))
    # Réordonner le dataframe
    train_3 = train_3.loc[:, cols]

    # Export des dataframes train / test version 1, 2, 3
    train_1.to_csv('data/train_1.csv')
    train_2.to_csv('data/train_2.csv')
    train_3.to_csv('data/train_3.csv')
    test_1.to_csv('data/test_1.csv')
    test_2.to_csv('data/test_2.csv')
    test_3.to_csv('data/test_3.csv')
    # dataframe df pour application Dash
    description_client = df[['SK_ID_CURR', 'CODE_GENDER','CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
        'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS']]
    description_client.to_csv('data/description_client.csv')