import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import dash_table
import plotly.graph_objects as go
from joblib import dump, load
import assets.fonctions_P7 as fct
import plotly.express as px

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

navbar = dbc.NavbarSimple(
    children=[
        dbc.Button("Sidebar", outline=True, color="secondary", className="mr-1", id="btn_sidebar"),
    ],
    brand="Dashboard",
    brand_href="#",
    color="dark",
    dark=True,
    fluid=True,
)


# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 62.5,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0.5rem 1rem",
    "background-color": "#f8f9fa",
}

SIDEBAR_HIDEN = {
    "position": "fixed",
    "top": 62.5,
    "left": "-16rem",
    "bottom": 0,
    "width": "16rem",
    "height": "100%",
    "z-index": 1,
    "overflow-x": "hidden",
    "transition": "all 0.5s",
    "padding": "0rem 0rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "transition": "margin-left .5s",
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE1 = {
    "transition": "margin-left .5s",
    "margin-left": "2rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [
        html.H2("Menu", className="display-4"),
        html.Hr(),
        html.P(
            "Attribution Prêt Client", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Descriptif client", href="/page-1", id="page-1-link", active=True),
                dbc.NavLink("Probabilité de défaut", href="/page-2", id="page-2-link"),
                dbc.NavLink("Comparatif client", href="/page-3", id="page-3-link"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    id="sidebar",
    style=SIDEBAR_STYLE,
)

content = html.Div(
    id="page-content",
    style=CONTENT_STYLE)



app.layout = html.Div(
    [
        dcc.Store(id='side_click'),
        dcc.Location(id="url"),
        navbar,
        sidebar,
        content,
    ],
)


@app.callback(
    [
        Output("sidebar", "style"),
        Output("page-content", "style"),
        Output("side_click", "data"),
    ],

    [Input("btn_sidebar", "n_clicks")],
    [
        State("side_click", "data"),
    ]
)
def toggle_sidebar(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = 'SHOW'

    return sidebar_style, content_style, cur_nclick

# Contenu commun aux 3 pages
# Intégration des dataset avec choix du nombre de lignes via paramètre nrows)
description_client = pd.read_csv('D:\documents\(2) OPENCLASSROOMS\PROJET 7\data\\train_3.csv', nrows=1000, sep=',', index_col=0)
description_client_2 = pd.read_csv('D:\documents\(2) OPENCLASSROOMS\PROJET 7\data\\test_3.csv', nrows=1000, sep=',', index_col=0)
coefs = pd.read_csv('D:\documents\(2) OPENCLASSROOMS\PROJET 7\data\\coefs.csv', sep=',', index_col=0)  
description_client_22 = pd.read_csv('D:\documents\(2) OPENCLASSROOMS\PROJET 7\data\\test_3_scoring.csv', nrows=1000, sep=',', index_col=0)

def generate_table(dataframe, cols, max_rows=1):
    # Génére une table composée d'une ligne (max_rows=1)
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in cols]),
            className="table"
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in cols
            ])  for i in range(min(len(dataframe), max_rows))
        ]),
    ])

### Contenu de la page 1
description_client_2['YEARS_BIRTH']=round(-description_client_2['DAYS_BIRTH']/365,0)
description_client_2['YEARS_EMPLOYED']=round(-description_client_2['DAYS_EMPLOYED']/365,0)
mask_identification=['SK_ID_CURR', 'CODE_GENDER', 'YEARS_BIRTH', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'CNT_FAM_MEMBERS', 'CNT_CHILDREN']
mask_revenu=['NAME_INCOME_TYPE', 'AMT_INCOME_TOTAL', 'YEARS_EMPLOYED', 'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE']
mask_revenu_par_pers=['INCOME_PER_PERSON']
mask_dm_pret=['NAME_CONTRACT_TYPE', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE']

# Affichage de la page 1
page_1_layout = html.Div(
        children=[
                # Titre de la page
                html.Div(
                    children=[
                        html.H1(
                            children="Descriptif du demandeur de prêt", className="header-title"
                        ),
                        html.P(
                            children="éléments d'identité et financiers",
                            className="header-description",
                        ),
                    ],
                    className="header",
                ),
                # Liste déroulante pour sélectionner la référence client
                html.Div(
                    children=[
                        html.Div(children="Référence du client", className="menu-title"),
                        dcc.Dropdown(
                            id='filtre_client',
                            options=[
                                {'label': i, 'value': i}
                                for i in np.sort(description_client_2['SK_ID_CURR'].unique())                            
                                ],
                            value=100001,
                            clearable=False,
                            persistence=True,
                            className="dropdown",
                        ),
                    ]
                ),
                # Première ligne de cellules
                html.Div(
                    children=[
                        html.H3('Elements d\'identification du demandeur'),
                        html.H4( id='identification_client'),
                    ]
                ),
                # Seconde ligne de cellules
                html.Div(
                    children=[
                    html.H3('Elements de revenus'),
                    html.H4( id='revenus'),
                    ]
                ),
                # 3ème ligne de cellules
                html.Div(
                    children=[
                    html.H3('Calcul du revenu par personne'),
                    html.H4( id='revenu_par_pers'),
                    ]
                ),
                # 4ème ligne de cellules
                html.Div(
                    children=[
                    html.H3('Elements relatifs à la demande de prêt'),
                    html.H4( id='demande_prêt'),
                    ]
                )
        ]      
    )

@app.callback(
    [Output('identification_client', 'children'), Output('revenus', 'children'),   Output('revenu_par_pers', 'children'), Output('demande_prêt', 'children')],
    Input('filtre_client', 'value')
)
def update_rows(Id_client):
    # Génére les différentes cellules contenant les valeurs du client
    data = description_client_2[description_client_2['SK_ID_CURR'] == Id_client]
    return generate_table(data, mask_identification), generate_table(data, mask_revenu), generate_table(data, mask_revenu_par_pers), generate_table(data, mask_dm_pret)
# FIN PAGE 1   

# Contenu page 2
### Rapppel du modèle exporté 
reglog_ponderation = load('D:\documents\(2) OPENCLASSROOMS\PROJET 7\data\\reglog1_pondération.joblib', mmap_mode=None)
### Preparation des donnees
# Rappel du pipe de préparation des variables
pipe_prep = load('D:\documents\(2) OPENCLASSROOMS\PROJET 7\data\\pipe_preparation_variable1.joblib', mmap_mode=None)
# Création de masque pour les callback
mask_scoring_proba = ['SCORE_PROBA']
mask_scoring_decision = ['DECISION']

# Définition du graphique affichant le top 20 des variables expplicatives selon l'importance de leurs coefficients
coefs = coefs.sort_values(by='|coeff|', ascending=False)
colors = ['risque diminué' if x < 0 else 'risque augmenté' for x in coefs.iloc[:20, 2]]
fig = px.bar(coefs.iloc[:20, 1], x= "|coeff|", orientation='h', color_discrete_sequence=px.colors.qualitative.Light24, color=colors)

# Définition de couleurs (rouge ou vert) pour afficher la décision de la banque
COLORS = [
    {
        'background': '#FFFACD',
        'text': '#008000' 
    },
    {
        'background': '#FFFACD',
        'text': '#FF0000'
    },
]

def cell_style(value):
    # défini une couleur selon l'argument d'entrée
    style = {}
    if value =='Le prêt est accordé':
        style = {
                'backgroundColor': COLORS[0]['background'],
                'color': COLORS[0]['text']
            }
    elif value == 'Le prêt est refusé':
        style = {
                'backgroundColor': COLORS[1]['background'],
                'color': COLORS[1]['text']
        }
    return style

def generate_table_colored(dataframe, cols, max_rows=1):
    # Génére les cellules avec du texte en couleur
    rows = []
    for i in range(min(len(dataframe), max_rows)):
        row = []
        for col in cols:
            value = dataframe.iloc[i][col]
            style = cell_style(value)
            row.append(html.Td(value, style=style))
        rows.append(html.Tr(row))

    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in cols])] +

        # Body
        rows)

# Affichage de la page 2
page_2_layout = html.Div(
    children=[
        # Titre de la page
        html.Div(
            children=[
                html.H1(
                    children="Probabilité de défaut de paiement", className="header-title"
                ),
                html.P(
                    children="décision de la banque sur la demande",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        # Liste déroulante pour sélectionner la référence client
        html.Div(
            children=[
                html.Div(children="Référence du client", className="menu-title"),
                dcc.Dropdown(
                    id='filtre_client',
                    options=[
                        {'label': i, 'value': i}
                        for i in np.sort(description_client_2['SK_ID_CURR'].unique())                            
                        ],
                    value=100001,
                    clearable=False,
                    persistence=True,
                    className="dropdown",
                ),
            ]
        ),
        # Première ligne de cellule
        html.Div(
            children=[
                html.H3('Probabilité de défaut du demandeur'),
                html.H4( id='défaut_client')
            ]
        ),
        # 2ème ligne de cellule
        html.Div(
            children=[
                html.H3("Décision de l'organisme bancaire"),
                html.H4(id='décision_banque')
                 #className="header-answer")
            ]
        ),
        # Graphique affichant le top 20 des variables expplicatives
        html.Div(
            children=[
                html.H3("Importance des critères de décision"),
                dcc.Graph(
                    id='critères_positifs',
                    figure=fig
                )
            ]
        )
    ]
)

@app.callback(
    [Output('défaut_client', 'children'), Output('décision_banque', 'children')],
    Input('filtre_client', 'value')
)
def update_rows(id_client):
    # 
    data = description_client_2[description_client_2['SK_ID_CURR'] == id_client]
    data_prep = pipe_prep.transform(data.iloc[:,1:-2])
    pred_model, Y_pred = fct.prediction_modele(reglog_ponderation, data_prep) # Y_pred donne les labels, pred_model donne la probailité d'appartenance à chaque classe
    data['SCORE_PROBA'] = pred_model[:, 1]
    data['SCORING'] = Y_pred
    data['SCORE_PROBA']=data['SCORE_PROBA'].apply(lambda x: np.round(x, 2))
    if data['SCORING'].any()==1:
        data['DECISION']='Le prêt est refusé'
    else :
        data['DECISION']='Le prêt est accordé'
    return generate_table(data, mask_scoring_proba), generate_table_colored(data, mask_scoring_decision)
# FIN PAGE 2

# CONTENU PAGE 3 
# classement ascendant des coefficients selon leur impportance
mask_col_imp=['AMT_GOODS_PRICE', 'AMT_CREDIT', 'EXT_SOURCE_3', 'POS_CNT_INSTALMENT_FUTURE_MEAN', 'BURO_CREDIT_TYPE_LIST_CLASSE', 'EXT_SOURCE_2', 
'ORGANIZATION_TYPE',  'POS_CNT_INSTALMENT_MEAN', 'CC_NAME_CONTRACT_STATUS_LIST_CLASSE', 'PREV_CHANNEL_TYPE_LIST_CLASSE', 'BURO_CREDIT_TYPE_LIST_CLASSE', 'POS_SK_ID_PREV_NUNIQUE',
 'AMT_ANNUITY', 'OCCUPATION_TYPE']

# figure initiale
fig1 = px.box(description_client_22, y="scoring")
fig2 = px.box(description_client_22, y="score_proba")

# Affichage de la page 3
page_3_layout = html.Div(
    children=[
        # Titre de la page
        html.Div(
            children=[
                html.H1(
                    children="Positionnement du demandeur par rapport à l'ensemble des demandeurs", className="header-title"
                ),
                html.P(
                    children="selon les principaux critères conduisant à la décision",
                    className="header-description",
                ),
            ],
            className="header",
        ),
        # Liste déroulante pour sélectionner la référence client
        html.Div(
            children=[
                html.Div(children="Référence du client", className="menu-title"),
                dcc.Dropdown(
                    id='filtre_client',
                    options=[
                        {'label': i, 'value': i}
                        for i in np.sort(description_client_22['SK_ID_CURR'].unique())                            
                        ],
                    value=100001,
                    clearable=False,
                    persistence=True,
                    className="dropdown",
                ),
            ]
        ),
        # Liste déroulante pour sélectionner les variables explicatives clées
        html.Div(
            children=[
                html.Div(children="Choix du critère d'étude", className="menu-title"),
                dcc.Dropdown(
                    id='filtre_variable',
                    options=[
                        {'label': i, 'value': i}
                        for i in mask_col_imp                            
                        ],
                    value='CODE_GENDER',
                    clearable=False,
                    className="dropdown",
                ),
            ]
        ),
        # Affichage de Pie Chart pour une modalité ou Jauge pour une numérique
        html.Div(
            children=[
                html.H3("Affichage de la valeur du critère étudié"),
                html.H4(id='valeur_chiffrée', style={'textAlign': 'center','color': 'green'}),
                dcc.Graph(id='graphe_modalité', figure=fig1)
            ],
            
        ),
        # Affichage d'un boxplot du score_proba selon la modalité considéré ou d'un boxplot de la variable numérique
        html.Div(
            children=[
                html.H3("Positionnement du demandeur selon le critère étudié"),
                dcc.Graph(id='boxplot', figure=fig2)
            ],
            
        ),
        
    ]
)

@app.callback(
    [Output('valeur_chiffrée', 'children'), Output('graphe_modalité', 'figure'), Output('boxplot', 'figure')],
    [Input('filtre_client', 'value'), Input('filtre_variable', 'value')]
)
def update_graph(id_client, var):
    # Affichage de pie chart, jauge et boxplot de la variable considérée pour un client donné
    data = description_client_22[description_client_22['SK_ID_CURR'] ==id_client]
    # distinction entre une variable catégorielle et une numérique
    if description_client_22[var].dtypes == 'object':
        fig1 = px.pie(description_client_22, names=var)
        fig2 = px.box(description_client_22, x=var, y="score_proba")
        modalite=data[var]
    else:
        fig1 = go.Figure(go.Indicator(
            domain = {'x': [0, 1], 'y': [0, 1]},
            value = float(data[var]),
            mode = "gauge+number+delta",
            title = {'text': "Valeur"},
            delta = {'reference': description_client_22[var].mean()},
            gauge = {'axis': {'range': [None, 1.05*(description_client_22[var].max())]},
                    'steps' : [
                        {'range': [0, description_client_22[var].median()], 'color': "lightgray"},
                        {'range': [description_client_22[var].median(), 1], 'color': "gray"}],
                    'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': description_client_22[var].mean()}}
            )
        )
        fig2 = px.box(description_client_2, y=var)
        modalite=None 
    return modalite, fig1, fig2
# FIN PAGE 3

# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    print(pathname)
    if pathname in ["/", "/page-1"]:
        return page_1_layout # html.Div("This is first page 1. YES!") 
    elif pathname == "/page-2":
        return page_2_layout # # html.P("This is the content of page 2. Yay!")
    elif pathname == "/page-3":
        return page_3_layout # html.P("Oh cool, this is page 3!")
    # If the user tries to reach a different page, return a 404 message
    return  dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=8086)