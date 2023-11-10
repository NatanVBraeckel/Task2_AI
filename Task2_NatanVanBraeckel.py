import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import category_encoders as ce

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor

import streamlit as st
import os

script_path = os.path.dirname(os.path.realpath(__file__))

students_math_df = pd.read_csv(os.path.join(script_path, "student-mat.csv"), sep=";")
students_por_df = pd.read_csv(os.path.join(script_path, "student-por.csv"), sep=";")
students_por_df_removed_0 = students_por_df[students_por_df['G3'] != 0]

feature_cols = ['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences']

X_with_0 = students_por_df[feature_cols]
y_with_0 = students_por_df[['G3']]

X_rm_0 = students_por_df_removed_0[feature_cols]
y_rm_0 = students_por_df_removed_0[['G3']]

ce_ord = ce.OrdinalEncoder(cols = feature_cols)
X_cat_with_0 = ce_ord.fit_transform(X_with_0)
X_cat_rm_0 = ce_ord.fit_transform(X_rm_0)

def print_performance_metrics(test, predict):
    # performance metrics of dataset with 0
    st.write(f"<p style='font-size: 25px; display: inline;'>RMSE: </h3> <p style='display: inline; font-size: 25px; color: green;'>{round(np.sqrt(metrics.mean_squared_error(test, predict)), 2)}</p>", unsafe_allow_html=True)

    st.write("")
    st.write("MAE".ljust(33), ":", round(metrics.mean_absolute_error(test, predict), 2))
    st.write("MSE".ljust(33), ":", round(metrics.mean_squared_error(test, predict), 2))
    st.write("R2".ljust(33), ":", round(metrics.r2_score(test, predict), 2))

def linear_regression(random_state= None):
    if(random_state == "geen"):
        X_train_with_0, X_test_with_0, y_train_with_0, y_test_with_0 = train_test_split(X_cat_with_0, y_with_0, test_size=0.2)
        X_train_rm_0, X_test_rm_0, y_train_rm_0, y_test_rm_0 = train_test_split(X_cat_rm_0, y_rm_0, test_size=0.2)
    else:
        X_train_with_0, X_test_with_0, y_train_with_0, y_test_with_0 = train_test_split(X_cat_with_0, y_with_0, test_size=0.2, random_state=random_state)
        X_train_rm_0, X_test_rm_0, y_train_rm_0, y_test_rm_0 = train_test_split(X_cat_rm_0, y_rm_0, test_size=0.2, random_state=random_state)

    # model with 0
    model = LinearRegression()
    model.fit(X_train_with_0, y_train_with_0)

    # model without 0
    model_rm_0 = LinearRegression()
    model_rm_0.fit(X_train_rm_0, y_train_rm_0)

    y_pred_with_0 = model.predict(X_test_with_0)

    y_pred_rm_0 = model_rm_0.predict(X_test_rm_0)

    lr_cols = st.columns(2)

    with lr_cols[0]:
        st.subheader("Dataset met 0 scores")
        st.write('Intercept:')
        st.write(str(model.intercept_))
        st.write('Coëfficiënten:')
        st.write(str(model.coef_))

        compare_df = pd.DataFrame({'Actueel': y_test_with_0.values.flatten(), 'Voorspeld': y_pred_with_0.flatten()})
        st.write("Een aantal voorspellingen")
        st.table(compare_df.head(21))

        print_performance_metrics(y_test_with_0, y_pred_with_0)

    with lr_cols[1]:
        st.subheader("Dataset zonder 0 scores")
        st.write('Intercept')
        st.write(str(model_rm_0.intercept_))
        st.write('Coëfficiënten')
        st.write(str(model_rm_0.coef_))

        compare_df_rm_0 = pd.DataFrame({'Actueel': y_test_rm_0.values.flatten(), 'Voorspeld': y_pred_rm_0.flatten()})
        st.write("Een aantal voorspellingen")
        st.table(compare_df_rm_0.head(21))

        print_performance_metrics(y_test_rm_0, y_pred_rm_0)

def lasso(alpha, random_state = None):
    X_train_rm_0, X_test_rm_0, y_train_rm_0, y_test_rm_0 = train_test_split(X_cat_rm_0, y_rm_0, test_size=0.2, random_state=random_state)

    lasso_regressor = Lasso(alpha=alpha)
    lasso_regressor.fit(X_train_rm_0, y_train_rm_0)

    # map the features to the lasso coefficients
    feature_coefficients = dict(zip(feature_cols, lasso_regressor.coef_))

    # sort the features based on the coefficients
    sorted_coefficients = sorted(feature_coefficients.items(), key=lambda x: abs(x[1]), reverse=True)

    cols = st.columns([.7,.3])

    with cols[0]:
        # print the non-zero coefficients for each feature
        st.write("Features en hun coëfficiënten (geordend op grootte)")
        for feature, coefficient in sorted_coefficients:
            if coefficient != 0:
                st.write(f"{feature}: {coefficient}")

    with cols[1]:
        y_pred_lasso = lasso_regressor.predict(X_test_rm_0)

        print_performance_metrics(y_test_rm_0, y_pred_lasso)

def k_neighbors_regressor(k=5, metric="euclidian", random_state=None):

    X_train_rm_0, X_test_rm_0, y_train_rm_0, y_test_rm_0 = train_test_split(X_cat_rm_0, y_rm_0, test_size=0.2, random_state=random_state)

    k_neighbors_regressor = KNeighborsRegressor(n_neighbors=k, metric=metric)
    k_neighbors_regressor.fit(X_train_rm_0, y_train_rm_0)

    y_pred_knr = k_neighbors_regressor.predict(X_test_rm_0)

    print_performance_metrics(y_test_rm_0, y_pred_knr)

def comparison(alpha, k, metric, random_state=None):
    X_train_rm_0, X_test_rm_0, y_train_rm_0, y_test_rm_0 = train_test_split(X_cat_rm_0, y_rm_0, test_size=0.2, random_state=random_state)

    #linear regression
    model_rm_0_lr = LinearRegression()
    model_rm_0_lr.fit(X_train_rm_0, y_train_rm_0)
    y_pred_rm_0_lr = model_rm_0_lr.predict(X_test_rm_0)

    #lasso
    lasso_regressor = Lasso(alpha=alpha)
    lasso_regressor.fit(X_train_rm_0, y_train_rm_0)
    y_pred_lasso = lasso_regressor.predict(X_test_rm_0)

    #kneighborsregressor
    k_neighbors_regressor = KNeighborsRegressor(n_neighbors=k, metric=metric)
    k_neighbors_regressor.fit(X_train_rm_0, y_train_rm_0)
    y_pred_knr = k_neighbors_regressor.predict(X_test_rm_0)

    cols = st.columns(3)
    with cols[0]:
        st.write("<p style='font-size: 28px; color: #FF4B4B'>Linear regression<p/>", unsafe_allow_html=True)
        print_performance_metrics(y_test_rm_0, y_pred_rm_0_lr)
    with cols[1]:
        st.write("<p style='font-size: 28px; color: #FF4B4B'>Lasso<p/>", unsafe_allow_html=True)
        print_performance_metrics(y_test_rm_0, y_pred_lasso)
    with cols[2]:
        st.write("<p style='font-size: 28px; color: #FF4B4B'>KNeighborSelector<p/>", unsafe_allow_html=True)
        print_performance_metrics(y_test_rm_0, y_pred_knr)

st.title(":red[Machine Learning Benchmarking]", anchor=False)

tab_intro, tab_linear, tab_lasso, tab_knr, tab_comparison = st.tabs(["Introductie", "Linear Regression", "Lasso", "KNeighborsRegressor", "Vergelijking"])

with tab_intro:
    # Display basic information
    # st.markdown("# Exploratory Data Analysis")

    # Display features and observations information
    st.subheader("Introductie")

    st.write("Mijn dataset gaat over scores die studenten halen op een vak.")
    st.write("In de link waren er 2 datasets gegeven, 1 voor math en 1 voor portugees.")

    st.write(f"#### Features en Observaties")

    st.write(f"Er zijn {students_math_df.shape[1]} kolommen in beide datasets.")
    st.write(f"Er zijn {students_math_df.shape[0]} observaties in the math dataset.")
    st.write(f"Er zijn {students_por_df.shape[0]} observaties in the portuguese dataset.")

    # Display Features information
    st.write("### Features")

    feature_table = """
    |  # 	| Feature | Betekenis	| Mogelijke waarden |
    |---	|---	|---	|--- |
    |   1	|   school	|   naam van school	| 'GP' - Gabriel Pereira <br /> 'MS' - Mousinho da Silveira  |
    |   2	|   sex	| gender | 'F' - female <br /> 'M' - male |
    |   3	|  age 	| leeftijd  | van 15 t.e.m 22 |
    |   4	|   address	| type adres | 'U' - urban <br /> 'R' - rural |
    |   5	|   famsize	|  familiegrootte | 'LE3' - <= 3 <br /> 'GT3' - >3 |
    |   6	|   Pstatus	|  status ouders | 'T' - together <br /> 'A' - apart |
    |   7	|   Medu	|  moeders educatie | 0 - geen <br /> 1 - 'primary education' (4th grade) <br /> 2 - 'primary education' (5th - 9th grade) <br /> 3 - 'secondary education' <br /> 4 - 'higher education' |
    |   8	|   Fedu	|  vaders educatie	| ^^  |
    |   9	|   Mjob	|  moeders job 	| "teacher" - leerkracht <br /> "health" - zorgsector <br /> "services" - ambtenaar <br /> "at_home" <br /> "other" |
    |   10	|   Fjob	|  vaders job 	| ^^ |
    |   11	|   reason	|  redenering achter schoolkeuze 	| "home" - dichtbij huis <br /> "reputation" - reputatie van de school <br /> "course" - bepaalde cursus <br /> "other" |
    |   12	|   guardian	|   voogd	| "mother" <br /> "father" <br /> "other" | 
    |   13	|   traveltime	|   pendeltijd	| 1 - < 15min <br /> 2 - 15-30min <br /> 3 - 30-60min <br /> 4 - 60min  |
    |   14	|   studytime	|   wekelijkse studietijd	| 1 - < 2u <br /> 2 - 2u-5u <br /> 3 - 5u-10u <br /> 4 - 10u |
    |   15	|   failures	|   aantal vorige failures van het vak	| maximum van 4 |
    |   16	|   schoolsup	|   extra educationele steun	| "yes" / "no" |
    |   17	|   famsup	|   familie educationele steun	| "yes" / "no" |
    |   18	|   paid	|   extra betaalde cursussen binnen het vak	| "yes" / "no" |
    |   19	|   activities	|   extra-curriculaire activiteiten	| "yes" / "no" |
    |   20	|   nursery	|   meegedaan aan kleuterschool	| "yes" / "no" |
    |   21	|   higher	|   interesse in hoger onderwijs	| "yes" / "no" |
    |   22	|   internet	|   thuis toegang tot internet	| "yes" / "no" |
    |   23	|   romantic	|   in een relatie	| "yes" / "no" |
    |   24	|   famrel	|   kwaliteit van familiale relaties	|  vanaf 1 - heel laag <br /> t.e.m. 5 - heel hoog |
    |   25	|   freetime	|   mate van vrije tijd buiten school	| vanaf 1 - heel laag <br /> t.e.m. 5 - heel hoog |
    |   26	|   goout	|   mate van uitgaan met vrienden	| vanaf 1 - heel laag <br /> t.e.m. 5 - heel hoog |
    |   27	|   Dalc	|   mate alcoholconsumptie schooldagen	| vanaf 1 - heel laag <br /> t.e.m. 5 - heel hoog |
    |   28	|   Walc	|   mate alcoholconsumptie weekenddagen	| vanaf 1 - heel laag <br /> t.e.m. 5 - heel hoog |
    |   29	|   health	|   mate van huidige gezondheid	| vanaf 1 - heel slecht <br /> t.e.m. 5 - heel goed |
    |   30	|   absences	|   hoeveelheid afwezigheden	| van 0 t.e.m. 93 |

   """

    st.markdown(feature_table, unsafe_allow_html=True)

    st.write("")
    st.write("Alle features leken mij wel nuttig in het voorspellen van een score, dus ik gebruik ze allemaal.")
    st.write("Later maak ik gebruik van een Lasso, een techniek die relevantie van features beter aantoont.")

    # Display Scores information
    st.write("### Scores - deze kunnen gebruikt worden als label.")

    label_table = """
    | # | Label | Betekenis | Mogelijke waarden |
    |--- |--- |--- |--- |
    | 31 | G1 | score eerste semester | van 0 t.e.m. 20 |
    | 32 | G2 | score tweede semester | van 0 t.e.m. 20 |
    | 33 | G3 | uiteindelijke score | van 0 t.e.m. 20 |

    """

    st.markdown(label_table, unsafe_allow_html=True)

    st.write("")
    st.write("Ik heb ervoor gekozen alleen G3 te gebruiken als label, dat vond ik duidelijker.")

    # Display null values information
    st.write(f"## Null Waarden")
    st.write(f"Er zijn {students_math_df.isnull().sum().sum() + students_por_df.isnull().sum().sum()} null waarden in beide datasets.")

    # Display scores of 0 information
    st.write(f"Aantal 0 scores:")
    st.markdown(f"- 0 scores voor math: {(students_math_df['G3'] == 0).sum()}")
    st.markdown(f"- 0 scores voor portuguese: {(students_por_df['G3'] == 0).sum()}")


    st.write(f"## Portugees of Math?")
    st.markdown("""
        Ik heb ervoor gekozen om door te gaan met de dataset voor scores voor portugees omdat er meer observaties zijn.\
        Ook zijn er minder finale scores van 0 in die dataset.

        Ik was aan het twijfelen om de datasets met elkaar te mergen, maar ik stootte op een probleem. Dus ik vraagde advies aan ChatGPT.
        #### De prompt die ik heb gegeven:

        *There are two datasets at my disposal, one for math and one for portuguese.\
        Is it possible to combine the two datasets?\
        Or would it mess with the data since the same student could be in the dataset twice?*

        #### Het begin van het antwoord dat ik heb gekregen:

        *Combining the datasets for math and Portuguese can be done, but it's important to consider how the data is structured and whether it makes sense to merge them. If the datasets share a common identifier, such as a student ID, you can merge them based on that identifier. However, as you pointed out, if the same student appears in both datasets, merging could lead to duplicate entries for that student.*

        Ik ga de datasets dus niet mergen omdat er geen unieke identifiers zijn voor studenten.""")

    st.write("")
    # bar chart
    st.write("## Distribution van Scores")

    plt.figure(figsize=(8, 6))
    score_counts = students_por_df['G3'].value_counts().sort_index()
    score_range = range(21)
    plt.bar(score_range, [score_counts.get(score, 0) for score in score_range], color="orange")
    plt.xlabel('Score')
    plt.ylabel('Aantal')
    plt.title('Distributie van scores')
    plt.xticks(score_range)
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    st.pyplot(plt)

    st.markdown("""
        Ik heb 2 opties omtrent de 0 scores:
        - De 0 scores behouden - misschien hebben de features effect op een 0 score. Zijnde niet meedoen of gewoon 0 scoren.
        - De 0 scores amputeren - studenten die niet meegedaan hebben aan het examen zouden waarschijnlijk wel hoger gescoord hebben dan een 0, waardoor de data incorrect is voor die studenten.

        Ik heb ervoor gekozen om de eerste ML techniek toe te passen op beide versies van de dataset.\
        Daar zal ik dan degene met het beste resultaat uit kiezen en die gebruiken in de extra ML technieken.
    """)

with tab_linear:
    st.subheader("Linear Regression")

    cols = st.columns(3)
    with cols[0]:
        random_state = st.text_input("Random state (niet verplicht)", key=1)

    if st.button('Bereken Linear Regression'):
        if random_state:
            try:
                random_state = int(random_state)
            except ValueError:
                st.error("Random state moet een geheel getal zijn")
                st.stop()
        else:
            random_state = None 
        linear_regression(random_state=random_state)

with tab_lasso:
    st.subheader("Lasso")

    with st.expander("Info"):
        st.write("Lasso maakt het voorspellen efficiënter door aan automatische feature selectie te doen.")
        st.write("Coëfficiënten voor features worden aangepast door penalties. Een penalty zal zorgen dat de grootte van de coëfficiënt voor een feature wordt verkleint (dichter bij 0).")
        st.write(" De grootte van de penalties die worden uitgereikt kunnen aangepast worden door alpha factor. Grotere alpha = grotere penalty.")
        st.write("Zo doet de techniek dus aan feature selection, omdat veel van de coëfficiënten van de features naar 0 zullen worden herleidt en dus geef effect meer zullen hebben (x*0 = 0).")

    cols = st.columns(3)
    with cols[0]:
        alpha = st.number_input("Alpha (grootte penalty)", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key=9)
    with cols[1]:
        random_state_lasso = st.text_input("Random state (niet verplicht)", key=2)

    if st.button('Bereken Lasso'):
        if random_state_lasso:
            try:
                random_state_lasso = int(random_state_lasso)
            except ValueError:
                st.error("Random state moet een geheel getal zijn")
                st.stop()
        else:
            random_state_lasso = None 
        lasso(alpha, random_state=random_state_lasso)

with tab_knr:
    st.subheader("KNeighborsRegressor")

    with st.expander("Info"):
        st.write("KNeighborsRegressor maakt zoals de naam al zegt gebruik van neighbors in de dataset.")
        st.write("Als er een nieuwe voorspelling gemaakt moet worden, zal deze techniek zich baseren op observaties die op deze nieuwe datapoint lijken.")
        st.write("Het ondergaat dus eigenlijk niet een training fase, omdat het gewoon de dichtsbijzijnde observaties gebruikt.")
        st.write("")
        st.write("Het aantal neighbors waarnaar gekeken zal worden kan aangepast worden met de \"k\" parameter.")
        st.write("")
        st.write("Hoe de afstand wordt berekend kan ook aangepast worden met de metric parameter")
        st.write("Default is deze euclidian, maar er zijn nog veel meer opties, waaronder bekende manieren zoals manhatten of cosine similarity.")

    cols = st.columns(3)
    with cols[0]:
        k = st.number_input("k (aantal buren)", min_value=2, max_value=100, value=5, step=1, key=11)
    with cols[1]:
        metric_options = ['euclidean', 'manhattan', 'cosine']
        metric = st.selectbox("Selecteer een afstandsmeting: ", metric_options, index=0, key=12)
    with cols[2]:
        random_state_knr = st.text_input("Random state (niet verplicht)", key=3)

    if st.button('Bereken KNeighborsRegressor'):
        if random_state_knr:
            try:
                random_state_knr = int(random_state_knr)
            except ValueError:
                st.error("Random state moet een geheel getal zijn")
                st.stop()
        else:
            random_state_knr = None 
        k_neighbors_regressor(k=k, metric=metric, random_state=random_state_knr)

with tab_comparison:
    st.subheader("Vergelijking")
    st.write("Vergelijk de drie Machine Learning technieken.")
    st.write("Elke techniek gebruikt dezelfde dataset (geen 0 scores) met dezelfde random state.")

    st.markdown("<hr>", unsafe_allow_html=True)

    cols = st.columns(3)
    with cols[0]:
        st.write("Alle technieken:")
        random_state_comp = st.text_input("Random state (niet verplicht)", key=4)
    with cols[1]:
        st.write("Lasso:")
        alpha_comp = st.number_input("Alpha (grootte penalty)", min_value=0.0, max_value=1.0, value=0.5, step=0.1, key=10)
    with cols[2]:
        st.write("KNeighborsRegressor")
        k_comp = st.number_input("k (aantal buren)", min_value=2, max_value=100, value=5, step=1, key=14)
        metric_options = ['euclidean', 'manhattan', 'cosine']
        metric_comp = st.selectbox("Selecteer een afstandsmeting: ", metric_options, index=0, key=15)


    if st.button('Bereken'):
        if random_state_comp:
            try:
                random_state_comp = int(random_state_comp)
            except ValueError:
                st.error("Random state moet een geheel getal zijn")
                st.stop()
        else:
            random_state_comp = None 
        comparison(alpha=alpha_comp, k=k_comp, metric=metric_comp, random_state=random_state_comp)