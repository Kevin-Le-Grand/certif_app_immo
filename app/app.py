import streamlit as st
from functions import *
from pages import *

#//////////////////////////////////////////////////////////////////////////////
#                          Main
#//////////////////////////////////////////////////////////////////////////////
def main():
    #*************************
    # Page d'authentification
    #*************************
    if 'username' not in st.session_state:
        authentification_page(conn)

    #*****************************************************************
    # Si l'authentification est réussie => accès au site de prédiction
    #*****************************************************************
    elif st.session_state.username:
        # Nettoyage de la page
        st.empty()
        # Affichage du logo dans la sidebar
        st.sidebar.markdown("""<div style="display: flex; justify-content: center; align-items: center;">
                            <img src="https://raw.githubusercontent.com/rastakoer/certif_app_immo/application/app/Logo.png" alt="Logo" width="200">
                            </div>""", unsafe_allow_html=True)
                   
        
        # État du formulaire False => formulaire non validé
        st.session_state.valid_formulaire=False
        st.session_state.commune="Sélectionnez une commune"
        st.session_state.departement="Sélectionnez un département"
        st.session_state.region="Sélectionnez une région"
        # Fonction pour remplir le formulaire
        formulaire(cursor)

        # Vérification que l'utilisateur a des droits 
        # + affichage d'un bouton permettant d'accéder à grafana
        cursor.execute(f"SELECT level FROM users WHERE username='{st.session_state.username}'")
        result = cursor.fetchone()
        if result is not None :
            if int(result[0]) == 1:
                st.sidebar.markdown("""
                    <div style="display: flex; justify-content: center; position: fixed; bottom: 0;" >
                        <a href="https://kevinlegrand.grafana.net/public-dashboards/cde8ec56de054eb295d3f68e0039aa63" target="_blank">
                            <button style="padding: 10px 20px;">Grafana</button>
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
        
        #----------------------------------------------------------------------------
        # Actions à produire en fonction de l'avancement du remplissage du formulaire 
        #----------------------------------------------------------------------------
        # Affichage de l'estimation
        if st.session_state.valid_formulaire:
            # Affichage de la page estimation
            formulaire_valide(cursor)
            # Enregistrement des données utilisateur et de la prédiction
            log_grafana()
            # Enregistrement en base de données de l'état de l'application
            log_status_grafana(0)
            
        # Affichage des statistiques sur la commune :
        elif st.session_state.commune!="Sélectionnez une commune":
            stat_commune(st.session_state.commune)
        # Affichage des statistiques sur le département :
        elif st.session_state.departement!="Sélectionnez un département":
            stat_departement(st.session_state.departement)
        # Affichage des statistiques sur la région
        elif st.session_state.region!="Sélectionnez une région":
            stat_region(st.session_state.region)
            
        # Affichage de la page d'accueil
        else:
            accueil()


if __name__ == "__main__":
    try:
        #///////////////////////////////////////////////////////////////////////////
        #                          Configuration de la page
        #///////////////////////////////////////////////////////////////////////////
        st.set_page_config(page_title="Immo App", page_icon=":rocket:", layout="wide")

        # Ajout du style CSS pour supprimer la marge haute
        st.markdown("""
        <style> .block-container {padding-top: 0;}
                .st-emotion-cache-16txtl3 {padding-top: 0;}
        </style>
        """, unsafe_allow_html=True)

        #///////////////////////////////////////////////////////////////////////////
        #  Connection à la base de données et création d'une table users si besoin
        #///////////////////////////////////////////////////////////////////////////
        # Connection avec la base de données
        conn = create_connection(os.environ['DB_HOST'],
                                os.environ['DB_USER'],
                                os.environ['DB_PASSWORD'],
                                int(os.environ['DB_PORT']),
                                "datagouv")
        cursor = conn.cursor()
        # Créer la table utilisateur si elle n'existe pas déjà
        # !!! Fonction commentée après avoir mis un utilisateur en lecture seule
        #create_user_table(conn,cursor)

        #////////////////////////////////////
        #       Programme principal
        #////////////////////////////////////
        main()
    except Exception as e:
        st.header(f"Une erreur s'est produite : {e}")
        log_crash_grafana(f"{e}")
        log_status_grafana(1)