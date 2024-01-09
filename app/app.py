import streamlit as st
from functions import *

#//////////////////////////////////////////////////////////////////////////////
#                          Main
#//////////////////////////////////////////////////////////////////////////////
def main():
    # Page d'authentification
    if 'username' not in st.session_state:
        authentification_page(conn)

    # Si l'authentification est réussie => accès 
    elif st.session_state.username:
        st.empty()
        # Affichage du logo
        st.sidebar.markdown("""<div style="display: flex; justify-content: center; align-items: center;">
                            <img src="https://raw.githubusercontent.com/rastakoer/certif_app_immo/application/app/Logo.png" alt="Logo" width="200">
                            </div>""", unsafe_allow_html=True)

        # État du formulaire False => formulaire non validé
        st.session_state.valid_formulaire=False
        
        # Fonction pour remplir le formulaire
        formulaire(cursor)
        
        # Actions à produire si formulaire valide
        if st.session_state.valid_formulaire:
            # Appel de l'API
            prediction = api_predict({'SURFACE_BATI' :st.session_state.surface_bati,
                                    'NB_PIECES' : st.session_state.nb_pieces,
                                    'NAME_TYPE_BIEN':st.session_state.type_de_bien,
                                    'Name_region' : st.session_state.region})
            # Affichage de la prédiction
            st.title(f"Le bien est estimé à {int(prediction['reponse'])} €")

            # Affichage des ventes réalisées dans la commune
            st.components.v1.html(affichage_ventes_proximite([st.session_state.region,
                                                            st.session_state.departement,
                                                            st.session_state.commune],cursor), height=500)



if __name__ == "__main__":
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
    #         Connection à la base de données et création d'une table users
    #///////////////////////////////////////////////////////////////////////////
    # Connection avec la base de données
    conn = create_connection(os.environ['DB_HOST'],
                               os.environ['DB_USER'],
                               os.environ['DB_PASSWORD'],
                               int(os.environ['DB_PORT']),
                               "datagouv")
    cursor = conn.cursor()
    # Créer la table utilisateur si elle n'existe pas déjà
    create_user_table(conn,cursor)

    #////////////////////////////////////
    #       Programme principal
    #////////////////////////////////////
    main()