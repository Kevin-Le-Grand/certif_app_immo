import unittest
import os
from functions import api_predict,create_connection

class TestAPICalls(unittest.TestCase):
    def test_create_connection(self):
        # Les informations de connection doivent être renseigné dans le dépôt GitHub
        host = os.environ['DB_HOST']
        user = os.environ['DB_USER']
        password = os.environ['DB_PASSWORD']
        port = int(os.environ['DB_PORT'])
        database = "datagouv"

        # Appel de la fonction à tester
        conn = create_connection(host, user, password, port, database)

        # Vérification que le cursor n'est pas None (indicatif d'une connexion réussie)
        self.assertIsNotNone(conn)

        # Fermeture de  la connexion après les tests
        conn.close()


    def test_api_predict_success(self):
        # Appel de la fonction de l'API avec des données type
        data = {"SURFACE_BATI": 150,
                "SURFACE_TERRAIN": 750,
                "prix_moyen_commune_m2": 1356}
        result = api_predict(data)

        # Vérification si le résultat est un dictionnaire
        self.assertEqual(dict, type(result))

        # Vérification que la clé 'reponse' est présente dans le JSON
        self.assertIn('reponse', result)

        # Vérification la valeur de la clé 'reponse' est de type float
        self.assertIsInstance(result['reponse'], float)

if __name__ == '__main__':
    unittest.main()
