import unittest
import os
from functions import api_predict,create_connection

class TestAPICalls(unittest.TestCase):
    def test_create_connection(self):
        # Remplacez ces valeurs par les informations de votre base de données de test
        host = os.environ['DB_HOST']
        user = os.environ['DB_USER']
        password = os.environ['DB_PASSWORD']
        port = int(os.environ['DB_PORT'])
        database = "datagouv"

        # Appel de la fonction à tester
        conn = create_connection(host, user, password, port, database)

        # Vérification que le cursor n'est pas None (indicatif d'une connexion réussie)
        self.assertIsNotNone(conn)

        # Fermer la connexion après les tests
        conn.close()


    def test_api_predict_success(self):
        # Appel de la fonction de l'API avec des données factices
        data = {'SURFACE_BATI' :250,
                'NB_PIECES' : 5,
                'NAME_TYPE_BIEN': "Maison",
                'Name_region' : "Normandie"}
        result = api_predict(data)
        self.assertEqual(dict,type(result.json()))

if __name__ == '__main__':
    unittest.main()
