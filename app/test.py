import unittest
from functions import api_predict

class TestAPICalls(unittest.TestCase):
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
