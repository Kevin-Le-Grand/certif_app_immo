import unittest
from fastapi.testclient import TestClient
from app import app
import boto3

class TestAPI(unittest.TestCase):
    client = TestClient(app)
    data={'SURFACE_BATI': 125, 
          'SURFACE_TERRAIN': 500,
        'prix_moyen_commune_m2': 1500.0, 
        }
    
    def test_reponse(self):
        """
        Vérifie que la réponse est correcte
        """
        reponse=self.client.post("/predict",
                     json=self.data)
        self.assertEqual(reponse.status_code,200)
        self.assertEqual(dict,type(reponse.json()))