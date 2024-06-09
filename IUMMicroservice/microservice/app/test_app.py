import unittest
import tempfile
import os
import json
import pickle
import shutil
from main import app, predictions, data_load_lock, load_predictions, safe_replace_file


class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.temp_dir = tempfile.mkdtemp()

        # Create a temporary directory and files for testing
        os.makedirs(os.path.join(self.temp_dir, 'data/regression'), exist_ok=True)
        os.makedirs(os.path.join(self.temp_dir, 'data/nn'), exist_ok=True)

        with open(os.path.join(self.temp_dir, 'data/regression/top_10_country_regression.pkl'), 'wb') as f:
            pickle.dump(['song1', 'song2', 'song3'], f)

        with open(os.path.join(self.temp_dir, 'data/nn/top_10_country_nn.pkl'), 'wb') as f:
            pickle.dump(['song4', 'song5', 'song6'], f)

        app.config['DATA_DIR'] = self.temp_dir
        load_predictions()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_index_assigns_group(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('You have been assigned to the', response.data.decode())

    def test_get_data_invalid_genre(self):
        response = self.app.get('/invalid_genre')
        self.assertEqual(response.status_code, 404)

    def test_toggle_ab_test(self):
        response = self.app.post('/toggle_ab_test', data=json.dumps({'ab_test_enabled': False, 'default_group': 'nn'}),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data.decode())
        self.assertFalse(data['ab_test_enabled'])
        self.assertEqual(data['default_group'], 'nn')


if __name__ == '__main__':
    unittest.main()
