"""
Manage JSON related utilities
"""
import json

class DictToJson():
    def __call__(self, d, json_path):
        """Saves dict of floats in json file
        Args:
            d: (dict) of float-castable values (np.float, int, float, etc.)
            json_path: (string) path to json file
        """
        with open(json_path, 'w') as f:
            # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
            d = {k: float(v) for k, v in d.items()}
            json.dump(d, f, indent=4)