import os
import json
import hashlib
import datetime as dt


class UtilsClass:
    @staticmethod
    def set_config(method):
        """Loads information from config file and returns dictionary with parameter for a fiven method."""
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "config.json")
        )
        with open(config_path, "r") as f:
            config = json.load(f)
        return config[method]

    @staticmethod
    def hash_for_dict(data: dict):
        """Returns hash for given data."""
        return hashlib.sha1(json.dumps(data, cls=_JSONEncoder).encode()).hexdigest()


class _JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        """If given, converts date to correct format"""
        if isinstance(obj, (dt.date, dt.datetime)):#, pd.Timestamp)):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)

# class _JSONDecoder(json.JSONDecoder):
#     def __init__(self, *args, **kwargs):
#         json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
#
#     def object_hook(self, obj):
#         ret = {}
#         for key, value in obj.items():
#             if key in {'timestamp', 'whatever'}:
#                 ret[key] = dt.fromisoformat(value)
#             else:
#                 ret[key] = value
#         return ret