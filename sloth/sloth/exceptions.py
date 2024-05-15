from sloth.features import FeatureDescription, DataType

class WrongDataType:
    @classmethod
    def check_equality(cls, feature, expected_type):
        if feature.data_type != expected_type.value:
            raise Exception('The data_type of feature ' + feature.name + ' (' + feature.data_type + ') does not equal the expected data_type ' + expected_type.value)
        
    @classmethod
    def check_inequality(cls, feature, expected_type):
        if feature.data_type == expected_type.value:
            raise Exception('The data_type of feature ' + feature.name + ' (' + feature.data_type + ') is not an allowed data_type.')
        
class FeatureDoesNotExist:
    @classmethod
    def check(cls, features: dict, feature_name: str):
        if feature_name not in features:
            raise WrongDataType('The feature ' + feature_name +' does not exist in given feature list')
        