"""Custom dataparser for Shadow Splat"""

from dataclasses import dataclass

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio, NerfstudioDataParserConfig

# @dataclass
# class ShadowSplatDataParserConfig(NerfstudioDataParserConfig):
#     """Dataset config"""
#     # TODO: add more fields here

@dataclass
class ShadowSplatDataparserOutputs(DataparserOutputs):
    # TODO: add lighting info fields
    pass


class ShadowSplatDataParser(Nerfstudio):
    """Shadow Splat DatasetParser"""

    config: NerfstudioDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):
        # Call parent method
        super()._generate_dataparser_outputs(split)
