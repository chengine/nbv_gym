"""Shadow Splat DataParser configuration file."""

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from shadow_splat.dataparser import ShadowSplatDataParserConfig

shadow_splat_dataparser = DataParserSpecification(config=ShadowSplatDataParserConfig())