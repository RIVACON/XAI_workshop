import logging
import warnings

logger = logging.getLogger('sloth')
try:
    import alibi
    alibi_installed = True
    from .anchors_alibi import AnchorsAlibi
except ModuleNotFoundError:
    alibi_installed = False
    logger.info('Python package alibi not installed, cannot compute anchors with this package.')
    warnings.warn('Python package alibi not installed, cannot compute anchors with this package.')
try:
    from .anchors_exp import AnchorsExp
    anchor_installed = True
except ModuleNotFoundError:
    anchor_installed = False
    logger.info('Python package anchor-exp not installed, cannot compute anchors with this package.')
    warnings.warn('Python package anchor-exp not installed, cannot compute anchors with ths package.')

if not(anchor_installed or alibi_installed):
    logger.info('Python packages alibi and anchor-exp not installed, cannot compute anchors with this package.')
    warnings.warn('Python packages alibi and anchor-exp not installed, cannot compute anchors with this package.')
else:
    from .anchors import Anchors