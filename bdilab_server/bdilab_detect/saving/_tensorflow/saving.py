import logging
from pathlib import Path

import dill  # dispatch table setting not done here as done in top-level saving.py file
import tensorflow as tf
# from tensorflow.keras.layers import Input, InputLayer

from bdilab_server.bdilab_detect.cd.hdddm import HDDDMDrift

logger = logging.getLogger(__name__)


def save_optimizer_config(optimizer: tf.keras.optimizers.Optimizer):
    """

    Parameters
    ----------
    optimizer
        The tensorflow optimizer to serialize.

    Returns
    -------
    The tensorflow optimizer's config dictionary.
    """
    return tf.keras.optimizers.serialize(optimizer)


#######################################################################################################
# TODO: Everything below here is legacy saving code, and will be removed in the future
#######################################################################################################
def save_embedding_legacy(embed,
                          embed_args: dict,
                          filepath: Path) -> None:
    """
    Save embeddings for text drift models.

    Parameters
    ----------
    embed
        Embedding model.
    embed_args
        Arguments for TransformerEmbedding module.
    filepath
        The save directory.
    """
    # create folder to save model in
    if not filepath.is_dir():
        logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        filepath.mkdir(parents=True, exist_ok=True)

    # Save embedding model
    logger.info('Saving embedding model to {}.'.format(filepath.joinpath('embedding.dill')))
    embed.save_pretrained(filepath)
    with open(filepath.joinpath('embedding.dill'), 'wb') as f:
        dill.dump(embed_args, f)


def save_detector_legacy(detector, filepath):
    detector_name = detector.meta['name']

    # save metadata
    logger.info('Saving metadata and detector to {}'.format(filepath))

    with open(filepath.joinpath('meta.dill'), 'wb') as f:
        dill.dump(detector.meta, f)

    if isinstance(detector, HDDDMDrift):
        state_dict = state_hdddm(detector)
    else:
        raise NotImplementedError('The %s detector does not have a legacy save method.' % detector_name)

    with open(filepath.joinpath(detector_name + '.dill'), 'wb') as f:
        dill.dump(state_dict, f)


def state_hdddm(cd):
    state_dict = {
        'args':
            {
                'X_baseline': cd.X_baseline
            },
        'kwargs':
            {
                'gamma': cd.gamma,
                'alpha': cd.alpha,
                'use_mmd2': cd.use_mmd2,
                'use_k2s_test': cd.use_k2s_test,
            },
        'other':
            {

            }
    }
    return state_dict
