import errno
import glob
import gzip
import json
import os
import pickle
import random
import shutil
import sys
import tarfile
import zipfile

import PIL
import six
from six.moves.urllib.error import HTTPError, URLError
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
import keras.backend as K
import numpy as np
import tensorflow as tf
from PIL import Image, ExifTags
from keras.layers import Dense, Activation
from keras.models import Model
from keras.preprocessing import image

from six.moves.urllib.request import urlopen

if sys.version_info[0] == 2:
    def urlretrieve(url, filename, reporthook=None, data=None):
        def chunk_read(response, chunk_size=8192, reporthook=None):
            content_type = response.info().get('Content-Length')
            total_size = -1
            if content_type is not None:
                total_size = int(content_type.strip())
            count = 0
            while True:
                chunk = response.read(chunk_size)
                count += 1
                if reporthook is not None:
                    reporthook(count, chunk_size, total_size)
                if chunk:
                    yield chunk
                else:
                    break

        response = urlopen(url, data)
        with open(filename, 'wb') as fd:
            for chunk in chunk_read(response, reporthook=reporthook):
                fd.write(chunk)
else:
    from six.moves.urllib.request import urlretrieve

def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.fawkes')
    datadir = os.path.join(datadir_base, cache_subdir)
    _makedirs_exist_ok(datadir)

    if untar:
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + '.tar.gz'
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if not os.path.exists(fpath):
        download = True

    if download:
        error_msg = 'URL fetch failure on {}: {} -- {}'
        dl_progress = None
        try:
            try:
                urlretrieve(origin, fpath, dl_progress)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)
            raise
        # ProgressTracker.progbar = None

    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format='tar')
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath


def load_extractor(name):
    model_dir = os.path.join(os.path.expanduser('~'), '.fawkes')
    os.makedirs(model_dir, exist_ok=True)
    model_file = os.path.join(model_dir, "{}.h5".format(name))
    emb_file = os.path.join(model_dir, "{}_emb.p.gz".format(name))
    os.environ["CUDA_VISIBLE_DEVICES"]="7"
    if os.path.exists(model_file):
        model = keras.models.load_model(model_file)
    else:
        print("Download models...")
        get_file("{}.h5".format(name), "http://mirror.cs.uchicago.edu/fawkes/files/{}.h5".format(name),
                 cache_dir=model_dir, cache_subdir='')
        model = keras.models.load_model(model_file)

    if not os.path.exists(emb_file):
        get_file("{}_emb.p.gz".format(name), "http://mirror.cs.uchicago.edu/fawkes/files/{}_emb.p.gz".format(name),
                 cache_dir=model_dir, cache_subdir='')

    if hasattr(model.layers[-1], "activation") and model.layers[-1].activation == "softmax":
        raise Exception(
            "Given extractor's last layer is softmax, need to remove the top layers to make it into a feature extractor")
    return model

def _makedirs_exist_ok(datadir):
    if six.PY2:
        # Python 2 doesn't have the exist_ok arg, so we try-except here.
        try:
            os.makedirs(datadir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    else:
        os.makedirs(datadir, exist_ok=True)  # pylint: disable=unexpected-keyword-arg



