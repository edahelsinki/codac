import math

from clustering.a3s import A3S
from clustering.acdm import ACDM
from clustering.cobra import COBRA
from clustering.cobras import COBRAS
from clustering.codac import CODAC
from clustering.deep_a3s import DeepA3S
from clustering.deep_acdm import DeepACDM
from clustering.deep_cobra import DeepCOBRA
from clustering.deep_cobras import DeepCOBRAS
from clustering.deep_ffqs import DeepFFQS
from clustering.ffqs import FFQS
from data import (
    get_blood_mnist,
    get_fashion_mnist,
    get_handwritten,
    get_har,
    get_mnist,
    get_optdigits,
    get_pendigits,
    get_reuters,
    get_segmentation,
    get_usps,
    get_waveform,
    get_webkb,
)

# ratio of data points that sets the upper limit for queries
QUERY_RATIO = 0.12


def cobra_init_clusters_bound(Q, K):
    # Q max number of queries, K number of true clusters
    return math.ceil(Q / K + (K + 1) / 2)


_data_specs = {
    "optdigits": {"n": 5620, "d": 64, "k": 10, "query_sample_size": 100},
    "pendigits": {"n": 10992, "d": 16, "k": 10, "query_sample_size": 100},
    "har": {"n": 10299, "d": 561, "k": 6, "query_sample_size": 100},
    "mnist": {"n": 70000, "d": 784, "k": 10, "query_sample_size": 100},
    "usps": {"n": 9298, "d": 256, "k": 10, "query_sample_size": 100},
    "fashion": {"n": 70000, "d": 784, "k": 10, "query_sample_size": 100},
    "waveform": {"n": 5000, "d": 21, "k": 3, "query_sample_size": 100},
    "handwritten": {"n": 2000, "d": 76, "k": 10, "query_sample_size": 100},
    "bloodmnist": {"n": 17092, "d": 2352, "k": 8, "query_sample_size": 100},
    "reuters": {"n": 8367, "d": 2000, "k": 5, "query_sample_size": 100},
    "webkb": {"n": 4518, "d": 2000, "k": 6, "query_sample_size": 100},
    "segmentation": {"n": 2310, "d": 19, "k": 7, "query_sample_size": 100},

}

# calculate the max number of queries and number of initial cluster for COBRA
for dataset_name in _data_specs.keys():
    data_dict = _data_specs[dataset_name]
    max_queries = int(round(data_dict["n"] * QUERY_RATIO, -2))  # round to nearest 100
    _data_specs[dataset_name]["max_queries"] = max_queries
    _data_specs[dataset_name]["cobra_init_clusters"] = cobra_init_clusters_bound(
        max_queries, data_dict["k"]
    )
    # _data_specs[dataset_name]["ae_layers"] = [_data_specs[dataset_name]["d"], 512, 256, 128, _data_specs[dataset_name]["k"]] # feedforward autoencoder architecture
    _data_specs[dataset_name]["ae_layers"] = [_data_specs[dataset_name]["d"], 500, 500, 2000, _data_specs[dataset_name]["k"]] # feedforward autoencoder architecture

MODEL_LOADERS = {
    "COBRA": COBRA,
    "Deep-COBRA": DeepCOBRA,
    "CODAC": CODAC,
    "ACDM": ACDM,
    "Deep-ACDM": DeepACDM,
    "COBRAS": COBRAS,
    "Deep-COBRAS": DeepCOBRAS,
    "A3S": A3S,
    "Deep-A3S": DeepA3S,
    "FFQS": FFQS,
    "Deep-FFQS": DeepFFQS,
}

DATA_LOADERS = {
    "bloodmnist": get_blood_mnist,
    "fashion": get_fashion_mnist,
    "handwritten": get_handwritten,
    "har": get_har,
    "mnist": get_mnist,
    "optdigits": get_optdigits,
    "pendigits": get_pendigits,
    "usps": get_usps,
    "waveform": get_waveform,
    "reuters": get_reuters,
    "webkb": get_webkb,
    "segmentation": get_segmentation,
}

dataset_configs = {}
for dataset_name in _data_specs.keys():
    dataset_configs[dataset_name] = {
        "ACDM": {"query_sample_size": _data_specs[dataset_name]["query_sample_size"]},
        "Deep-ACDM": {
           "query_sample_size": _data_specs[dataset_name]["query_sample_size"]
        },
        "A3S": {"query_sample_size": _data_specs[dataset_name]["query_sample_size"]},
        "Deep-A3S": {"query_sample_size": _data_specs[dataset_name]["query_sample_size"]},
        "CODAC": {
            "n_clusters": _data_specs[dataset_name]["k"],
            "query_sample_size": _data_specs[dataset_name]["query_sample_size"],
            "max_queries": _data_specs[dataset_name]["max_queries"],
            "embedding_size": _data_specs[dataset_name]["k"],
        },
        "COBRA": {
            "query_sample_size": _data_specs[dataset_name]["query_sample_size"],
            "gt_n_clusters": _data_specs[dataset_name]["k"],
        },
        "Deep-COBRA": {
            "query_sample_size": _data_specs[dataset_name]["query_sample_size"],
            "gt_n_clusters": _data_specs[dataset_name]["k"],
        },
        "COBRAS": {
            "query_sample_size": _data_specs[dataset_name]["query_sample_size"],
            "max_queries": _data_specs[dataset_name]["max_queries"],
        },
        "Deep-COBRAS": {
            "query_sample_size": _data_specs[dataset_name]["query_sample_size"],
            "max_queries": _data_specs[dataset_name]["max_queries"],
        },
        "FFQS": {
            "n_clusters": _data_specs[dataset_name]["k"],
            "query_sample_size": _data_specs[dataset_name]["query_sample_size"],
            "max_queries": _data_specs[dataset_name]["max_queries"],
        },
        "Deep-FFQS": {
            "n_clusters": _data_specs[dataset_name]["k"],
            "query_sample_size": _data_specs[dataset_name]["query_sample_size"],
            "max_queries": _data_specs[dataset_name]["max_queries"],
        },
    }

DATASETS = {}
for dataset_name in _data_specs.keys():
    DATASETS[dataset_name] = {
        "name": dataset_name,
        "loader": DATA_LOADERS[dataset_name],
        "n_clusters": _data_specs[dataset_name]["k"],
        "max_queries": _data_specs[dataset_name]["max_queries"],
        "model_config": {
            "COBRA": dataset_configs[dataset_name]["COBRA"],
            "Deep-COBRA": dataset_configs[dataset_name]["Deep-COBRA"],
            "COBRAS": dataset_configs[dataset_name]["COBRAS"],
            "Deep-COBRAS": dataset_configs[dataset_name]["Deep-COBRAS"],
            "CODAC": dataset_configs[dataset_name]["CODAC"],
            "ACDM": dataset_configs[dataset_name]["ACDM"],
            "Deep-ACDM": dataset_configs[dataset_name]["Deep-ACDM"],
            "A3S": dataset_configs[dataset_name]["A3S"],
            "Deep-A3S": dataset_configs[dataset_name]["Deep-A3S"],
            "FFQS": dataset_configs[dataset_name]["FFQS"],
            "Deep-FFQS": dataset_configs[dataset_name]["Deep-FFQS"],
        },
    }