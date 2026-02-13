import codecs
import collections
import errno
import gzip
import json
import math
import os
import os.path
import pickle
import random
import urllib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from scipy.linalg import norm
from sklearn.cluster import KMeans

# from lib.utils import acc
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from torch.autograd import Variable
from torch.nn import Parameter


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    # urls = [
    #     'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    #     'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    #     'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    #     'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    # ]
    urls = [
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    @property
    def targets(self):
        if self.train:
            return self.train_labels
        else:
            return self.test_labels

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.use_cuda = torch.cuda.is_available()

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
            self.train_data = self.train_data.view(self.train_data.size(0), -1).float()*0.02
            # self.train_data = self.train_data.view(self.train_data.size(0), -1).float()/255
            self.train_labels = self.train_labels.int()
            if self.use_cuda:
                self.train_data = self.train_data.cuda()
                self.train_labels = self.train_labels.cuda()
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            self.test_data = self.test_data.view(self.test_data.size(0), -1).float()*0.02
            # self.test_data = self.test_data.view(self.test_data.size(0), -1).float()/255
            self.test_labels = self.test_labels.int()
            if self.use_cuda:
                self.test_data = self.test_data.cuda()
                self.test_labels = self.test_labels.cuda()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot']


class Reuters(data.Dataset):
    # To download the processed reuters data, please run the script named as "download_data.sh"
    training_file = "reutersidf10k_train.npy"
    test_file = "reutersidf10k_test.npy"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.use_cuda = torch.cuda.is_available()

        if download:
            self.download()

        if self.train:
            rtk10k_train = np.load(os.path.join(self.root, self.training_file)).item()
            self.train_data, self.train_labels = torch.tensor(rtk10k_train['data'], dtype=torch.float32), torch.tensor(rtk10k_train['label'], dtype=torch.int)
            if self.use_cuda:
                self.train_data = self.train_data.cuda()
                self.train_labels = self.train_labels.cuda()
        else:
            rtk10k_test = np.load(os.path.join(self.root, self.test_file)).item()
            self.test_data, self.test_labels = torch.tensor(rtk10k_test['data'], dtype=torch.float32), torch.tensor(
                rtk10k_test['label'], dtype=torch.int)
            if self.use_cuda:
                self.test_data = self.test_data.cuda()
                self.test_labels = self.test_labels.cuda()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)



def weights_xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias.data, 0)


class Dataset(data.Dataset):
    def __init__(self, data, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.labels = labels
        if torch.cuda.is_available():
            self.data = self.data.cuda()
            self.labels = self.labels.cuda()

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        # img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def masking_noise(data, frac):
    """
    data: Tensor
    frac: fraction of unit to be masked out
    """
    data_noise = data.clone()
    rand = torch.rand(data.size())
    data_noise[rand<frac] = 0
    return data_noise


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    # from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def detect_wrong(y_true, y_pred):
    """
    Simulating instance difficulty constraints. Require scikit-learn installed
    
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        A mask vector M =  1xn which indicates the difficulty degree
        We treat k-means as weak learner and set low confidence (0.1) for incorrect instances.
        Set high confidence (1) for correct instances.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    mapping_dict = {}
    for pair in ind:
        mapping_dict[pair[0]] = pair[1]
    wrong_preds = []
    for i in range(y_pred.size):
        if mapping_dict[y_pred[i]] != y_true[i]:
            wrong_preds.append(-0.1)   # low confidence -0.1 set for k-means weak learner
        else:
            wrong_preds.append(1)
    return np.array(wrong_preds)


def transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, n):
    """
    This function calculate the total transtive closure for must-links and the full entailment
    for cannot-links. 
    
    # Arguments
        ml_ind1, ml_ind2 = instances within a pair of must-link constraints
        cl_ind1, cl_ind2 = instances within a pair of cannot-link constraints
        n = total training instance number

    # Return
        transtive closure (must-links)
        entailment of cannot-links
    """
    ml_graph = dict()
    cl_graph = dict()
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in zip(ml_ind1, ml_ind2):
        add_both(ml_graph, i, j)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
    for (i, j) in zip(cl_ind1, cl_ind2):
        add_both(cl_graph, i, j)
        for y in ml_graph[j]:
            add_both(cl_graph, i, y)
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)
    ml_res_set = set()
    cl_res_set = set()
    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise Exception('inconsistent constraints between %d and %d' % (i, j))
            if i <= j:
                ml_res_set.add((i, j))
            else:
                ml_res_set.add((j, i))
    for i in cl_graph:
        for j in cl_graph[i]:
            if i <= j:
                cl_res_set.add((i, j))
            else:
                cl_res_set.add((j, i))
    ml_res1, ml_res2 = [], []
    cl_res1, cl_res2 = [], []
    for (x, y) in ml_res_set:
        ml_res1.append(x)
        ml_res2.append(y)
    for (x, y) in cl_res_set:
        cl_res1.append(x)
        cl_res2.append(y)
    return np.array(ml_res1), np.array(ml_res2), np.array(cl_res1), np.array(cl_res2)


def generate_random_pair(y, num):
    """
    Generate random pairwise constraints.
    """
    ml_ind1, ml_ind2 = [], []
    cl_ind1, cl_ind2 = [], []
    y = y.to(torch.device("cpu"))
    y = y.numpy()
    while num > 0:
        tmp1 = random.randint(0, y.shape[0] - 1)
        tmp2 = random.randint(0, y.shape[0] - 1)
        if tmp1 == tmp2:
            continue
        if y[tmp1] == y[tmp2]:
            ml_ind1.append(tmp1)
            ml_ind2.append(tmp2)
        else:
            cl_ind1.append(tmp1)
            cl_ind2.append(tmp2)
        num -= 1
    return np.array(ml_ind1), np.array(ml_ind2), np.array(cl_ind1), np.array(cl_ind2)


def generate_mnist_triplets(y, num):
    """
    Generate random triplet constraints
    """
    # To download the trusted_embedding for mnist data, run the script download_model.sh
    # Or you can create your own truseted embedding by running our pairwise constraints model
    # with 100000 randomly generated constraints.
    mnist_embedding = np.load("../model/mnist_triplet_embedding.npy")
    anchor_inds, pos_inds, neg_inds = [], [], []
    while num > 0:
        tmp_anchor_index = random.randint(0, y.shape[0] - 1)
        tmp_pos_index = random.randint(0, y.shape[0] - 1)
        tmp_neg_index = random.randint(0, y.shape[0] - 1)
        pos_distance = norm(mnist_embedding[tmp_anchor_index]-mnist_embedding[tmp_pos_index], 2)
        neg_distance = norm(mnist_embedding[tmp_anchor_index]-mnist_embedding[tmp_neg_index], 2)
        # 35 is selected by grid search which produce human trusted positive/negative pairs
        if neg_distance <= pos_distance + 35:
            continue
        anchor_inds.append(tmp_anchor_index)
        pos_inds.append(tmp_pos_index)
        neg_inds.append(tmp_neg_index)
        num -= 1
    return np.array(anchor_inds), np.array(pos_inds), np.array(neg_inds)


def generate_triplet_constraints_continuous(y, num):
    """
    Generate random triplet constraints
    """
    # To download the trusted_embedding for mnist data, run the script download_model.sh
    # Or you can create your own truseted embedding by running our pairwise constraints model
    # with 100000 randomly generated constraints.
    fashion_embedding = np.load("../model/fashion_triplet_embedding.npy")
    anchor_inds, pos_inds, neg_inds = [], [], []
    while num > 0:
        tmp_anchor_index = random.randint(0, y.shape[0] - 1)
        tmp_pos_index = random.randint(0, y.shape[0] - 1)
        tmp_neg_index = random.randint(0, y.shape[0] - 1)
        pos_distance = norm(fashion_embedding[tmp_anchor_index]-fashion_embedding[tmp_pos_index], 2)
        neg_distance = norm(fashion_embedding[tmp_anchor_index]-fashion_embedding[tmp_neg_index], 2)
        # 80 is selected by grid search which produce human trusted positive/negative pairs
        if neg_distance <= pos_distance + 80:
            continue
        anchor_inds.append(tmp_anchor_index)
        pos_inds.append(tmp_pos_index)
        neg_inds.append(tmp_neg_index)
        num -= 1
    return np.array(anchor_inds), np.array(pos_inds), np.array(neg_inds)


class MSELoss(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, input, target):
        return torch.mean((input-target)**2)


def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


class IDEC(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_clusters=10,
        encodeLayer=[400], decodeLayer=[400], activation="relu", dropout=0, alpha=1., gamma=0.1):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self.decoder = buildNetwork([z_dim] + decodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec = nn.Linear(decodeLayer[-1], input_dim)

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.gamma = gamma
        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)
        h = self.decoder(z)
        xrecon = self._dec(h)
        # compute q -> NxK
        q = self.soft_assign(z)
        return z, q, xrecon

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def encodeBatch(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        encoded = []
        self.eval()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch)
            z,_, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        kldloss = kld(p, q)
        return self.gamma*kldloss

    def recon_loss(self, x, xrecon):
        recon_loss = torch.mean((xrecon-x)**2)
        return recon_loss

    def pairwise_loss(self, p1, p2, cons_type):
        if cons_type == "ML":
            ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
            return ml_loss
        else:
            cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
            return cl_loss

    def global_size_loss(self, p, cons_detail):
        m_p = torch.mean(p, dim=0)
        m_p = m_p / torch.sum(m_p)
        return torch.sum((m_p-cons_detail)*(m_p-cons_detail))

    def difficulty_loss(self, q, mask):
        mask = mask.unsqueeze_(-1)
        mask = mask.expand(q.shape[0], q.shape[1])
        mask_q = q * mask
        diff_loss = -torch.norm(mask_q, 2)
        penalty_degree = 0.1
        return penalty_degree * diff_loss

    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def triplet_loss(self, anchor, positive, negative, margin_constant):
        # loss = max(d(anchor, negative) - d(anchor, positve) + margin, 0), margin > 0
        # d(x, y) = q(x) * q(y)
        negative_dis = torch.sum(anchor * negative, dim=1)
        positive_dis = torch.sum(anchor * positive, dim=1)
        margin = margin_constant * torch.ones(negative_dis.shape).cuda()
        diff_dis = negative_dis - positive_dis
        penalty = diff_dis + margin
        triplet_loss = 1*torch.max(penalty, torch.zeros(negative_dis.shape).cuda())

        return torch.mean(triplet_loss)

    def satisfied_constraints(self,ml_ind1,ml_ind2,cl_ind1, cl_ind2,y_pred):
        
        if ml_ind1.size == 0 or ml_ind2.size == 0 or cl_ind1.size == 0 or cl_ind2.size == 0:
            return 1.1

        count = 0
        satisfied = 0
        for (i, j) in zip(ml_ind1, ml_ind2):
            count += 1
            if y_pred[i] == y_pred[j]:
                satisfied += 1
        for (i, j) in zip(cl_ind1, cl_ind2):
            count += 1
            if y_pred[i] != y_pred[j]:
                satisfied += 1

        return float(satisfied)/count


    def predict(self, X):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        latent = self.encodeBatch(X)
        q = self.soft_assign(latent)

        # evalute the clustering performance
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        return y_pred

    def predict_(self, X, y):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        latent = self.encodeBatch(X)
        q = self.soft_assign(latent)

        # evalute the clustering performance
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        y = y.data.cpu().numpy()
        if y is not None:
            print("ari: %.5f, nmi: %.5f" % (adjusted_rand_score(y, y_pred), normalized_mutual_info_score(y, y_pred)))
            final_acc = adjusted_rand_score(y, y_pred)
            final_nmi = normalized_mutual_info_score(y, y_pred)
        return final_acc, final_nmi

    def fit(self,anchor, positive, negative, ml_ind1,ml_ind2,cl_ind1, cl_ind2, mask, use_global, ml_p, cl_p, X,y=None, lr=0.001, batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, use_kmeans=True, plotting="",clustering_loss_weight=1):    
        
        # save intermediate results for plotting
        intermediate_results = collections.defaultdict(lambda:{})

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("=====Training IDEC=======")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)

        if use_kmeans:
            print("Initializing cluster centers with kmeans.")
            kmeans = KMeans(self.n_clusters, n_init=20)
            data = self.encodeBatch(X)
            y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            y_pred_last = y_pred
            self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        else:
            # use kmeans to randomly initialize cluster ceters
            print("Randomly initializing cluster centers.")
            kmeans = KMeans(self.n_clusters, n_init=1, max_iter=1)
            data = self.encodeBatch(X)
            y_pred = kmeans.fit_predict(data.data.cpu().numpy())
            y_pred_last = y_pred
            self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))

        if y is not None:
            y = y.cpu().numpy()
            # print("Kmeans acc: %.5f, nmi: %.5f" % (acc(y, y_pred), normalized_mutual_info_score(y, y_pred)))
        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        ml_num_batch = int(math.ceil(1.0*ml_ind1.shape[0]/batch_size))
        cl_num_batch = int(math.ceil(1.0*cl_ind1.shape[0]/batch_size))
        tri_num_batch = int(math.ceil(1.0*anchor.shape[0]/batch_size))
        cl_num = cl_ind1.shape[0]
        ml_num = ml_ind1.shape[0]
        tri_num = anchor.shape[0]

        final_acc, final_nmi, final_epoch = 0, 0, 0
        update_ml = 1
        update_cl = 1
        update_triplet = 1
        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                # update the targe distribution p
                latent = self.encodeBatch(X)
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                if use_global:
                    y_dict = collections.defaultdict(list)
                    ind1, ind2 = [], []
                    for i in range(y_pred.shape[0]):
                        y_dict[y_pred[i]].append(i)
                    for key in y_dict.keys():
                        if y is not None:
                            print("predicted class: ", key, " total: ", len(y_dict[key]))
                            #, " mapped index(ground truth): ", np.bincount(y[y_dict[key]]).argmax())

                if y is not None:
                    print("ari: %.5f, nmi: %.5f" % (adjusted_rand_score(y, y_pred), normalized_mutual_info_score(y, y_pred)))
                    print("satisfied constraints: %.5f"%self.satisfied_constraints(ml_ind1,ml_ind2,cl_ind1, cl_ind2,y_pred))
                    final_acc = adjusted_rand_score(y, y_pred)
                    final_nmi = normalized_mutual_info_score(y, y_pred)
                    final_epoch = epoch

                # save model for plotting
                if plotting and (epoch in [10,20,30,40] or epoch%50 == 0 or epoch == num_epochs-1):
                    
                    df = pd.DataFrame(latent.cpu().numpy())
                    df["y"] = y
                    df.to_pickle(os.path.join(plotting,"save_model_%d.pkl"%(epoch)))
                    
                    intermediate_results["ari"][str(epoch)] = adjusted_rand_score(y, y_pred)
                    intermediate_results["nmi"][str(epoch)] = normalized_mutual_info_score(y, y_pred)
                    with open(os.path.join(plotting,"intermediate_results.json"), "w") as fp:
                        json.dump(intermediate_results, fp)

                # check stop criterion
                try:
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / num
                    y_pred_last = y_pred
                    if epoch>0 and delta_label < tol:
                        print('delta_label ', delta_label, '< tol ', tol)
                        print("Reach tolerance threshold. Stopping training.")

                        # save model for plotting
                        if plotting:
                            
                            df = pd.DataFrame(latent.cpu().numpy())
                            df["y"] = y
                            df.to_pickle(os.path.join(plotting,"save_model_%d.pkl"%epoch))
                            
                            intermediate_results["ari"][str(epoch)] = adjusted_rand_score(y, y_pred)
                            intermediate_results["nmi"][str(epoch)] = normalized_mutual_info_score(y, y_pred)
                            with open(os.path.join(plotting,"intermediate_results.json"), "w") as fp:
                                json.dump(intermediate_results, fp)
                        break
                except:
                    pass

            # train 1 epoch for clustering loss
            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            instance_constraints_loss_val = 0.0
            global_loss_val = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                mask_batch = mask[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch)
                target = Variable(pbatch)
                cons_detail = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
                #global_cons = torch.from_numpy(cons_detail).float().to("cuda")
                global_cons = torch.from_numpy(cons_detail).float().to(device)
                #global_cons = torch.from_numpy(cons_detail).float()

                z, qbatch, xrecon = self.forward(inputs)
                if use_global == False:
                    cluster_loss = self.cluster_loss(target, qbatch)
                    recon_loss = self.recon_loss(inputs, xrecon)
                    instance_constraints_loss = self.difficulty_loss(qbatch, mask_batch)
                    loss = cluster_loss + recon_loss + instance_constraints_loss
                    loss.backward()
                    optimizer.step()
                    cluster_loss_val += cluster_loss.data * len(inputs)
                    recon_loss_val += recon_loss.data * len(inputs)
                    instance_constraints_loss_val += instance_constraints_loss.data * len(inputs)
                    train_loss = clustering_loss_weight*cluster_loss_val + recon_loss_val + instance_constraints_loss_val
                else:
                    cluster_loss = self.cluster_loss(target, qbatch)
                    recon_loss = self.recon_loss(inputs, xrecon)
                    global_loss = self.global_size_loss(qbatch, global_cons)
                    loss = cluster_loss + recon_loss + global_loss
                    loss.backward()
                    optimizer.step()
                    cluster_loss_val += cluster_loss.data * len(inputs)
                    recon_loss_val += recon_loss.data * len(inputs)
                    train_loss = clustering_loss_weight*cluster_loss_val + recon_loss_val


            if instance_constraints_loss_val != 0.0:
                print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f Reconstruction Loss: %.4f Instance Difficulty Loss: %.4f"% (
                    epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num, instance_constraints_loss_val / num))
            elif global_loss_val != 0.0 and use_global:
                print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f Reconstruction Loss: %.4f Global Loss: %.4f"% (
                    epoch + 1, train_loss / num + global_loss_val/num_batch, cluster_loss_val / num, recon_loss_val / num, global_loss_val / num_batch))
            else:
                print("#Epoch %3d: Total: %.4f Clustering Loss: %.4f Reconstruction Loss: %.4f" % (
                    epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num))
            ml_loss = 0.0
            if epoch % update_ml == 0:
                for ml_batch_idx in range(ml_num_batch):
                    px1 = X[ml_ind1[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    px2 = X[ml_ind2[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    pbatch1 = p[ml_ind1[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx + 1)*batch_size)]]
                    pbatch2 = p[ml_ind2[ml_batch_idx*batch_size : min(ml_num, (ml_batch_idx+1)*batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    inputs2 = Variable(px2)
                    target1 = Variable(pbatch1)
                    target2 = Variable(pbatch2)
                    z1, q1, xr1 = self.forward(inputs1)
                    z2, q2, xr2 = self.forward(inputs2)
                    loss = (ml_p*self.pairwise_loss(q1, q2, "ML")+self.recon_loss(inputs1, xr1) + self.recon_loss(inputs2, xr2))
                    # 0.1 for mnist/reuters, 1 for fashion, the parameters are tuned via grid search on validation set
                    ml_loss += loss.data
                    loss.backward()
                    optimizer.step()

            cl_loss = 0.0
            if epoch % update_cl == 0:
                for cl_batch_idx in range(cl_num_batch):
                    px1 = X[cl_ind1[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx+1)*batch_size)]]
                    px2 = X[cl_ind2[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx+1)*batch_size)]]
                    pbatch1 = p[cl_ind1[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx + 1)*batch_size)]]
                    pbatch2 = p[cl_ind2[cl_batch_idx*batch_size : min(cl_num, (cl_batch_idx+1)*batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    inputs2 = Variable(px2)
                    target1 = Variable(pbatch1)
                    target2 = Variable(pbatch2)
                    z1, q1, xr1 = self.forward(inputs1)
                    z2, q2, xr2 = self.forward(inputs2)
                    loss = cl_p*self.pairwise_loss(q1, q2, "CL")
                    cl_loss += loss.data
                    loss.backward()
                    optimizer.step()

            if ml_num_batch >0 and cl_num_batch > 0:
                print("Pairwise Total:", round(float(ml_loss.cpu()), 2) + float(cl_loss.cpu()), "ML loss", float(ml_loss.cpu()), "CL loss:", float(cl_loss.cpu()))
            triplet_loss = 0.0
            if epoch % update_triplet == 0:
                for tri_batch_idx in range(tri_num_batch):
                    px1 = X[anchor[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx+1)*batch_size)]]
                    px2 = X[positive[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx+1)*batch_size)]]
                    px3 = X[negative[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx+1)*batch_size)]]
                    pbatch1 = p[anchor[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx + 1)*batch_size)]]
                    pbatch2 = p[positive[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx+1)*batch_size)]]
                    pbatch3 = p[negative[tri_batch_idx*batch_size : min(tri_num, (tri_batch_idx+1)*batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    inputs2 = Variable(px2)
                    inputs3 = Variable(px3)
                    target1 = Variable(pbatch1)
                    target2 = Variable(pbatch2)
                    target3 = Variable(pbatch3)
                    z1, q1, xr1 = self.forward(inputs1)
                    z2, q2, xr2 = self.forward(inputs2)
                    z3, q3, xr3 = self.forward(inputs3)
                    loss = self.triplet_loss(q1, q2, q3, 0.1)
                    triplet_loss += loss.data
                    loss.backward()
                    optimizer.step()
            if tri_num_batch > 0:
                print("Triplet Loss:", triplet_loss)
        return final_acc, final_nmi, final_epoch
