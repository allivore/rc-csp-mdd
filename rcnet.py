import numpy as np
from scipy.sparse import random as sparse_random
from scipy.sparse.linalg import eigs
from scipy.stats import zscore


def rc_activations(
    train_input_sequence, dynamics_length,
    approx_reservoir_size, degree, radius, worker_id,
    w_in_init, normalization=None, lr=1
):
    rng = np.random.default_rng(worker_id)
    N, input_dim = np.shape(train_input_sequence)
    if normalization == "zscore":
        train_input_sequence = zscore(train_input_sequence, axis=0)

    nodes_per_input = int(np.ceil(approx_reservoir_size/input_dim))
    reservoir_size = int(input_dim*nodes_per_input)
    sparsity = degree/reservoir_size
    W_h = sparse_random(reservoir_size, reservoir_size, density=sparsity, random_state=rng)
    eigenvalues = eigs(W_h, maxiter=1000000)[0]
    eigenvalues = np.abs(eigenvalues)
    W_h = (W_h/np.max(eigenvalues))*radius

    if w_in_init == "classic":
        W_in = np.zeros((reservoir_size, input_dim))
        q = int(reservoir_size/input_dim)
        for i in range(0, input_dim):
            W_in[i*q:(i+1)*q,i] = -1 + 2*rng.random(q)
    elif w_in_init == "random_30%":
        W_in = np.zeros((reservoir_size, input_dim))
        nconnections = int(reservoir_size*0.3)
        for channel in range(input_dim):
            connected_neurons = rng.choice(reservoir_size, nconnections)
            W_in[connected_neurons, channel] = rng.uniform(-1, 1)
    else:
        try:
            division = np.loadtxt(f"groups/{w_in_init}.txt")
        except FileNotFoundError:
            raise ValueError("Unknown W_in initialization method")
        W_in = np.zeros((reservoir_size, input_dim))
        q = int(reservoir_size/input_dim)
        for neuron in range(input_dim):
            group_channels = np.where(division == division[neuron])
            for group_channel in group_channels[0]:
                W_in[neuron*q:(neuron*q)+q, group_channel] = rng.uniform(-1, 1, size=q)
    # elif w_in_init == "first_2_words":
    #     division = np.loadtxt("groups/first_2_words.txt")
    #     W_in = np.zeros((reservoir_size, input_dim))
    #     for neuron in range(reservoir_size):
    #         group_channels = np.where(division == division[neuron])
    #         for group_channel in group_channels[0]:
    #             W_in[neuron, group_channel] = rng.uniform(-1, 1)
    # else:
    #     raise ValueError("Unknown W_in initialization method")

    tl = N - dynamics_length

    h = np.zeros((reservoir_size, 1))
    for t in range(dynamics_length):
        i = np.reshape(train_input_sequence[t], (-1,1))
        h = (1-lr)*h + lr*np.tanh(W_h @ h + W_in @ i)

    H = []

    for t in range(tl - 1):
        i = np.reshape(train_input_sequence[t+dynamics_length], (-1,1))
        h = (1-lr)*h + lr*np.tanh(W_h @ h + W_in @ i)
        h_aug = h.copy()
        h_aug[::2] = pow(h_aug[::2], 2.0)
        H.append(h_aug[:,0])

    return np.array(H)


def w_in_classic(reservoir_size: int, input_dim: int, rng: np.random.Generator):
    W_in = np.zeros((reservoir_size, input_dim))
    q = int(reservoir_size/input_dim)
    for i in range(0, input_dim):
        W_in[i*q:(i+1)*q,i] = -1 + 2*rng.random(q)
