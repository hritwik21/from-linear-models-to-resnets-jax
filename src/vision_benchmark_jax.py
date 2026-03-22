"""JAX vision benchmark comparing linear models, MLPs, CNNs, VGG-style networks, and ResNet-style networks on grayscale CIFAR-10.
This script is extracted from the original experiment notebook and keeps the end-to-end training flow intact.
"""

import numpy as np
import jax
import pickle
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import lax

import os, tarfile, requests
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE   = "cifar-10-python.tar.gz"
EXTRACTED = "cifar-10-batches-py"

if not os.path.exists(ARCHIVE):
    print(f"downloading {ARCHIVE} …")
    with requests.get(CIFAR_URL, stream=True) as r:
        r.raise_for_status()
        with open(ARCHIVE, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk: f.write(chunk)
    print("Download dataset is now complete.")

if not os.path.isdir(EXTRACTED):
    print(f"extracting {ARCHIVE} …")
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        tf.extractall(".")
    print(f"extracted to ./{EXTRACTED}")

DATA_DIR = EXTRACTED

def load_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    data = batch['data'].reshape(-1, 3, 32, 32).astype(np.float32)
    labels = np.array(batch['labels'], dtype=np.int64)
    return data, labels

def rgb_to_gray(x):
    r = x[:, 0, :, :]
    g = x[:, 1, :, :]
    b = x[:, 2, :, :]
    gray = 0.299*r+0.587*g+0.114*b
    return gray

def load_gray_data(dir):
    train_imgs, train_labels = [],[]
    
    # training data ->
    for i in range(1, 6):
        data, labels = load_batch(os.path.join(dir, f"data_batch_{i}"))
        train_imgs.append(data)
        train_labels.append(labels)
    train_imgs = np.concatenate(train_imgs, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    # test data ->
    test_imgs, test_labels = load_batch(os.path.join(dir, "test_batch"))
    
    # convert both test and train data to grayscale ->
    train_gray = rgb_to_gray(train_imgs)
    test_gray  = rgb_to_gray(test_imgs)
    
    # normalize the data ->
    train_x = (train_gray/255.0).reshape(-1,32*32)
    test_x  = (test_gray/255.0).reshape(-1,32*32)
    print(f"Size of training data is {train_x.shape}")
    print(f"Size of test data is {test_x.shape}")
    return train_x, train_labels, test_x, test_labels

def plot_errors(train_errors, test_errors, title):
    epochs = np.arange(1, len(train_errors) + 1)
    plt.figure()
    plt.plot(epochs, train_errors, label="Training Error")
    plt.plot(epochs, test_errors, label="Test Error")
    plt.title(title)
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

train_x, train_y, test_x, test_y = load_gray_data(DATA_DIR)

# Single Layer Perceptron Table 2 Scenario
def init_linear_params(key,in_dimension=1024,out_dimension=10):
    key, subkey = jax.random.split(key)
    W = jax.random.normal(key, (in_dimension,out_dimension))*(1.0/jnp.sqrt(in_dimension))
    b = jnp.zeros((out_dimension,))
    return W, b

def linear_forward_pass(W, b, x):
    logits = jnp.dot(x, W) + b
    return logits

def cross_entropy_loss(W, b, x, y):
    logits = linear_forward_pass(W, b, x)
    log_prob = jax.nn.log_softmax(logits,axis=-1)
    neg_log_likelihood = -log_prob[jnp.arange(x.shape[0]),y]
    return neg_log_likelihood.mean()

def accuracy(W, b, x, y):
    logits = linear_forward_pass(W, b, x)
    preds = jnp.argmax(logits,axis=-1)
    return (preds==y).mean()

def mini_batch(X, y, batch_size=64, rng=None):
    N = X.shape[0]
    if rng is None:
        index = np.arange(N)
    else:
        index = np.array(rng.permutation(N))
    for left in range(0, N, batch_size):
        right = left+batch_size
        batch_index=index[left:right]
        yield X[batch_index],y[batch_index]

# Table 2 Single Layer Perceptron - SGD
def train_sgd_step(W, b, x, y, lr):
    def loss_fn(W_, B_):
        return cross_entropy_loss(W_, B_, x, y)
    loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(W, b)
    dW, db = grads
    W_new = W-lr*dW # update weight param
    B_new = b-lr*db # update bias
    return W_new, B_new, loss

def fit_sgd(train_x, train_y, test_x, test_y, lr, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    W, b = init_linear_params(key)

    train_errors = []
    test_errors = []
    
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            W, b, loss = train_sgd_step(W, b, xn, yn, lr)

        train_accuracy = float(accuracy(W, b, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(accuracy(W, b, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[Linear Classfier - SGD] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return W,b, train_errors, test_errors

# Optimizer 1 - SGD 
W_sgd, b_sgd, train_errors, test_errors = fit_sgd(train_x, train_y, test_x, test_y, lr=0.005)
plot_errors(train_errors, test_errors, "Linear Classifier - SGD | Training Error v/s Epoch | Test Error v/s Epoch")

os.makedirs("saved_models/linear_classifier", exist_ok=True)
np.save("saved_models/linear_classifier/sgd_train.npy", np.array(train_errors))
np.save("saved_models/linear_classifier/sgd_test.npy",  np.array(test_errors))
np.savez("saved_models/linear_classifier/sgd_model.npz", W=W_sgd, b=b_sgd)

# Linear Classifier - SGD with momentum Table 2
def train_sgdm_step(W, b, mean_g_W, mean_g_b, x, y, lr, beta):
    def loss_fn(W_, B_):
        return cross_entropy_loss(W_, B_, x, y)
    loss, (dW, db) = jax.value_and_grad(loss_fn, argnums=(0, 1))(W, b)

    mean_g_W_new = (1.0 - beta) * mean_g_W + beta * dW
    mean_g_b_new = (1.0 - beta) * mean_g_b + beta * db

    W_new = W - lr * mean_g_W_new
    B_new = b - lr * mean_g_b_new
    return W_new, B_new, mean_g_W_new, mean_g_b_new, loss


def fit_sgdm(train_x, train_y, test_x, test_y, lr, beta=0.9, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    W, b = init_linear_params(key)
    mean_g_W = jnp.zeros_like(W)
    mean_g_b = jnp.zeros_like(b)
    train_errors=[]
    test_errors=[]
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            W, b, mean_g_W, mean_g_b, loss = train_sgdm_step(W, b, mean_g_W, mean_g_b, xn, yn, lr, beta)
        
        train_accuracy = float(accuracy(W, b, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(accuracy(W, b, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[Linear Classifier - SGDM] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return W, b, train_errors, test_errors

# Table 2 Linear Classifier - SGD with momentum
W_sgd, b_sgd, train_errors, test_errors = fit_sgdm(train_x, train_y, test_x, test_y, lr=0.005)
plot_errors(train_errors, test_errors, "Linear Classifier - SGDM | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/linear_classifier/sgdm_train.npy", np.array(train_errors))
np.save("saved_models/linear_classifier/sgdm_test.npy",  np.array(test_errors))
np.savez("saved_models/linear_classifier/sgdm_model.npz", W=W_sgd, b=b_sgd)

# Linear Classifier - Adam Table 2
def train_adam_step(W, b, mean_g_W, mean_g_b, mean_g2_W, mean_g2_b, x, y, lr, beta1, beta2, eps=1e-7):
    def loss_fn(W_, B_):
        return cross_entropy_loss(W_, B_, x, y)
        
    loss, (dW, db) = jax.value_and_grad(loss_fn, argnums=(0, 1))(W, b)
    
    mean_g_W_new = beta1*mean_g_W + (1.0-beta1)*dW
    mean_g_b_new = beta1*mean_g_b + (1.0-beta1)*db

    mean_g2_W_new = beta2*mean_g2_W + (1.0-beta2)*(dW * dW)
    mean_g2_b_new = beta2*mean_g2_b + (1.0-beta2)*(db * db)

    W_new = W - lr*mean_g_W_new/(jnp.sqrt(mean_g2_W_new) + eps)
    b_new = b - lr*mean_g_b_new/(jnp.sqrt(mean_g2_b_new) + eps)
    
    return (W_new, b_new, mean_g_W_new, mean_g_b_new, mean_g2_W_new, mean_g2_b_new, loss)

def fit_adam(train_x, train_y, test_x, test_y, lr, beta1=0.9, beta2=0.999, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    W, b = init_linear_params(key)

    train_errors=[]
    test_errors=[]
    mean_g_W  = jnp.zeros_like(W)
    mean_g_b  = jnp.zeros_like(b)
    mean_g2_W = jnp.zeros_like(W)
    mean_g2_b = jnp.zeros_like(b)

    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            (W, b, mean_g_W, mean_g_b, mean_g2_W, mean_g2_b, loss) = train_adam_step(W, b,mean_g_W, mean_g_b, 
                                                                                     mean_g2_W, mean_g2_b, xn, yn, 
                                                                                     lr=lr, beta1=beta1, beta2=beta2)

        train_accuracy = float(accuracy(W, b, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(accuracy(W, b, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[Linear Classifier - Adam] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")

    return W, b, train_errors, test_errors


# Optimizer 3 - Adam
W_sgd, b_sgd, train_errors, test_errors = fit_adam(train_x, train_y, test_x, test_y, lr=0.0001)
plot_errors(train_errors, test_errors, "Linear Classifier - Adam | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/linear_classifier/adam_train.npy", np.array(train_errors))
np.save("saved_models/linear_classifier/adam_test.npy",  np.array(test_errors))
np.savez("saved_models/linear_classifier/adam_model.npz", W=W_sgd, b=b_sgd)

# Architecture 2 - Shallow MLP Scenario
def init_shallow_mlp_params(key, in_dimension=1024, hidden_dimension=128, out_dimension=10):
    k1, k2 = jax.random.split(key)
    W1 = jax.random.normal(k1, (in_dimension, hidden_dimension))*(1.0/jnp.sqrt(in_dimension))
    b1 = jnp.zeros((hidden_dimension,))
    W2 = jax.random.normal(k2, (hidden_dimension, out_dimension))*(1.0/jnp.sqrt(hidden_dimension))
    b2 = jnp.zeros((out_dimension,))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def shallow_mlp_forward_pass(params, x):
    h = jnp.tanh(jnp.dot(x, params["W1"]) + params["b1"])
    logits = jnp.dot(h, params["W2"]) + params["b2"]
    return logits

def shallow_mlp_loss_func(params, x, y):
    logits = shallow_mlp_forward_pass(params, x)
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    neg_log_likelihood = -log_prob[jnp.arange(x.shape[0]),y]
    return neg_log_likelihood.mean()

def shallow_mlp_accuracy(params, x, y):
    logits = shallow_mlp_forward_pass(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return (preds==y).mean()

def train_shallow_mlp_sgd_step(params, x, y, lr):
    def loss_fn(p):
        return shallow_mlp_loss_func(p, x, y)
    loss, grads = jax.value_and_grad(loss_fn)(params)

    updated_params = {"W1": params["W1"]-lr*grads["W1"], "b1": params["b1"]-lr*grads["b1"],
                  "W2": params["W2"]-lr*grads["W2"], "b2": params["b2"]-lr*grads["b2"],}
    return updated_params, loss

def fit_sgd_shallow_mlp(train_x, train_y, test_x, test_y, lr, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_shallow_mlp_params(key)

    train_errors = []
    test_errors = []
    
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            params, loss = train_shallow_mlp_sgd_step(params, xn, yn, lr)

        train_accuracy = float(shallow_mlp_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(shallow_mlp_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[Shallow MLP - SGD] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return params, train_errors, test_errors

# Shallow MLP - SGD
params, train_errors, test_errors = fit_sgd_shallow_mlp(train_x, train_y, test_x, test_y, lr=0.01)
plot_errors(train_errors, test_errors, "Shallow MLP - SGD | Training Error v/s Epoch | Test Error v/s Epoch")

os.makedirs("saved_models/shallow_mlp", exist_ok=True)
np.save("saved_models/shallow_mlp/sgd_train.npy", np.array(train_errors))
np.save("saved_models/shallow_mlp/sgd_test.npy",  np.array(test_errors))
np.savez("saved_models/shallow_mlp/sgd_model.npz", W1=np.array(params["W1"]), 
         b1=np.array(params["b1"]), W2=np.array(params["W2"]), 
         b2=np.array(params["b2"]),)

# Shallow MLP  Table 3 - SGD with momentum
def shallow_mlp_sgdm_step(params, mean_g, x, y, lr, beta):
    def loss_fn(p):
        return shallow_mlp_loss_func(p, x, y)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    new_mean_g = {}
    new_params = {}
    for name in params.keys():
        mg = (1.0-beta)*mean_g[name]+grads[name]*beta
        w_new = params[name] - lr * mg
        new_mean_g[name] = mg
        new_params[name] = w_new
    return new_params, new_mean_g, loss

def fit_shallow_mlp_sgdm(train_x, train_y, test_x, test_y, lr, beta=0.9, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_shallow_mlp_params(key)
    mean_g = {k: jnp.zeros_like(v) for k, v in params.items()}
    train_errors=[]
    test_errors=[]
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            params, mean_g, loss = shallow_mlp_sgdm_step(params, mean_g, xn, yn, lr, beta)
        
        train_accuracy = float(shallow_mlp_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(shallow_mlp_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[Shallow MLP - SGDM] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return params, train_errors, test_errors

# Shallow MLP - SGD with momentum
params, train_errors, test_errors = fit_shallow_mlp_sgdm(train_x, train_y, test_x, test_y, lr=0.01)
plot_errors(train_errors, test_errors, "Shallow MLP - SGD with momentum | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/shallow_mlp/sgdm_train.npy", np.array(train_errors))
np.save("saved_models/shallow_mlp/sgdm_test.npy",  np.array(test_errors))
np.savez("saved_models/shallow_mlp/sgdm_model.npz", W1=np.array(params["W1"]), 
         b1=np.array(params["b1"]), W2=np.array(params["W2"]), 
         b2=np.array(params["b2"]),)

# Shallow MLP - Adam
def shallow_mlp_adam_step(params, mean_g, mean_g2, x, y, lr, beta1, beta2, eps=1e-7):
    def loss_fn(p):
        return shallow_mlp_loss_func(p, x, y)
    loss, grads = jax.value_and_grad(loss_fn)(params)

    new_params = {}
    new_mean_g = {}
    new_mean_g2 = {}

    for name in params.keys():
        g = grads[name]
        mg = beta1*mean_g[name]+(1.0-beta1)*g
        mg2 = beta2*mean_g2[name]+(1.0-beta2)*(g*g)
        w_new = params[name]-lr* mg/(jnp.sqrt(mg2)+eps)

        new_params[name] = w_new
        new_mean_g[name] = mg
        new_mean_g2[name] = mg2

    return new_params, new_mean_g, new_mean_g2, loss

def fit_adam_shallow_mlp(train_x, train_y, test_x, test_y, lr, beta1=0.9, beta2=0.9, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_shallow_mlp_params(key)
    mean_g = {k: jnp.zeros_like(v) for k, v in params.items()}
    mean_g2 = {k: jnp.zeros_like(v) for k, v in params.items()}

    train_errors=[]
    test_errors=[]

    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            (params, mean_g, mean_g2, loss) = shallow_mlp_adam_step(params, mean_g, mean_g2, xn, yn, lr=lr, beta1=beta1, beta2=beta2)

        train_accuracy = float(shallow_mlp_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(shallow_mlp_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[Shallow MLP - Adam] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")

    return params, train_errors, test_errors

# Shallow MLP - Adam
params, train_errors, test_errors = fit_adam_shallow_mlp(train_x, train_y, test_x, test_y, lr=0.0005)
plot_errors(train_errors, test_errors, "Shallow MLP - Adam | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/shallow_mlp/adam_train.npy", np.array(train_errors))
np.save("saved_models/shallow_mlp/adam_test.npy",  np.array(test_errors))
np.savez("saved_models/shallow_mlp/adam_model.npz", W1=np.array(params["W1"]), 
         b1=np.array(params["b1"]), W2=np.array(params["W2"]), 
         b2=np.array(params["b2"]),)

# Table 4 - Deep MLP Scenario
def init_deep_mlp_params(key, in_dimension=1024, hidden_dimension=128, out_dimension=10):
    k1, k2 = jax.random.split(key)
    k3, k4 = jax.random.split(k2)
    k5, k5  = jax.random.split(k4)
    params = {
        "W1": jax.random.normal(k1, (in_dimension, hidden_dimension))*(1.0 / jnp.sqrt(in_dimension)),
        "b1": jnp.zeros((hidden_dimension,)),
        
        "W2": jax.random.normal(k3, (hidden_dimension, hidden_dimension))*(1.0 / jnp.sqrt(hidden_dimension)),
        "b2": jnp.zeros((hidden_dimension,)),
        
        "W3": jax.random.normal(k4, (hidden_dimension, hidden_dimension))*(1.0 / jnp.sqrt(hidden_dimension)),
        "b3": jnp.zeros((hidden_dimension,)),

        "W4": jax.random.normal(k5, (hidden_dimension, out_dimension))*(1.0 / jnp.sqrt(hidden_dimension)),
        "b4": jnp.zeros((out_dimension,)),
    }
    return params

def deep_mlp_forward_pass(params, x):
    h1 = jnp.tanh(jnp.dot(x, params["W1"]) + params["b1"])
    h2 = jnp.tanh(jnp.dot(h1, params["W2"]) + params["b2"])
    h3 = jnp.tanh(jnp.dot(h2, params["W3"]) + params["b3"])
    logits = jnp.dot(h3, params["W4"]) + params["b4"]
    return logits

def deep_mlp_loss_func(params, x, y):
    logits = deep_mlp_forward_pass(params, x)
    log_prob = jax.nn.log_softmax(logits,axis=-1)
    neg_log_likelihood = -log_prob[jnp.arange(x.shape[0]),y]
    return neg_log_likelihood.mean()

def deep_mlp_accuracy(params, x, y):
    logits = deep_mlp_forward_pass(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return (preds==y).mean()

def train_deep_mlp_sgd_step(params, x, y, lr):
    def loss_fn(p):
        return deep_mlp_loss_func(p, x, y)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated_params = {}
    for name in params.keys():
        updated_params[name]=params[name]-lr*grads[name]
    return updated_params, loss

def fit_sgd_deep_mlp(train_x, train_y, test_x, test_y, lr, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_deep_mlp_params(key)

    train_errors = []
    test_errors = []
    
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            params, loss = train_deep_mlp_sgd_step(params, xn, yn, lr)

        train_accuracy = float(deep_mlp_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(deep_mlp_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[Deep MLP - SGD] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return params, train_errors, test_errors

# Deep MLP - SGD
params, train_errors, test_errors = fit_sgd_deep_mlp(train_x, train_y, test_x, test_y, lr=0.01)
plot_errors(train_errors, test_errors, "Deep MLP - SGD | Training Error v/s Epoch | Test Error v/s Epoch")

os.makedirs("saved_models/deep_mlp", exist_ok=True)
np.save("saved_models/deep_mlp/sgd_train.npy", np.array(train_errors))
np.save("saved_models/deep_mlp/sgd_test.npy",  np.array(test_errors))
np.savez("saved_models/deep_mlp/sgd_model.npz", W1=np.array(params["W1"]), b1=np.array(params["b1"]), 
         W2=np.array(params["W2"]), b2=np.array(params["b2"]), 
         W3=np.array(params["W3"]), b3=np.array(params["b3"]), 
         W4=np.array(params["W4"]), b4=np.array(params["b4"]),)

# Deep MLP - SGD with momentum
def deep_mlp_sgdm_step(params, mean_g, x, y, lr, beta):
    def loss_fn(p):
        return deep_mlp_loss_func(p, x, y)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    new_mean_g = {}
    new_params = {}
    for name in params.keys():
        mg = (1.0-beta)*mean_g[name]+grads[name]*beta
        w_new = params[name] - lr * mg
        new_mean_g[name] = mg
        new_params[name] = w_new
    return new_params, new_mean_g, loss

def fit_deep_mlp_sgdm(train_x, train_y, test_x, test_y, lr, beta=0.9, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_deep_mlp_params(key)
    mean_g = {k: jnp.zeros_like(v) for k, v in params.items()}
    train_errors=[]
    test_errors=[]
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            params, mean_g, loss = deep_mlp_sgdm_step(params, mean_g, xn, yn, lr, beta)
        
        train_accuracy = float(deep_mlp_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(deep_mlp_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[Deep MLP - SGDM] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return params, train_errors, test_errors

# Deep MLP - SGD
params, train_errors, test_errors = fit_deep_mlp_sgdm(train_x, train_y, test_x, test_y, lr=0.01)
plot_errors(train_errors, test_errors, "Deep MLP - SGD with momentum | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/deep_mlp/sgdm_train.npy", np.array(train_errors))
np.save("saved_models/deep_mlp/sgdm_test.npy",  np.array(test_errors))
np.savez("saved_models/deep_mlp/sgdm_model.npz", W1=np.array(params["W1"]), b1=np.array(params["b1"]), 
         W2=np.array(params["W2"]), b2=np.array(params["b2"]), 
         W3=np.array(params["W3"]), b3=np.array(params["b3"]), 
         W4=np.array(params["W4"]), b4=np.array(params["b4"]),)

def deep_mlp_adam_step(params, mean_g, mean_g2, x, y, lr, beta1, beta2, eps=1e-8):
    def loss_fn(p):
        return deep_mlp_loss_func(p, x, y)
    loss, grads = jax.value_and_grad(loss_fn)(params)

    new_params = {}
    new_mean_g = {}
    new_mean_g2 = {}

    for name in params.keys():
        g = grads[name]
        mg = beta1*mean_g[name]+(1.0-beta1)*g
        mg2 = beta2*mean_g2[name]+(1.0-beta2)*(g*g)
        w_new = params[name]-lr* mg/(jnp.sqrt(mg2)+eps)

        new_params[name] = w_new
        new_mean_g[name] = mg
        new_mean_g2[name] = mg2

    return new_params, new_mean_g, new_mean_g2, loss

def fit_adam_deep_mlp(train_x, train_y, test_x, test_y, lr, beta1=0.9, beta2=0.9, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_deep_mlp_params(key)
    mean_g = {k: jnp.zeros_like(v) for k, v in params.items()}
    mean_g2 = {k: jnp.zeros_like(v) for k, v in params.items()}

    train_errors=[]
    test_errors=[]

    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            (params, mean_g, mean_g2, loss) = deep_mlp_adam_step(params, mean_g, mean_g2, xn, yn, lr=lr, beta1=beta1, beta2=beta2)

        train_accuracy = float(deep_mlp_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(deep_mlp_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[Deep MLP - Adam] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")

    return params, train_errors, test_errors

# Deep MLP - Adam
params, train_errors, test_errors = fit_adam_deep_mlp(train_x, train_y, test_x, test_y, lr=0.0008)
plot_errors(train_errors, test_errors, "Deep MLP - Adam | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/deep_mlp/adam_train.npy", np.array(train_errors))
np.save("saved_models/deep_mlp/adam_test.npy",  np.array(test_errors))
np.savez("saved_models/deep_mlp/adam_model.npz", W1=np.array(params["W1"]), b1=np.array(params["b1"]), 
         W2=np.array(params["W2"]), b2=np.array(params["b2"]), 
         W3=np.array(params["W3"]), b3=np.array(params["b3"]), 
         W4=np.array(params["W4"]), b4=np.array(params["b4"]),)

# Table 5 - Deep MLP with ReLU Scenario
def init_deep_mlp_relu_params(key, in_dimension=1024, hidden_dimension=128, out_dimension=10):
    k1, k2 = jax.random.split(key)
    k3, k4 = jax.random.split(k2)
    k5, k5  = jax.random.split(k4)
    params = {
        "W1": jax.random.normal(k1, (in_dimension, hidden_dimension))*(1.0 / jnp.sqrt(in_dimension)),
        "b1": jnp.zeros((hidden_dimension,)),
        
        "W2": jax.random.normal(k3, (hidden_dimension, hidden_dimension))*(1.0 / jnp.sqrt(hidden_dimension)),
        "b2": jnp.zeros((hidden_dimension,)),
        
        "W3": jax.random.normal(k4, (hidden_dimension, hidden_dimension))*(1.0 / jnp.sqrt(hidden_dimension)),
        "b3": jnp.zeros((hidden_dimension,)),

        "W4": jax.random.normal(k5, (hidden_dimension, out_dimension))*(1.0 / jnp.sqrt(hidden_dimension)),
        "b4": jnp.zeros((out_dimension,)),
    }
    return params
    
def deep_mlp_relu_forward_pass(params, x):
    h1 = jnp.maximum(jnp.dot(x, params["W1"]) + params["b1"], 0.0)
    h2 = jnp.maximum(jnp.dot(h1, params["W2"]) + params["b2"], 0.0)
    h3 = jnp.maximum(jnp.dot(h2, params["W3"]) + params["b3"], 0.0)
    logits = jnp.dot(h3, params["W4"]) + params["b4"]
    return logits

def deep_mlp_relu_loss_func(params, x, y):
    logits = deep_mlp_relu_forward_pass(params, x)
    log_prob = jax.nn.log_softmax(logits,axis=-1)
    neg_log_likelihood = -log_prob[jnp.arange(x.shape[0]),y]
    return neg_log_likelihood.mean()

def deep_mlp_relu_accuracy(params, x, y):
    logits = deep_mlp_relu_forward_pass(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return (preds==y).mean()

def train_deep_mlp_relu_sgd_step(params, x, y, lr):
    def loss_fn(p):
        return deep_mlp_relu_loss_func(p, x, y)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated_params = {}
    for name in params.keys():
        updated_params[name]=params[name]-lr*grads[name]
    return updated_params, loss

def fit_sgd_deep_mlp_relu(train_x, train_y, test_x, test_y, lr, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_deep_mlp_relu_params(key)

    train_errors = []
    test_errors = []
    
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            params, loss = train_deep_mlp_relu_sgd_step(params, xn, yn, lr)

        train_accuracy = float(deep_mlp_relu_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(deep_mlp_relu_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[Deep MLP ReLU - SGD] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return params, train_errors, test_errors

# Deep MLP ReLU - SGD
params, train_errors, test_errors = fit_sgd_deep_mlp_relu(train_x, train_y, test_x, test_y, lr=0.01)
plot_errors(train_errors, test_errors, "Deep MLP ReLU- SGD | Training Error v/s Epoch | Test Error v/s Epoch")

os.makedirs("saved_models/deep_mlp_relu", exist_ok=True)
np.save("saved_models/deep_mlp_relu/sgd_train.npy", np.array(train_errors))
np.save("saved_models/deep_mlp_relu/sgd_test.npy",  np.array(test_errors))
np.savez("saved_models/deep_mlp_relu/sgd_model.npz", W1=np.array(params["W1"]), b1=np.array(params["b1"]), 
         W2=np.array(params["W2"]), b2=np.array(params["b2"]), 
         W3=np.array(params["W3"]), b3=np.array(params["b3"]), 
         W4=np.array(params["W4"]), b4=np.array(params["b4"]),)

# Deep MLP ReLU - SGD with momentum
def deep_mlp_relu_sgdm_step(params, mean_g, x, y, lr, beta):
    def loss_fn(p):
        return deep_mlp_relu_loss_func(p, x, y)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    new_mean_g = {}
    new_params = {}
    for name in params.keys():
        mg = (1.0-beta)*mean_g[name]+grads[name]*beta
        w_new = params[name] - lr * mg
        new_mean_g[name] = mg
        new_params[name] = w_new
    return new_params, new_mean_g, loss

def fit_deep_mlp_relu_sgdm(train_x, train_y, test_x, test_y, lr, beta=0.9, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_deep_mlp_relu_params(key)
    mean_g = {k: jnp.zeros_like(v) for k, v in params.items()}
    train_errors=[]
    test_errors=[]
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            params, mean_g, loss = deep_mlp_relu_sgdm_step(params, mean_g, xn, yn, lr, beta)
        
        train_accuracy = float(deep_mlp_relu_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(deep_mlp_relu_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[Deep MLP ReLU - SGDM] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return params, train_errors, test_errors

# Deep MLP - SGD
params, train_errors, test_errors = fit_deep_mlp_relu_sgdm(train_x, train_y, test_x, test_y, lr=0.01)
plot_errors(train_errors, test_errors, "Deep MLP ReLU - SGD with momentum | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/deep_mlp_relu/sgdm_train.npy", np.array(train_errors))
np.save("saved_models/deep_mlp_relu/sgdm_test.npy",  np.array(test_errors))
np.savez("saved_models/deep_mlp_relu/sgdm_model.npz", W1=np.array(params["W1"]), b1=np.array(params["b1"]), 
         W2=np.array(params["W2"]), b2=np.array(params["b2"]), 
         W3=np.array(params["W3"]), b3=np.array(params["b3"]), 
         W4=np.array(params["W4"]), b4=np.array(params["b4"]),)

# Deep MLP ReLU - Adam
def deep_mlp_relu_adam_step(params, mean_g, mean_g2, x, y, lr, beta1, beta2, eps=1e-8):
    def loss_fn(p):
        return deep_mlp_relu_loss_func(p, x, y)
    loss, grads = jax.value_and_grad(loss_fn)(params)

    new_params = {}
    new_mean_g = {}
    new_mean_g2 = {}

    for name in params.keys():
        g = grads[name]
        mg = beta1*mean_g[name]+(1.0-beta1)*g
        mg2 = beta2*mean_g2[name]+(1.0-beta2)*(g*g)
        w_new = params[name]-lr* mg/(jnp.sqrt(mg2)+eps)

        new_params[name] = w_new
        new_mean_g[name] = mg
        new_mean_g2[name] = mg2

    return new_params, new_mean_g, new_mean_g2, loss

def fit_adam_deep_mlp_relu(train_x, train_y, test_x, test_y, lr, beta1=0.9, beta2=0.9, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_deep_mlp_relu_params(key)
    mean_g = {k: jnp.zeros_like(v) for k, v in params.items()}
    mean_g2 = {k: jnp.zeros_like(v) for k, v in params.items()}

    train_errors=[]
    test_errors=[]

    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            (params, mean_g, mean_g2, loss) = deep_mlp_relu_adam_step(params, mean_g, mean_g2, xn, yn, lr=lr, beta1=beta1, beta2=beta2)

        train_accuracy = float(deep_mlp_relu_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(deep_mlp_relu_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[Deep MLP ReLU - Adam] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")

    return params, train_errors, test_errors

params, train_errors, test_errors = fit_adam_deep_mlp_relu(train_x, train_y, test_x, test_y, lr=0.001)
plot_errors(train_errors, test_errors, "Deep MLP ReLU - Adam | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/deep_mlp_relu/adam_train.npy", np.array(train_errors))
np.save("saved_models/deep_mlp_relu/adam_test.npy",  np.array(test_errors))
np.savez("saved_models/deep_mlp_relu/adam_model.npz", W1=np.array(params["W1"]), b1=np.array(params["b1"]), 
         W2=np.array(params["W2"]), b2=np.array(params["b2"]), 
         W3=np.array(params["W3"]), b3=np.array(params["b3"]), 
         W4=np.array(params["W4"]), b4=np.array(params["b4"]),)

# CNN
def to_cnn_input(x_flat):
    # Convert back to 32x32 shape
    return x_flat.reshape(-1, 1, 32, 32)

def init_cnn_params(key, num_classes=10):
    k1, k2, k3 = jax.random.split(key, 3)

    # Layer 1: Conv2D: 32 filters, 3 × 3 kernels, 1 channel, ReLU activation
    conv1_shape = (32, 1, 3, 3)
    conv1_n_in  = 1*3*3
    conv1_n_out = 32*3*3
    conv1_std   = 1.0/jnp.sqrt(conv1_n_in+conv1_n_out)
    layer1_conv_W = conv1_std*jax.random.normal(k1, conv1_shape)
    layer1_conv_b = jnp.zeros((32,))

    # Layer 3 Conv2D: 64 filters, 3 × 3 kernels, 32 channels, ReLU activation
    conv2_shape = (64, 32, 3, 3)
    conv2_n_in  = 32*3*3
    conv2_n_out = 64*3*3
    conv2_std   = 1.0/jnp.sqrt(conv2_n_in + conv2_n_out)
    layer3_conv_W = conv2_std*jax.random.normal(k2, conv2_shape)
    layer3_conv_b = jnp.zeros((64,))

    # Layer 5 O/p after two 2x2 downsamples: 32->16->8, channels=64 -> 64*8*8
    fc_in  = 64*8*8
    fc_out = num_classes
    fc_std = 1.0/jnp.sqrt(fc_in+fc_out)
    output_fc_W = fc_std*jax.random.normal(k3, (fc_in, fc_out))
    output_fc_b = jnp.zeros((fc_out,))
    return {
        "layer1_conv_W": layer1_conv_W,
        "layer1_conv_b": layer1_conv_b,
        "layer3_conv_W": layer3_conv_W,
        "layer3_conv_b": layer3_conv_b,
        "output_fc_W": output_fc_W,
        "output_fc_b": output_fc_b,
    }

def conv_relu(x, W, b, stride=1):
    y = lax.conv_general_dilated(x, W, window_strides=(stride, stride), padding="SAME", dimension_numbers=("NCHW", "OIHW", "NCHW"))
    y = y + b.reshape(1, -1, 1, 1)
    return jnp.maximum(y, 0.0)

def maxpool_2x2(x):
    return lax.reduce_window(x,-jnp.inf,lax.max, window_dimensions=(1, 1, 2, 2), 
                             window_strides=(1, 1, 2, 2), padding="VALID")

def cnn_forward_pass(params, x):
    h1  = conv_relu(x, params["layer1_conv_W"], params["layer1_conv_b"], stride=1)
    h1p = maxpool_2x2(h1)

    h2  = conv_relu(h1p, params["layer3_conv_W"], params["layer3_conv_b"], stride=1)
    h2p = maxpool_2x2(h2)

    flat = h2p.reshape(x.shape[0], -1)
    logits = jnp.dot(flat, params["output_fc_W"]) + params["output_fc_b"]
    return logits

def cnn_loss_func(params, x, y):
    logits = cnn_forward_pass(params, x)
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    neg_log_likelihood = -log_prob[jnp.arange(x.shape[0]), y]
    return neg_log_likelihood.mean()

def cnn_accuracy(params, x, y):
    logits = cnn_forward_pass(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return (preds==y).mean()

def train_cnn_sgd_step(params, x, y, lr):
    def loss_fn(p):
        return cnn_loss_func(p, x, y)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated_params = {k: params[k]-lr*grads[k] for k in params.keys()}
    return updated_params, loss

def fit_sgd_cnn(train_x, train_y, test_x, test_y, lr, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_cnn_params(key)

    train_errors = []
    test_errors = []
    
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            params, loss = train_cnn_sgd_step(params, xn, yn, lr)

        train_accuracy = float(cnn_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(cnn_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[CNN - SGD] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return params, train_errors, test_errors

# CNN - SGD
train_x_cnn = train_x.reshape(-1, 1, 32, 32)
test_x_cnn  = test_x.reshape(-1, 1, 32, 32)
params, train_errors, test_errors = fit_sgd_cnn(train_x_cnn, train_y, test_x_cnn, test_y, lr=0.01)
plot_errors(train_errors, test_errors, "CNN - SGD | Training Error v/s Epoch | Test Error v/s Epoch")

os.makedirs("saved_models/cnn_table_6")
np.save("saved_models/cnn_table_6/sgd_train.npy", np.array(train_errors))
np.save("saved_models/cnn_table_6/sgd_test.npy",  np.array(test_errors))
np.savez("saved_models/cnn_table_6/sgd_model",
        layer1_conv_W=np.array(params["layer1_conv_W"]), layer1_conv_b=np.array(params["layer1_conv_b"]),
        layer3_conv_W=np.array(params["layer3_conv_W"]), layer3_conv_b=np.array(params["layer3_conv_b"]),
        output_fc_W=np.array(params["output_fc_W"]), output_fc_b=np.array(params["output_fc_b"]),)

# CNN Table 6 - SGD with momentum
def train_cnn_sgdm_step(params, mean_g, x, y, lr, beta):
    def loss_fn(p):
        return cnn_loss_func(p, x, y)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    new_mean_g = {}
    new_params = {}
    for name in params.keys():
        mg = (1.0-beta)*mean_g[name]+grads[name]*beta
        w_new = params[name] - lr * mg
        new_mean_g[name] = mg
        new_params[name] = w_new
    return new_params, new_mean_g, loss

def fit_sgdm_cnn(train_x, train_y, test_x, test_y, lr, beta=0.9, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_cnn_params(key)
    mean_g = {k: jnp.zeros_like(v) for k, v in params.items()}
    train_errors=[]
    test_errors=[]
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            params, mean_g, loss = train_cnn_sgdm_step(params, mean_g, xn, yn, lr, beta)
        
        train_accuracy = float(cnn_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(cnn_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[CNN - SGDM] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return params, train_errors, test_errors

# CNN Table 6 - SGDM
train_x_cnn = train_x.reshape(-1, 1, 32, 32)
test_x_cnn  = test_x.reshape(-1, 1, 32, 32)
params, train_errors, test_errors = fit_sgdm_cnn(train_x_cnn, train_y, test_x_cnn, test_y, lr=0.005)
plot_errors(train_errors, test_errors, "CNN - SGD with momentum | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/cnn_table_6/sgdm_train.npy", np.array(train_errors))
np.save("saved_models/cnn_table_6/sgdm_test.npy",  np.array(test_errors))
np.savez("saved_models/cnn_table_6/sgdm_model",
        layer1_conv_W=np.array(params["layer1_conv_W"]), layer1_conv_b=np.array(params["layer1_conv_b"]),
        layer3_conv_W=np.array(params["layer3_conv_W"]), layer3_conv_b=np.array(params["layer3_conv_b"]),
        output_fc_W=np.array(params["output_fc_W"]), output_fc_b=np.array(params["output_fc_b"]),)

# CNN Table 6 - Adam
def train_cnn_adam_step(params, mean_g, mean_g2, x, y, lr, beta1, beta2, eps=1e-7):
    def loss_fn(p):
        return cnn_loss_func(p, x, y)
    loss, grads = jax.value_and_grad(loss_fn)(params)

    new_params = {}
    new_mean_g = {}
    new_mean_g2 = {}

    for name in params.keys():
        g = grads[name]
        mg = beta1*mean_g[name]+(1.0-beta1)*g
        mg2 = beta2*mean_g2[name]+(1.0-beta2)*(g*g)
        w_new = params[name]-lr* mg/(jnp.sqrt(mg2)+eps)

        new_params[name] = w_new
        new_mean_g[name] = mg
        new_mean_g2[name] = mg2

    return new_params, new_mean_g, new_mean_g2, loss

def fit_adam_cnn(train_x, train_y, test_x, test_y, lr, beta1=0.9, beta2=0.9, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_cnn_params(key)
    mean_g = {k: jnp.zeros_like(v) for k, v in params.items()}
    mean_g2 = {k: jnp.zeros_like(v) for k, v in params.items()}

    train_errors=[]
    test_errors=[]

    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            (params, mean_g, mean_g2, loss) = train_cnn_adam_step(params, mean_g, mean_g2, xn, yn, lr=lr, beta1=beta1, beta2=beta2)

        train_accuracy = float(cnn_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(cnn_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[CNN - Adam] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")

    return params, train_errors, test_errors

train_x_cnn = train_x.reshape(-1, 1, 32, 32)
test_x_cnn  = test_x.reshape(-1, 1, 32, 32)
params, train_errors, test_errors = fit_adam_cnn(train_x_cnn, train_y, test_x_cnn, test_y, lr=0.001)
plot_errors(train_errors, test_errors, "CNN - Adam | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/cnn_table_6/adam_train.npy", np.array(train_errors))
np.save("saved_models/cnn_table_6/adam_test.npy",  np.array(test_errors))
np.savez("saved_models/cnn_table_6/adam_model",
        layer1_conv_W=np.array(params["layer1_conv_W"]), layer1_conv_b=np.array(params["layer1_conv_b"]),
        layer3_conv_W=np.array(params["layer3_conv_W"]), layer3_conv_b=np.array(params["layer3_conv_b"]),
        output_fc_W=np.array(params["output_fc_W"]), output_fc_b=np.array(params["output_fc_b"]),)

## CNN (with dropout) - Table 7

def dropout_func(x, key, p=0.5, is_training=True):
    if not is_training:
        return x
    prob = 1.0-p
    mask = jax.random.bernoulli(key, prob, x.shape)
    return x*mask/prob

def cnn_forward_pass_dropout(params, x, key=None, dropout=0.5, is_training=True):
    h1  = conv_relu(x, params["layer1_conv_W"], params["layer1_conv_b"], stride=1)
    h1p = maxpool_2x2(h1)

    h2  = conv_relu(h1p, params["layer3_conv_W"], params["layer3_conv_b"], stride=1)
    h2p = maxpool_2x2(h2)

    flat = h2p.reshape(x.shape[0], -1)
    
    if key is None:
        # Test scenario
        flat = dropout_func(flat, jax.random.PRNGKey(0), p=dropout, is_training=False)
    else:
        flat = dropout_func(flat, key, p=dropout, is_training=is_training)
        
    logits = jnp.dot(flat, params["output_fc_W"]) + params["output_fc_b"]
    return logits

def cnn_loss_func_dropout(params, x, y, key, dropout):
    logits = cnn_forward_pass_dropout(params, x, key=key, dropout=dropout, is_training=True)
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    neg_log_likelihood = -log_prob[jnp.arange(x.shape[0]), y]
    return neg_log_likelihood.mean()

def cnn_accuracy_dropout(params, x, y):
    logits = cnn_forward_pass_dropout(params, x, key=None, dropout=0.5, is_training=False)
    preds = jnp.argmax(logits, axis=-1)
    return (preds==y).mean()

def train_cnn_sgd_step_dropout(params, x, y, lr, key, dropout):
    def loss_fn(p):
        return cnn_loss_func_dropout(p, x, y, key, dropout=dropout)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated_params = {k: params[k]-lr*grads[k] for k in params.keys()}
    return updated_params, loss

def fit_sgd_cnn_dropout(train_x, train_y, test_x, test_y, lr, dropout, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_cnn_params(key)

    train_errors = []
    test_errors = []
    
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            key, subkey = jax.random.split(key)
            params, loss = train_cnn_sgd_step_dropout(params, xn, yn, lr, subkey, dropout)

        train_accuracy = float(cnn_accuracy_dropout(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(cnn_accuracy_dropout(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[CNN - SGD] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return params, train_errors, test_errors

# CNN with dropout - SGD
train_x_cnn = train_x.reshape(-1, 1, 32, 32)
test_x_cnn  = test_x.reshape(-1, 1, 32, 32)
params, train_errors, test_errors = fit_sgd_cnn_dropout(train_x_cnn, train_y, test_x_cnn, test_y, lr=0.01, dropout=0.5)
plot_errors(train_errors, test_errors, "CNN with dropout - SGD | Training Error v/s Epoch | Test Error v/s Epoch")

os.makedirs("saved_models/cnn_table_7")
np.save("saved_models/cnn_table_7/sgd_train.npy", np.array(train_errors))
np.save("saved_models/cnn_table_7/sgd_test.npy",  np.array(test_errors))
np.savez("saved_models/cnn_table_7/sgd_model",
        layer1_conv_W=np.array(params["layer1_conv_W"]), layer1_conv_b=np.array(params["layer1_conv_b"]),
        layer3_conv_W=np.array(params["layer3_conv_W"]), layer3_conv_b=np.array(params["layer3_conv_b"]),
        output_fc_W=np.array(params["output_fc_W"]), output_fc_b=np.array(params["output_fc_b"]),)

# CNN with dropout Table 7 - SGD with momentum
def train_cnn_sgdm_step_dropout(params, mean_g, x, y, lr, beta, key, dropout):
    def loss_fn(p):
        return cnn_loss_func_dropout(p, x, y, key, dropout=dropout)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    new_mean_g = {}
    new_params = {}
    for name in params.keys():
        mg = (1.0-beta)*mean_g[name]+grads[name]*beta
        w_new = params[name] - lr * mg
        new_mean_g[name] = mg
        new_params[name] = w_new
    return new_params, new_mean_g, loss

def fit_sgdm_cnn_dropout(train_x, train_y, test_x, test_y, lr, dropout, beta=0.9, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_cnn_params(key)
    mean_g = {k: jnp.zeros_like(v) for k, v in params.items()}
    train_errors=[]
    test_errors=[]
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            key, subkey = jax.random.split(key)
            params, mean_g, loss = train_cnn_sgdm_step_dropout(params, mean_g, xn, yn, lr, beta, subkey, dropout)
        
        train_accuracy = float(cnn_accuracy_dropout(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(cnn_accuracy_dropout(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[CNN with dropout - SGDM] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return params, train_errors, test_errors

# CNN Table 7 - SGDM
train_x_cnn = train_x.reshape(-1, 1, 32, 32)
test_x_cnn  = test_x.reshape(-1, 1, 32, 32)
params, train_errors, test_errors = fit_sgdm_cnn_dropout(train_x_cnn, train_y, test_x_cnn, test_y, lr=0.005, dropout=0.5)
plot_errors(train_errors, test_errors, "CNN with dropout - SGD with momentum | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/cnn_table_7/sgdm_train.npy", np.array(train_errors))
np.save("saved_models/cnn_table_7/sgdm_test.npy",  np.array(test_errors))
np.savez("saved_models/cnn_table_7/sgdm_model",
        layer1_conv_W=np.array(params["layer1_conv_W"]), layer1_conv_b=np.array(params["layer1_conv_b"]),
        layer3_conv_W=np.array(params["layer3_conv_W"]), layer3_conv_b=np.array(params["layer3_conv_b"]),
        output_fc_W=np.array(params["output_fc_W"]), output_fc_b=np.array(params["output_fc_b"]),)

# CNN with dropout Table 7 - Adam
def train_cnn_adam_step_dropout(params, mean_g, mean_g2, x, y, lr, beta1, beta2, key, dropout, eps=1e-7):
    def loss_fn(p):
        return cnn_loss_func_dropout(p, x, y, key, dropout=dropout)
    loss, grads = jax.value_and_grad(loss_fn)(params)

    new_params = {}
    new_mean_g = {}
    new_mean_g2 = {}

    for name in params.keys():
        g = grads[name]
        mg = beta1*mean_g[name]+(1.0-beta1)*g
        mg2 = beta2*mean_g2[name]+(1.0-beta2)*(g*g)
        w_new = params[name]-lr* mg/(jnp.sqrt(mg2)+eps)

        new_params[name] = w_new
        new_mean_g[name] = mg
        new_mean_g2[name] = mg2

    return new_params, new_mean_g, new_mean_g2, loss

def fit_adam_cnn_dropout(train_x, train_y, test_x, test_y, lr, dropout, beta1=0.9, beta2=0.9, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_cnn_params(key)
    mean_g = {k: jnp.zeros_like(v) for k, v in params.items()}
    mean_g2 = {k: jnp.zeros_like(v) for k, v in params.items()}

    train_errors=[]
    test_errors=[]

    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            key, subkey = jax.random.split(key)
            (params, mean_g, mean_g2, loss) = train_cnn_adam_step_dropout(params, mean_g, mean_g2, xn, yn, lr=lr, beta1=beta1, beta2=beta2, key=subkey, dropout=dropout)

        train_accuracy = float(cnn_accuracy_dropout(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(cnn_accuracy_dropout(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[CNN - Adam] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")

    return params, train_errors, test_errors

train_x_cnn = train_x.reshape(-1, 1, 32, 32)
test_x_cnn  = test_x.reshape(-1, 1, 32, 32)
params, train_errors, test_errors = fit_adam_cnn_dropout(train_x_cnn, train_y, test_x_cnn, test_y, lr=0.001, dropout=0.5)
plot_errors(train_errors, test_errors, "CNN with Dropout - Adam | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/cnn_table_7/adam_train.npy", np.array(train_errors))
np.save("saved_models/cnn_table_7/adam_test.npy",  np.array(test_errors))
np.savez("saved_models/cnn_table_7/adam_model",
        layer1_conv_W=np.array(params["layer1_conv_W"]), layer1_conv_b=np.array(params["layer1_conv_b"]),
        layer3_conv_W=np.array(params["layer3_conv_W"]), layer3_conv_b=np.array(params["layer3_conv_b"]),
        output_fc_W=np.array(params["output_fc_W"]), output_fc_b=np.array(params["output_fc_b"]),)

#Table 8 - VGG
def init_vgg_params(key, num_classes=10):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    #Layer 1
    conv1_shape = (32, 1, 3, 3)
    conv1_n_in  = 1*3*3
    conv1_n_out = 32*3*3
    conv1_std   = 1.0/jnp.sqrt(conv1_n_in+conv1_n_out)
    layer1_conv_W = conv1_std*jax.random.normal(k1, conv1_shape)
    layer1_conv_b = jnp.zeros((32,))

    #Layer 2
    conv2_shape = (32, 32, 3, 3)
    conv2_n_in  = 32*3*3
    conv2_n_out = 32*3*3
    conv2_std   = 1.0/jnp.sqrt(conv2_n_in + conv2_n_out)
    layer2_conv_W = conv2_std*jax.random.normal(k2, conv2_shape)
    layer2_conv_b = jnp.zeros((32,))

    #Layer 4
    conv3_shape = (64, 32, 3, 3)
    conv3_n_in  = 32*3*3
    conv3_n_out = 64*3*3
    conv3_std   = 1.0/jnp.sqrt(conv3_n_in+conv3_n_out)
    layer4_conv_W = conv3_std*jax.random.normal(k3, conv3_shape)
    layer4_conv_b = jnp.zeros((64,))

    #Layer 5
    conv4_shape = (64, 64, 3, 3)
    conv4_n_in  = 64*3*3
    conv4_n_out = 64*3*3
    conv4_std   = 1.0/jnp.sqrt(conv4_n_in + conv4_n_out)
    layer5_conv_W = conv4_std*jax.random.normal(k4, conv4_shape)
    layer5_conv_b = jnp.zeros((64,))

    fc_in  = 64*8*8
    fc_out = num_classes
    fc_std = 1.0/jnp.sqrt(fc_in+fc_out)
    output_fc_W = fc_std*jax.random.normal(k5, (fc_in, fc_out))
    output_fc_b = jnp.zeros((fc_out,))
    return {
        "layer1_conv_W": layer1_conv_W,
        "layer1_conv_b": layer1_conv_b,
        "layer2_conv_W": layer2_conv_W,
        "layer2_conv_b": layer2_conv_b,
        "layer4_conv_W": layer4_conv_W,
        "layer4_conv_b": layer4_conv_b,
        "layer5_conv_W": layer5_conv_W,
        "layer5_conv_b": layer5_conv_b,
        "output_fc_W": output_fc_W,
        "output_fc_b": output_fc_b,
    }

def vgg_forward_pass(params, x, key=None, dropout=0.5, is_training=True):
    h1  = conv_relu(x, params["layer1_conv_W"], params["layer1_conv_b"], stride=1)
    h2  = conv_relu(h1, params["layer2_conv_W"], params["layer2_conv_b"], stride=1)
    h2p = maxpool_2x2(h2)
    h3  = conv_relu(h2p, params["layer4_conv_W"], params["layer4_conv_b"], stride=1)
    h4  = conv_relu(h3, params["layer5_conv_W"], params["layer5_conv_b"], stride=1)
    h4p = maxpool_2x2(h4)

    flat = h4p.reshape(x.shape[0], -1)
    if key is None:
        # Test scenario
        flat = dropout_func(flat, jax.random.PRNGKey(0), p=dropout, is_training=False)
    else:
        flat = dropout_func(flat, key, p=dropout, is_training=is_training)
        
    logits = jnp.dot(flat, params["output_fc_W"]) + params["output_fc_b"]
    return logits

def vgg_loss_func(params, x, y, key, dropout):
    logits = vgg_forward_pass(params, x, key=key, dropout=dropout, is_training=True)
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    neg_log_likelihood = -log_prob[jnp.arange(x.shape[0]), y]
    return neg_log_likelihood.mean()

def vgg_accuracy(params, x, y):
    logits = vgg_forward_pass(params, x, key=None, dropout=0.5, is_training=False)
    preds = jnp.argmax(logits, axis=-1)
    return (preds==y).mean()

# Table 8 VGG - SGD
def train_vgg_sgd_step(params, x, y, lr, key, dropout):
    def loss_fn(p):
        return vgg_loss_func(p, x, y, key, dropout=dropout)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated_params = {k: params[k]-lr*grads[k] for k in params.keys()}
    return updated_params, loss

def fit_sgd_vgg(train_x, train_y, test_x, test_y, lr, dropout, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_vgg_params(key)

    train_errors = []
    test_errors = []
    
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            key, subkey = jax.random.split(key)
            params, loss = train_vgg_sgd_step(params, xn, yn, lr, subkey, dropout)

        train_accuracy = float(vgg_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(vgg_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[VGG - SGD] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return params, train_errors, test_errors

# Table 8 VGG - SGD
train_x_vgg = train_x.reshape(-1, 1, 32, 32)
test_x_vgg  = test_x.reshape(-1, 1, 32, 32)
params, train_errors, test_errors = fit_sgd_vgg(train_x_vgg, train_y, test_x_vgg, test_y, lr=0.01, dropout=0.5)
plot_errors(train_errors, test_errors, "VGG - SGD | Training Error v/s Epoch | Test Error v/s Epoch")

os.makedirs("saved_models/vgg_table_8", exist_ok=True)
np.save("saved_models/vgg_table_8/sgd_train.npy", np.array(train_errors))
np.save("saved_models/vgg_table_8/sgd_test.npy",  np.array(test_errors))
np.savez("saved_models/vgg_table_8/sgd_model",
        layer1_conv_W=np.array(params["layer1_conv_W"]), layer1_conv_b=np.array(params["layer1_conv_b"]),
        layer2_conv_W=np.array(params["layer2_conv_W"]), layer2_conv_b=np.array(params["layer2_conv_b"]),
        layer4_conv_W=np.array(params["layer4_conv_W"]), layer4_conv_b=np.array(params["layer4_conv_b"]),
        layer5_conv_W=np.array(params["layer5_conv_W"]), layer5_conv_b=np.array(params["layer5_conv_b"]),
        output_fc_W=np.array(params["output_fc_W"]), output_fc_b=np.array(params["output_fc_b"]),)

# VGG Table 8 - SGD with momentum
def train_vgg_sgdm_step(params, mean_g, x, y, lr, beta, key, dropout):
    def loss_fn(p):
        return vgg_loss_func(p, x, y, key, dropout=dropout)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    new_mean_g = {}
    new_params = {}
    for name in params.keys():
        mg = (1.0-beta)*mean_g[name]+grads[name]*beta
        w_new = params[name] - lr * mg
        new_mean_g[name] = mg
        new_params[name] = w_new
    return new_params, new_mean_g, loss

def fit_sgdm_vgg(train_x, train_y, test_x, test_y, lr, dropout, beta=0.9, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_vgg_params(key)
    mean_g = {k: jnp.zeros_like(v) for k, v in params.items()}
    train_errors=[]
    test_errors=[]
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            key, subkey = jax.random.split(key)
            params, mean_g, loss = train_vgg_sgdm_step(params, mean_g, xn, yn, lr, beta, subkey, dropout)
        
        train_accuracy = float(vgg_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(vgg_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[VGG - SGDM] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return params, train_errors, test_errors

# VGG Table 8 - SGDM
train_x_vgg = train_x.reshape(-1, 1, 32, 32)
test_x_vgg  = test_x.reshape(-1, 1, 32, 32)
params, train_errors, test_errors = fit_sgdm_vgg(train_x_vgg, train_y, test_x_vgg, test_y, lr=0.01, dropout=0.5)
plot_errors(train_errors, test_errors, "VGG - SGD with momentum | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/vgg_table_8/sgdm_train.npy", np.array(train_errors))
np.save("saved_models/vgg_table_8/sgdm_test.npy",  np.array(test_errors))
np.savez("saved_models/vgg_table_8/sgdm_model",
        layer1_conv_W=np.array(params["layer1_conv_W"]), layer1_conv_b=np.array(params["layer1_conv_b"]),
        layer2_conv_W=np.array(params["layer2_conv_W"]), layer2_conv_b=np.array(params["layer2_conv_b"]),
        layer4_conv_W=np.array(params["layer4_conv_W"]), layer4_conv_b=np.array(params["layer4_conv_b"]),
        layer5_conv_W=np.array(params["layer5_conv_W"]), layer5_conv_b=np.array(params["layer5_conv_b"]),
        output_fc_W=np.array(params["output_fc_W"]), output_fc_b=np.array(params["output_fc_b"]),)

# VGG Table 8 - Adam
def train_vgg_adam_step(params, mean_g, mean_g2, x, y, lr, beta1, beta2, key, dropout, eps=1e-9):
    def loss_fn(p):
        return vgg_loss_func(p, x, y, key, dropout=dropout)
    loss, grads = jax.value_and_grad(loss_fn)(params)

    new_params = {}
    new_mean_g = {}
    new_mean_g2 = {}

    for name in params.keys():
        g = grads[name]
        mg = beta1*mean_g[name]+(1.0-beta1)*g
        mg2 = beta2*mean_g2[name]+(1.0-beta2)*(g*g)
        w_new = params[name]-lr* mg/(jnp.sqrt(mg2)+eps)

        new_params[name] = w_new
        new_mean_g[name] = mg
        new_mean_g2[name] = mg2

    return new_params, new_mean_g, new_mean_g2, loss

def fit_adam_vgg(train_x, train_y, test_x, test_y, lr, dropout, beta1=0.9, beta2=0.999, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_vgg_params(key)
    mean_g = {k: jnp.zeros_like(v) for k, v in params.items()}
    mean_g2 = {k: jnp.zeros_like(v) for k, v in params.items()}

    train_errors=[]
    test_errors=[]

    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            key, subkey = jax.random.split(key)
            (params, mean_g, mean_g2, loss) = train_vgg_adam_step(params, mean_g, mean_g2, xn, yn, lr=lr, beta1=beta1, beta2=beta2, key=subkey, dropout=dropout)

        train_accuracy = float(vgg_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(vgg_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[VGG - Adam] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")

    return params, train_errors, test_errors

train_x_vgg = train_x.reshape(-1, 1, 32, 32)
test_x_vgg  = test_x.reshape(-1, 1, 32, 32)
params, train_errors, test_errors = fit_adam_vgg(train_x_vgg, train_y, test_x_vgg, test_y, lr=0.001, dropout=0.5)
plot_errors(train_errors, test_errors, "VGG - Adam | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/vgg_table_8/adam_train.npy", np.array(train_errors))
np.save("saved_models/vgg_table_8/adam_test.npy",  np.array(test_errors))
np.savez("saved_models/vgg_table_8/adam_model",
        layer1_conv_W=np.array(params["layer1_conv_W"]), layer1_conv_b=np.array(params["layer1_conv_b"]),
        layer2_conv_W=np.array(params["layer2_conv_W"]), layer2_conv_b=np.array(params["layer2_conv_b"]),
        layer4_conv_W=np.array(params["layer4_conv_W"]), layer4_conv_b=np.array(params["layer4_conv_b"]),
        layer5_conv_W=np.array(params["layer5_conv_W"]), layer5_conv_b=np.array(params["layer5_conv_b"]),
        output_fc_W=np.array(params["output_fc_W"]), output_fc_b=np.array(params["output_fc_b"]),)

#Table 9 - ResNet
def init_resnet_params(key, num_classes=10):
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

    #Layer 1
    layer1_shape = (32, 1, 3, 3)
    layer1_std   = 1.0/jnp.sqrt(1*3*3+32*3*3)
    layer1_conv_W = layer1_std*jax.random.normal(k1, layer1_shape)
    layer1_conv_b = jnp.zeros((32,))

    #Layer 2
    layer2_shape = (32, 32, 3, 3)
    layer2_std   = 1.0/jnp.sqrt(32*3*3 + 32*3*3)
    layer2_conv_W = layer2_std*jax.random.normal(k2, layer2_shape)
    layer2_conv_b = jnp.zeros((32,))

     #Layer 3 -> Projection
    layer3_shape = (32, 1, 1, 1)
    layer3_std   = 1.0/jnp.sqrt(1+32)
    layer3_projection_W = layer3_std*jax.random.normal(k3, layer3_shape)
    layer3_projection_b = jnp.zeros((32,))

    #Layer 6
    layer6_shape = (64, 32, 3, 3)
    layer6_std   = 1.0/jnp.sqrt(32*3*3 + 64*3*3)
    layer6_conv_W = layer6_std*jax.random.normal(k4, layer6_shape)
    layer6_conv_b = jnp.zeros((64,))

    #Layer 7
    layer7_shape = (64, 64, 3, 3)
    layer7_std   = 1.0/jnp.sqrt(64*3*3 + 64*3*3)
    layer7_conv_W = layer7_std*jax.random.normal(k5, layer7_shape)
    layer7_conv_b = jnp.zeros((64,))

     #Layer 8 -> Projection
    layer8_shape = (64, 32, 1, 1)
    layer8_std   = 1.0/jnp.sqrt(32+64)
    layer8_projection_W = layer8_std*jax.random.normal(k6, layer8_shape)
    layer8_projection_b = jnp.zeros((64,))
    
    fc_in  = 64*8*8
    fc_out = num_classes
    fc_std = 1.0/jnp.sqrt(fc_in+fc_out)
    output_fc_W = fc_std*jax.random.normal(k7, (fc_in, fc_out))
    output_fc_b = jnp.zeros((fc_out,))
    return {
        "layer1_conv_W": layer1_conv_W,
        "layer1_conv_b": layer1_conv_b,
        "layer2_conv_W": layer2_conv_W,
        "layer2_conv_b": layer2_conv_b,
        "layer3_projection_W": layer3_projection_W,
        "layer3_projection_b": layer3_projection_b,
        
        "layer6_conv_W": layer6_conv_W,
        "layer6_conv_b": layer6_conv_b,
        "layer7_conv_W": layer7_conv_W,
        "layer7_conv_b": layer7_conv_b,
        "layer8_projection_W": layer8_projection_W,
        "layer8_projection_b": layer8_projection_b,
        
        "output_fc_W": output_fc_W,
        "output_fc_b": output_fc_b,
    }

def conv_lin(x, W, b, stride=1):
    y = lax.conv_general_dilated(x, W, window_strides=(stride, stride), padding="SAME", dimension_numbers=("NCHW", "OIHW", "NCHW"))
    return y + b.reshape(1, -1, 1, 1)

def resnet_forward_pass(params, x, key=None, dropout=0.5, is_training=True):
    h1  = conv_relu(x, params["layer1_conv_W"], params["layer1_conv_b"], stride=1)
    h2  = conv_lin(h1, params["layer2_conv_W"], params["layer2_conv_b"], stride=1)
    h3 = conv_lin(x, params["layer3_projection_W"], params["layer3_projection_b"], stride=1)
    h4 = jnp.maximum(h2+h3, 0.0)
    h4p = maxpool_2x2(h4)
    
    h5  = conv_relu(h4p, params["layer6_conv_W"], params["layer6_conv_b"], stride=1)
    h6  = conv_lin(h5, params["layer7_conv_W"], params["layer7_conv_b"], stride=1)
    h7 = conv_lin(h4p, params["layer8_projection_W"], params["layer8_projection_b"], stride=1)
    h8 = jnp.maximum(h7+h6, 0.0)
    h8p = maxpool_2x2(h8)

    flat = h8p.reshape(x.shape[0], -1)
    if key is None:
        # Test scenario
        flat = dropout_func(flat, jax.random.PRNGKey(0), p=dropout, is_training=False)
    else:
        flat = dropout_func(flat, key, p=dropout, is_training=is_training)
        
    logits = jnp.dot(flat, params["output_fc_W"]) + params["output_fc_b"]
    return logits

def resnet_loss_func(params, x, y, key, dropout):
    logits = resnet_forward_pass(params, x, key=key, dropout=dropout, is_training=True)
    log_prob = jax.nn.log_softmax(logits, axis=-1)
    neg_log_likelihood = -log_prob[jnp.arange(x.shape[0]), y]
    return neg_log_likelihood.mean()

def resnet_accuracy(params, x, y):
    logits = resnet_forward_pass(params, x, key=None, dropout=0.5, is_training=False)
    preds = jnp.argmax(logits, axis=-1)
    return (preds==y).mean()

# Table 9 ResNet - SGD
def train_resnet_sgd_step(params, x, y, lr, key, dropout):
    def loss_fn(p):
        return resnet_loss_func(p, x, y, key, dropout=dropout)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updated_params = {k: params[k]-lr*grads[k] for k in params.keys()}
    return updated_params, loss

def fit_sgd_resnet(train_x, train_y, test_x, test_y, lr, dropout, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_resnet_params(key)

    train_errors = []
    test_errors = []
    
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            key, subkey = jax.random.split(key)
            params, loss = train_resnet_sgd_step(params, xn, yn, lr, subkey, dropout)

        train_accuracy = float(resnet_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(resnet_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[ResNet - SGD] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return params, train_errors, test_errors

# Table 9 ResNet - SGD
train_x_resnet = train_x.reshape(-1, 1, 32, 32)
test_x_resnet  = test_x.reshape(-1, 1, 32, 32)
params, train_errors, test_errors = fit_sgd_resnet(train_x_resnet, train_y, test_x_resnet, test_y, lr=0.01, dropout=0.5)
plot_errors(train_errors, test_errors, "ResNet - SGD | Training Error v/s Epoch | Test Error v/s Epoch")

os.makedirs("saved_models/resnet_table_9", exist_ok=True)
np.save("saved_models/resnet_table_9/sgd_train.npy", np.array(train_errors))
np.save("saved_models/resnet_table_9/sgd_test.npy",  np.array(test_errors))
np.savez("saved_models/resnet_table_9/sgd_model.npz",
        layer1_conv_W=np.array(params["layer1_conv_W"]), layer1_conv_b=np.array(params["layer1_conv_b"]),
        layer2_conv_W=np.array(params["layer2_conv_W"]), layer2_conv_b=np.array(params["layer2_conv_b"]),
        layer3_projection_W=np.array(params["layer3_projection_W"]), layer3_projection_b=np.array(params["layer3_projection_b"]),
        layer6_conv_W=np.array(params["layer6_conv_W"]), layer6_conv_b=np.array(params["layer6_conv_b"]),
        layer7_conv_W=np.array(params["layer7_conv_W"]), layer7_conv_b=np.array(params["layer7_conv_b"]),
        layer8_projection_W=np.array(params["layer8_projection_W"]), layer8_projection_b=np.array(params["layer8_projection_b"]),
        output_fc_W=np.array(params["output_fc_W"]), output_fc_b=np.array(params["output_fc_b"]),)

# ResNet Table 9 - SGD with momentum
def train_resnet_sgdm_step(params, mean_g, x, y, lr, beta, key, dropout):
    def loss_fn(p):
        return resnet_loss_func(p, x, y, key, dropout=dropout)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    new_mean_g = {}
    new_params = {}
    for name in params.keys():
        mg = (1.0-beta)*mean_g[name]+grads[name]*beta
        w_new = params[name] - lr * mg
        new_mean_g[name] = mg
        new_params[name] = w_new
    return new_params, new_mean_g, loss

def fit_sgdm_resnet(train_x, train_y, test_x, test_y, lr, dropout, beta=0.9, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_resnet_params(key)
    mean_g = {k: jnp.zeros_like(v) for k, v in params.items()}
    train_errors=[]
    test_errors=[]
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            key, subkey = jax.random.split(key)
            params, mean_g, loss = train_resnet_sgdm_step(params, mean_g, xn, yn, lr, beta, subkey, dropout)
        
        train_accuracy = float(resnet_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(resnet_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[ResNet - SGDM] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")
    return params, train_errors, test_errors

# ResNet Table 9 - SGDM
train_x_resnet = train_x.reshape(-1, 1, 32, 32)
test_x_resnet  = test_x.reshape(-1, 1, 32, 32)
params, train_errors, test_errors = fit_sgdm_resnet(train_x_resnet, train_y, test_x_resnet, test_y, lr=0.01, dropout=0.5)
plot_errors(train_errors, test_errors, "ResNet - SGD with momentum | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/resnet_table_9/sgdm_train.npy", np.array(train_errors))
np.save("saved_models/resnet_table_9/sgdm_test.npy",  np.array(test_errors))
np.savez("saved_models/resnet_table_9/sgdm_model.npz",
        layer1_conv_W=np.array(params["layer1_conv_W"]), layer1_conv_b=np.array(params["layer1_conv_b"]),
        layer2_conv_W=np.array(params["layer2_conv_W"]), layer2_conv_b=np.array(params["layer2_conv_b"]),
        layer3_projection_W=np.array(params["layer3_projection_W"]), layer3_projection_b=np.array(params["layer3_projection_b"]),
        layer6_conv_W=np.array(params["layer6_conv_W"]), layer6_conv_b=np.array(params["layer6_conv_b"]),
        layer7_conv_W=np.array(params["layer7_conv_W"]), layer7_conv_b=np.array(params["layer7_conv_b"]),
        layer8_projection_W=np.array(params["layer8_projection_W"]), layer8_projection_b=np.array(params["layer8_projection_b"]),
        output_fc_W=np.array(params["output_fc_W"]), output_fc_b=np.array(params["output_fc_b"]),)

# ResNet Table 9 - Adam
def train_resnet_adam_step(params, mean_g, mean_g2, x, y, lr, beta1, beta2, key, dropout, eps=1e-9):
    def loss_fn(p):
        return resnet_loss_func(p, x, y, key, dropout=dropout)
    loss, grads = jax.value_and_grad(loss_fn)(params)

    new_params = {}
    new_mean_g = {}
    new_mean_g2 = {}

    for name in params.keys():
        g = grads[name]
        mg = beta1*mean_g[name]+(1.0-beta1)*g
        mg2 = beta2*mean_g2[name]+(1.0-beta2)*(g*g)
        w_new = params[name]-lr* mg/(jnp.sqrt(mg2)+eps)

        new_params[name] = w_new
        new_mean_g[name] = mg
        new_mean_g2[name] = mg2

    return new_params, new_mean_g, new_mean_g2, loss

def fit_adam_resnet(train_x, train_y, test_x, test_y, lr, dropout, beta1=0.9, beta2=0.999, epochs=20, batch_size=64, seed=0):
    key = jax.random.PRNGKey(seed)
    params = init_resnet_params(key)
    mean_g = {k: jnp.zeros_like(v) for k, v in params.items()}
    mean_g2 = {k: jnp.zeros_like(v) for k, v in params.items()}

    train_errors=[]
    test_errors=[]

    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        for xn, yn in mini_batch(train_x, train_y, batch_size, rng):
            xn = jnp.array(xn)
            yn = jnp.array(yn)
            key, subkey = jax.random.split(key)
            (params, mean_g, mean_g2, loss) = train_resnet_adam_step(params, mean_g, mean_g2, xn, yn, lr=lr, beta1=beta1, beta2=beta2, key=subkey, dropout=dropout)

        train_accuracy = float(resnet_accuracy(params, jnp.array(train_x), jnp.array(train_y)))
        train_errors.append(1-train_accuracy)
        test_accuracy = float(resnet_accuracy(params, jnp.array(test_x), jnp.array(test_y)))
        test_errors.append(1-test_accuracy)
        print(f"[ResNet - Adam] epoch={epoch+1:02d} | loss={float(loss):.4f} | test_accuracy={test_accuracy:.4f}")

    return params, train_errors, test_errors

train_x_resnet = train_x.reshape(-1, 1, 32, 32)
test_x_resnet  = test_x.reshape(-1, 1, 32, 32)
params, train_errors, test_errors = fit_adam_resnet(train_x_resnet, train_y, test_x_resnet, test_y, lr=0.001, dropout=0.5)
plot_errors(train_errors, test_errors, "ResNet - Adam | Training Error v/s Epoch | Test Error v/s Epoch")

np.save("saved_models/resnet_table_9/adam_train.npy", np.array(train_errors))
np.save("saved_models/resnet_table_9/adam_test.npy",  np.array(test_errors))
np.savez("saved_models/resnet_table_9/adam_model.npz",
        layer1_conv_W=np.array(params["layer1_conv_W"]), layer1_conv_b=np.array(params["layer1_conv_b"]),
        layer2_conv_W=np.array(params["layer2_conv_W"]), layer2_conv_b=np.array(params["layer2_conv_b"]),
        layer3_projection_W=np.array(params["layer3_projection_W"]), layer3_projection_b=np.array(params["layer3_projection_b"]),
        layer6_conv_W=np.array(params["layer6_conv_W"]), layer6_conv_b=np.array(params["layer6_conv_b"]),
        layer7_conv_W=np.array(params["layer7_conv_W"]), layer7_conv_b=np.array(params["layer7_conv_b"]),
        layer8_projection_W=np.array(params["layer8_projection_W"]), layer8_projection_b=np.array(params["layer8_projection_b"]),
        output_fc_W=np.array(params["output_fc_W"]), output_fc_b=np.array(params["output_fc_b"]),)



