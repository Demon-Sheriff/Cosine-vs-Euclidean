import time
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import expm_multiply

def get_embeddings(words):
    """
    Generates word embeddings using a pre-trained model.
    """
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Generating embeddings...")
    embeddings = model.encode(words)
    return embeddings

def kl_divergence(P, Q):
    """
    Calculates the KL divergence between two probability distributions.
    """
    return np.sum(P * np.log(P / Q))

def get_perplexity_P(X, perplexity=30.0, tol=1e-5, max_iter=50, metric='euclidean'):
    """
    Calculates the conditional probabilities P_j|i and the joint probabilities P_ij.
    """
    n = X.shape[0]
    P = np.zeros((n, n))
    beta = np.ones(n)
    logU = np.log(perplexity)

    # Calculate pairwise distances
    if metric == 'euclidean':
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    elif metric == 'cosine':
        # For cosine, we use the cosine distance which is 1 - cosine_similarity
        D = squareform(pdist(X, 'cosine'))
    else:
        raise ValueError("Metric not supported")

    # Binary search for beta for each point
    for i in range(n):
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.arange(n) != i]

        for _ in range(max_iter):
            # Compute perplexity and entropy
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            if np.abs(Hdiff) < tol:
                break
            
            if Hdiff > 0:
                betamin = beta[i]
                if betamax == np.inf:
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if betamin == -np.inf:
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
        
        P[i, np.arange(n) != i] = thisP

    # Symmetrize P
    P = (P + P.T) / (2 * n)
    return P

def Hbeta(D, beta):
    """
    Helper function for get_perplexity_P.
    """
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P /= sumP
    return H, P

def tsne_from_scratch(X, n_components=2, perplexity=30.0, n_iter=1000, learning_rate=200.0, metric='euclidean'):
    """
    Custom t-SNE implementation.
    """
    # Initialization
    n = X.shape[0]
    Y = np.random.randn(n, n_components)
    dY = np.zeros((n, n_components))
    iY = np.zeros((n, n_components))
    gains = np.ones((n, n_components))
    min_grad = 1e-7

    # Calculate P matrix
    print(f"Calculating P-values with {metric} distance...")
    P = get_perplexity_P(X, perplexity=perplexity, metric=metric)
    P = P * 4  # Early exaggeration
    P = np.maximum(P, 1e-12)

    # Training
    print("Starting t-SNE optimization...")
    for iter_num in range(n_iter):
        # Compute Q matrix
        sum_Y = np.sum(np.square(Y), 1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y))
        num[np.arange(n), np.arange(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (n_components, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < 0.01] = 0.01
        iY = learning_rate * (gains * dY)
        Y = Y - iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute KL divergence
        if (iter_num + 1) % 100 == 0:
            C = kl_divergence(P, Q)
            print(f"Iteration {iter_num + 1}: KL divergence is {C}")

        # Stop early exaggeration
        if iter_num == 100:
            P = P / 4

    return Y, kl_divergence(P, Q)

def plot_results(Y, words, categories, title, filename):
    """
    Plots the t-SNE results.
    """
    print(f"Generating plot: {title}")
    plt.figure(figsize=(14, 10))
    palette = sns.color_palette("hsv", len(np.unique(categories)))
    sns.scatterplot(x=Y[:, 0], y=Y[:, 1], hue=categories, legend='full', palette=palette)
    
    for i, word in enumerate(words):
        plt.annotate(word, (Y[i, 0], Y[i, 1]))
        
    plt.title(title)
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()

if __name__ == '__main__':
    # Vocabulary with categories
    words = [
        # Countries
        "france", "germany", "china", "japan", "canada", "australia",
        # Animals
        "dog", "cat", "lion", "tiger", "elephant", "monkey",
        # Fruits
        "apple", "banana", "orange", "grape", "strawberry", "pineapple",
        # Verbs
        "run", "jump", "eat", "sleep", "read", "write"
    ]
    categories = np.array(
        [0]*6 + [1]*6 + [2]*6 + [3]*6
    )
    category_names = ["Countries", "Animals", "Fruits", "Verbs"]
    
    # Generate embeddings
    X = get_embeddings(words)

    # --- Experiment 1: t-SNE with Euclidean Distance ---
    print("\n--- Running t-SNE with Euclidean Distance ---")
    start_time_euclidean = time.time()
    Y_euclidean, kl_euclidean = tsne_from_scratch(X, n_components=2, perplexity=10.0, n_iter=1000, learning_rate=200.0, metric='euclidean')
    end_time_euclidean = time.time()
    time_euclidean = end_time_euclidean - start_time_euclidean
    plot_results(Y_euclidean, words, [category_names[c] for c in categories], "t-SNE with Euclidean Distance", "tsne_euclidean.png")

    # --- Experiment 2: t-SNE with Cosine Distance ---
    print("\n--- Running t-SNE with Cosine Distance ---")
    start_time_cosine = time.time()
    Y_cosine, kl_cosine = tsne_from_scratch(X, n_components=2, perplexity=10.0, n_iter=1000, learning_rate=200.0, metric='cosine')
    end_time_cosine = time.time()
    time_cosine = end_time_cosine - start_time_cosine
    plot_results(Y_cosine, words, [category_names[c] for c in categories], "t-SNE with Cosine Distance", "tsne_cosine.png")

    # --- Benchmark Results ---
    print("\n--- Benchmark Results ---")
    print(f"Euclidean Distance t-SNE:")
    print(f"  - Computation Time: {time_euclidean:.2f} seconds")
    print(f"  - Final KL Divergence: {kl_euclidean:.4f}")
    print(f"\nCosine Distance t-SNE:")
    print(f"  - Computation Time: {time_cosine:.2f} seconds")
    print(f"  - Final KL Divergence: {kl_cosine:.4f}")

    # --- Comparison with scikit-learn's t-SNE ---
    print("\n--- Running scikit-learn's t-SNE for comparison ---")
    # scikit-learn with euclidean
    tsne_sklearn_euc = TSNE(n_components=2, perplexity=10.0, n_iter=1000, learning_rate='auto', init='random', metric='euclidean')
    Y_sklearn_euc = tsne_sklearn_euc.fit_transform(X)
    plot_results(Y_sklearn_euc, words, [category_names[c] for c in categories], "scikit-learn t-SNE (Euclidean)", "tsne_sklearn_euclidean.png")
    
    # scikit-learn with cosine
    tsne_sklearn_cos = TSNE(n_components=2, perplexity=10.0, n_iter=1000, learning_rate='auto', init='random', metric='cosine')
    Y_sklearn_cos = tsne_sklearn_cos.fit_transform(X)
    plot_results(Y_sklearn_cos, words, [category_names[c] for c in categories], "scikit-learn t-SNE (Cosine)", "tsne_sklearn_cosine.png")

    print("\nExperiment finished. Check the generated PNG files for visualizations.")
