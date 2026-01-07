import numpy as np
from typing import Dict, Any, Tuple


def make_noise_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_x_cases(
    Ne: int,
    n_dense: int = 200,
    n_sparse: int = 200,
    seed: int = 123,
    dense_mean: float = 0.0,
    dense_std: float = 1.0,
    sparse_k: Tuple[int, ...] = (1, 2),
    sparse_binary: bool = True,
    sparse_signed: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Returns:
      {"dense": (Ne, n_dense), "sparse": (Ne, n_sparse)}
    """
    rng = np.random.default_rng(seed)

    x_dense = rng.normal(dense_mean, dense_std, size=(Ne, n_dense))

    x_sparse = np.zeros((Ne, n_sparse), dtype=float)
    for j in range(n_sparse):
        k = rng.choice(sparse_k)
        idx = rng.choice(Ne, size=k, replace=False)

        if sparse_binary:
            vals = np.ones(k)
            if sparse_signed:
                vals *= rng.choice([-1.0, 1.0], size=k)
        else:
            # simple alternative: uniform amplitudes (still sparse)
            vals = rng.uniform(0.5, 1.5, size=k)
            if sparse_signed:
                vals *= rng.choice([-1.0, 1.0], size=k)

        x_sparse[idx, j] = vals

    return {"dense": x_dense, "sparse": x_sparse}


def rel_l2_err(x_true: np.ndarray, x_hat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Relative L2 error per case (column):
      ||x - xhat||_2 / (||x||_2 + eps)
    Shapes: (Ne, N)
    Returns: (N,)
    """
    return np.linalg.norm(x_true - x_hat, axis=0) / (np.linalg.norm(x_true, axis=0) + eps)


def run_test1(
    A: np.ndarray,
    x_cases: Dict[str, np.ndarray],
    noise_mu: float = 0.0,
    noise_sigma: float = 0.0,
    noise_seed: int | None = 999,
) -> Dict[str, Any]:
    """
    Test 1 (matrix-only):
      clean:  y = A x, xhat = pinv(A) y
      noisy:  y_noisy = y + eta, eta ~ N(mu, sigma^2)

    Returns all arrays + errors in-memory, no output.
    """
    B = np.linalg.pinv(A)
    rng = make_noise_rng(noise_seed)

    out: Dict[str, Any] = {"A": A, "B": B, "noise": {"mu": noise_mu, "sigma": noise_sigma, "seed": noise_seed}, "cases": {}}

    for name, x_true in x_cases.items():
        y = A @ x_true
        x_hat = B @ y
        err_clean = rel_l2_err(x_true, x_hat)

        eta = rng.normal(loc=noise_mu, scale=noise_sigma, size=y.shape)
        y_noisy = y + eta
        x_hat_noisy = B @ y_noisy
        err_noisy = rel_l2_err(x_true, x_hat_noisy)

        out["cases"][name] = {
            "x_true": x_true,
            "y": y,
            "x_hat": x_hat,
            "err_clean": err_clean,
            "eta": eta,
            "y_noisy": y_noisy,
            "x_hat_noisy": x_hat_noisy,
            "err_noisy": err_noisy,
        }

    return out



if __name__ == "__main__":
    Ne = 9
    A = np.load('/Users/brisarojas/Desktop/phantom_simulation/run_phantom/F.npy')
    x_cases = generate_x_cases(Ne)

    mu_noise = 0.0
    sigma_noise = 0.00001

    results = run_test1(A, x_cases, noise_mu=mu_noise, noise_sigma=sigma_noise, noise_seed=42343242)

    for case_name, case_data in results["cases"].items():
        print(f"Case: {case_name}")
        print(f"  Clean error (mean): {np.mean(case_data['err_clean'])}")
        print(f"  Noisy error (mean): {np.mean(case_data['err_noisy'])}")
    print("Test completed.")
