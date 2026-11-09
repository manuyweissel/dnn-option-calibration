"""
Analytic and numerical pricing for Black-Scholes, Local Volatility, and Heston models.

Provides ground-truth option prices and Greeks for training and evaluation.
All implementations are deterministic and vectorized for efficiency.

Heston: Uses Carr-Madan FFT via characteristic function integration.
LocalVol: Uses finite-difference Dupire PDE solver.
BS: Analytic closed-form solution.
"""

from typing import Optional, Tuple
import numpy as np
from scipy import stats
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline
import warnings


class BSPricer:
    """
    Black-Scholes analytic pricer for European call options.
    
    Model: dS_t = r S_t dt + σ S_t dW_t
    
    Attributes:
        S0: Initial underlying price.
        r: Risk-free rate.
        sigma: Volatility.
        
    Example:
        >>> pricer = BSPricer(S0=1.0, r=0.02, sigma=0.2)
        >>> K = np.array([0.9, 1.0, 1.1])
        >>> T = np.array([0.5, 0.5, 0.5])
        >>> prices = pricer.price(K, T)
    """
    
    def __init__(self, S0: float = 1.0, r: float = 0.02, sigma: float = 0.2):
        """
        Initialize BS pricer.
        
        Args:
            S0: Initial underlying price.
            r: Risk-free rate.
            sigma: Volatility (constant).
        """
        self.S0 = S0
        self.r = r
        self.sigma = sigma
    
    def _d1_d2(self, K: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute d1 and d2 for BS formula."""
        T = np.maximum(T, 1e-8)
        d1 = (np.log(self.S0 / K) + (self.r + 0.5 * self.sigma**2) * T) / (
            self.sigma * np.sqrt(T)
        )
        d2 = d1 - self.sigma * np.sqrt(T)
        return d1, d2
    
    def price(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Compute call option prices.
        
        Args:
            K: Strike prices, shape (N,).
            T: Times to maturity, shape (N,).
            
        Returns:
            Option prices, shape (N,).
        """
        K = np.asarray(K)
        T = np.asarray(T)
        T = np.maximum(T, 1e-8)
        
        d1, d2 = self._d1_d2(K, T)
        price = self.S0 * stats.norm.cdf(d1) - K * np.exp(-self.r * T) * stats.norm.cdf(d2)
        
        return price
    
    def delta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute spot-Delta (dC/dS) = N(d1)."""
        d1, _ = self._d1_d2(np.asarray(K), np.maximum(np.asarray(T), 1e-8))
        return stats.norm.cdf(d1)

    def gamma(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute spot-Gamma (d2C/dS2)."""
        T = np.maximum(np.asarray(T), 1e-8)
        d1, _ = self._d1_d2(np.asarray(K), T)
        return stats.norm.pdf(d1) / (self.S0 * self.sigma * np.sqrt(T))

    def vega(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute Vega (dC/dsigma)."""
        T = np.maximum(np.asarray(T), 1e-8)
        d1, _ = self._d1_d2(np.asarray(K), T)
        return self.S0 * stats.norm.pdf(d1) * np.sqrt(T)

    def theta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute spot-Theta (-dC/dT)."""
        K = np.asarray(K)
        T = np.maximum(np.asarray(T), 1e-8)
        d1, d2 = self._d1_d2(K, T)
        term1 = -self.S0 * stats.norm.pdf(d1) * self.sigma / (2 * np.sqrt(T))
        term2 = -self.r * K * np.exp(-self.r * T) * stats.norm.cdf(d2)
        return term1 + term2

    def strike_delta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute strike-Delta (dC/dK) = -e^{-rT} N(d2)."""
        K = np.asarray(K)
        T = np.maximum(np.asarray(T), 1e-8)
        _, d2 = self._d1_d2(K, T)
        return -np.exp(-self.r * T) * stats.norm.cdf(d2)

    def strike_gamma(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute strike-Gamma (d2C/dK2) = e^{-rT} n(d2) / (K sigma sqrt(T))."""
        K = np.asarray(K)
        T = np.maximum(np.asarray(T), 1e-8)
        _, d2 = self._d1_d2(K, T)
        return np.exp(-self.r * T) * stats.norm.pdf(d2) / (K * self.sigma * np.sqrt(T))

    def strike_theta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute strike-Theta (-dC/dT) using analytic derivative of BS formula."""
        K = np.asarray(K)
        T = np.maximum(np.asarray(T), 1e-8)
        d1, d2 = self._d1_d2(K, T)
        dC_dT = (self.S0 * stats.norm.pdf(d1) * self.sigma / (2 * np.sqrt(T))
                 + self.r * K * np.exp(-self.r * T) * stats.norm.cdf(d2))
        return -dC_dT
    
    def implied_vol(
        self, K: np.ndarray, T: np.ndarray, price: np.ndarray, 
        tol: float = 1e-6, max_iter: int = 100
    ) -> np.ndarray:
        """
        Compute implied volatility via Newton-Raphson (vectorized).
        
        Args:
            K: Strike prices.
            T: Times to maturity.
            price: Observed option prices.
            tol: Convergence tolerance.
            max_iter: Maximum iterations.
            
        Returns:
            Implied volatilities, shape (N,).
        """
        K = np.asarray(K)
        T = np.asarray(T)
        price = np.asarray(price)
        
        # Initial guess using Brenner-Subrahmanyam approximation
        sigma = np.sqrt(2 * np.pi / T) * price / self.S0
        sigma = np.clip(sigma, 0.01, 2.0)
        
        for _ in range(max_iter):
            # Vectorized BS pricing at current sigma
            T_safe = np.maximum(T, 1e-8)
            d1 = (np.log(self.S0 / K) + (self.r + 0.5 * sigma**2) * T_safe) / (
                sigma * np.sqrt(T_safe)
            )
            d2 = d1 - sigma * np.sqrt(T_safe)
            
            bs_price = self.S0 * stats.norm.cdf(d1) - K * np.exp(-self.r * T_safe) * stats.norm.cdf(d2)
            vega_val = self.S0 * stats.norm.pdf(d1) * np.sqrt(T_safe)
            
            diff = bs_price - price
            if np.abs(diff).max() < tol:
                break
            
            # Newton update with damping
            sigma = sigma - diff / (vega_val + 1e-10)
            sigma = np.clip(sigma, 0.01, 2.0)
        
        return sigma


class LocalVolPricer:
    """
    Local volatility pricer using Dupire PDE finite-difference solver.
    
    Model: dS_t = r S_t dt + σ_loc(t, S_t) S_t dW_t
    Surface: σ_loc(K, T) = σ_0 * (K / S_0)^α * (1 + β * T)
    
    Prices are computed by solving the Dupire forward PDE:
        ∂C/∂T = (1/2)σ²(K,T) K² ∂²C/∂K² - r K ∂C/∂K
    with initial condition C(K, 0) = max(S0 - K, 0).
    
    Attributes:
        S0: Initial underlying price.
        r: Risk-free rate.
        sigma0: Base volatility.
        alpha: Moneyness exponent.
        beta: Maturity slope.
    """
    
    def __init__(
        self,
        S0: float = 1.0,
        r: float = 0.02,
        sigma0: float = 0.2,
        alpha: float = 0.0,
        beta: float = 0.0,
        n_K: int = 200,
        n_T: int = 100,
    ):
        """Initialize local vol pricer with grid parameters."""
        self.S0 = S0
        self.r = r
        self.sigma0 = sigma0
        self.alpha = alpha
        self.beta = beta
        self.n_K = n_K
        self.n_T = n_T
        
        # Precompute price surface via Dupire PDE
        self._precomputed = False
        self._K_grid = None
        self._T_grid = None
        self._C_grid = None
        self._interp = None
    
    def local_vol_surface(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Compute local volatility surface.
        
        Args:
            K: Strike prices.
            T: Times to maturity.
            
        Returns:
            Local volatilities, shape (N,).
        """
        K = np.asarray(K)
        T = np.asarray(T)
        return self.sigma0 * (K / self.S0) ** self.alpha * (1 + self.beta * T)
    
    def _solve_dupire_pde(self, K_min: float, K_max: float, T_max: float) -> None:
        """
        Solve Dupire forward PDE using Crank-Nicolson finite differences
        with a banded (tridiagonal) solver for O(n) per time-step.

        The forward PDE for call prices is:
            dC/dT = (1/2) sigma^2(K,T) K^2 d2C/dK2 - r K dC/dK

        Initial condition: C(K, 0) = max(S0 - K, 0)
        Boundary conditions:
            C(K_min, T) = S0 - K_min exp(-rT)  (deep ITM)
            C(K_max, T) = 0                     (deep OTM)
        """
        from scipy.linalg import solve_banded

        dK = (K_max - K_min) / (self.n_K - 1)
        dT = T_max / (self.n_T - 1)

        self._K_grid = np.linspace(K_min, K_max, self.n_K)
        self._T_grid = np.linspace(0, T_max, self.n_T)
        nK = self.n_K

        C = np.maximum(self.S0 - self._K_grid, 0.0)
        self._C_grid = np.zeros((self.n_T, nK))
        self._C_grid[0, :] = C.copy()

        for n in range(1, self.n_T):
            T_mid = 0.5 * (self._T_grid[n] + self._T_grid[n - 1])
            sigma_sq = self.local_vol_surface(self._K_grid, T_mid) ** 2
            Karr = self._K_grid

            alpha_c = 0.25 * dT * sigma_sq * Karr ** 2 / dK ** 2
            beta_c = 0.25 * dT * self.r * Karr / dK

            # Central difference of -r K dC/dK: +beta_c on C_{j-1}, -beta_c on C_{j+1}
            sub = -alpha_c - beta_c
            diag = 1.0 + 2.0 * alpha_c
            sup = -alpha_c + beta_c

            sub_r = alpha_c + beta_c
            diag_r = 1.0 - 2.0 * alpha_c
            sup_r = alpha_c - beta_c

            rhs = diag_r * C
            rhs[1:] += sub_r[1:] * C[:-1]
            rhs[:-1] += sup_r[:-1] * C[1:]

            diag[0] = 1.0; sup[0] = 0.0
            diag[-1] = 1.0; sub[-1] = 0.0
            rhs[0] = self.S0 - K_min * np.exp(-self.r * self._T_grid[n])
            rhs[-1] = 0.0

            ab = np.zeros((3, nK))
            ab[0, 1:] = sup[:-1]
            ab[1, :] = diag
            ab[2, :-1] = sub[1:]

            C = solve_banded((1, 1), ab, rhs)
            C = np.maximum(C, 0.0)
            self._C_grid[n, :] = C

        self._interp = RectBivariateSpline(
            self._T_grid, self._K_grid, self._C_grid, kx=3, ky=3
        )
        self._precomputed = True
    
    def price(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Compute option prices via Dupire PDE.

        Args:
            K: Strike prices.
            T: Times to maturity.

        Returns:
            Option prices, shape (N,).
        """
        K = np.asarray(K).flatten()
        T = np.asarray(T).flatten()

        if not self._precomputed:
            K_min = max(0.01, K.min() * 0.5)
            K_max = K.max() * 2.0
            T_max = max(T.max() * 1.5, 1.0)
            self._solve_dupire_pde(K_min, K_max, T_max)

        T_c = np.clip(T, self._T_grid[0], self._T_grid[-1])
        K_c = np.clip(K, self._K_grid[0], self._K_grid[-1])
        prices = self._interp(T_c, K_c, grid=False)
        return np.maximum(prices, 0.0)
    
    def delta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute Delta via finite differences on S0."""
        h = 0.01 * self.S0
        
        # Store original S0
        S0_orig = self.S0
        
        # Price at S0 + h
        self.S0 = S0_orig + h
        self._precomputed = False
        price_up = self.price(K, T)
        
        # Price at S0 - h
        self.S0 = S0_orig - h
        self._precomputed = False
        price_down = self.price(K, T)
        
        # Restore S0
        self.S0 = S0_orig
        self._precomputed = False
        
        return (price_up - price_down) / (2 * h)
    
    def gamma(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute Gamma via finite differences on S0."""
        h = 0.01 * self.S0
        
        S0_orig = self.S0
        
        # Price at S0 + h
        self.S0 = S0_orig + h
        self._precomputed = False
        price_up = self.price(K, T)
        
        # Price at S0
        self.S0 = S0_orig
        self._precomputed = False
        price_mid = self.price(K, T)
        
        # Price at S0 - h
        self.S0 = S0_orig - h
        self._precomputed = False
        price_down = self.price(K, T)
        
        # Restore S0
        self.S0 = S0_orig
        self._precomputed = False
        
        return (price_up - 2 * price_mid + price_down) / (h**2)
    
    def vega(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute Vega via finite differences on sigma0."""
        h = 0.01
        
        sigma0_orig = self.sigma0
        
        # Price at sigma0 + h
        self.sigma0 = sigma0_orig + h
        self._precomputed = False
        price_up = self.price(K, T)
        
        # Price at sigma0 - h
        self.sigma0 = sigma0_orig - h
        self._precomputed = False
        price_down = self.price(K, T)
        
        # Restore sigma0
        self.sigma0 = sigma0_orig
        self._precomputed = False
        
        return (price_up - price_down) / (2 * h)
    
    def theta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute spot-Theta via finite differences on T."""
        h = 0.01
        T = np.asarray(T)
        price_curr = self.price(K, T)
        price_shifted = self.price(K, T + h)
        return -(price_shifted - price_curr) / h

    def strike_delta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute strike-Delta (dC/dK) via central finite differences."""
        K = np.asarray(K).flatten()
        T = np.asarray(T).flatten()
        h = 0.001 * np.maximum(np.abs(K), 0.01)
        price_up = self.price(K + h, T)
        price_down = self.price(K - h, T)
        return (price_up - price_down) / (2 * h)

    def strike_gamma(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute strike-Gamma (d2C/dK2) via central finite differences."""
        K = np.asarray(K).flatten()
        T = np.asarray(T).flatten()
        h = 0.001 * np.maximum(np.abs(K), 0.01)
        price_up = self.price(K + h, T)
        price_mid = self.price(K, T)
        price_down = self.price(K - h, T)
        return (price_up - 2 * price_mid + price_down) / (h**2)

    def strike_theta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute strike-Theta (-dC/dT) via finite differences.

        Uses central differences in the interior and one-sided differences
        at boundary maturities where a symmetric stencil is unavailable.
        """
        K = np.asarray(K).flatten()
        T = np.asarray(T).flatten()
        h = 0.001 * np.maximum(np.abs(T), 0.01)

        dC_dT = np.empty_like(T)
        lower_bnd = T - h <= 0
        interior = ~lower_bnd

        if interior.any():
            p_up = self.price(K[interior], T[interior] + h[interior])
            p_dn = self.price(K[interior], T[interior] - h[interior])
            dC_dT[interior] = (p_up - p_dn) / (2 * h[interior])

        if lower_bnd.any():
            p0 = self.price(K[lower_bnd], T[lower_bnd])
            p_up = self.price(K[lower_bnd], T[lower_bnd] + h[lower_bnd])
            dC_dT[lower_bnd] = (p_up - p0) / h[lower_bnd]

        return -dC_dT


class HestonPricer:
    """
    Heston stochastic volatility pricer using characteristic function integration.
    
    Model:
        dS_t = r S_t dt + √V_t S_t dW^1_t
        dV_t = κ(θ - V_t) dt + ξ √V_t dW^2_t
        <dW^1, dW^2>_t = ρ dt
    
    Prices are computed via the Lewis (2000) formula:
        C = S0 - (√K e^{-rT} / π) ∫_0^∞ Re[e^{-iu log(K)} φ(u - i/2) / (u² + 1/4)] du
    
    where φ(u) is the characteristic function of log(S_T).
    
    Attributes:
        S0: Initial underlying price.
        r: Risk-free rate.
        kappa: Mean reversion rate.
        theta_param: Long-term variance.
        xi: Volatility of variance.
        rho: Correlation between S and V Brownian motions.
        v0: Initial variance.
    """
    
    def __init__(
        self,
        S0: float = 1.0,
        r: float = 0.02,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.5,
        rho: float = -0.5,
        v0: float = 0.04,
    ):
        """Initialize Heston pricer."""
        self.S0 = S0
        self.r = r
        self.kappa = kappa
        self.theta_param = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0
    
    def _characteristic_function(self, u: complex, T: float) -> complex:
        """
        Heston characteristic function φ(u) = E[e^{iu log(S_T)}].
        
        Uses the formulation from Gatheral (2006) for numerical stability.
        
        Args:
            u: Complex argument.
            T: Time to maturity.
            
        Returns:
            Complex value of characteristic function.
        """
        # Parameters
        kappa = self.kappa
        theta = self.theta_param
        xi = self.xi
        rho = self.rho
        v0 = self.v0
        r = self.r
        
        # Heston characteristic function components
        # Using "Little Heston Trap" formulation for numerical stability
        i = 1j
        
        # Auxiliary quantities
        d = np.sqrt((rho * xi * i * u - kappa)**2 + xi**2 * (i * u + u**2))
        g = (kappa - rho * xi * i * u - d) / (kappa - rho * xi * i * u + d)
        
        # Avoid numerical issues
        if np.abs(g) > 1e10:
            g = np.sign(np.real(g)) * 1e10
        
        # Exponential terms
        exp_dT = np.exp(-d * T)
        
        # C and D coefficients
        C = r * i * u * T + (kappa * theta / xi**2) * (
            (kappa - rho * xi * i * u - d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g))
        )
        
        D = ((kappa - rho * xi * i * u - d) / xi**2) * (
            (1 - exp_dT) / (1 - g * exp_dT)
        )
        
        # Characteristic function
        phi = np.exp(C + D * v0 + i * u * np.log(self.S0))
        
        return phi
    
    def _price_single(self, K: float, T: float) -> float:
        """
        Compute single option price via Lewis integration.
        
        Uses the Lewis (2000) formula which is more stable than Carr-Madan.
        
        Args:
            K: Strike price.
            T: Time to maturity.
            
        Returns:
            Option price.
        """
        if T < 1e-8:
            # At expiry, return intrinsic value
            return max(self.S0 - K, 0.0)
        
        def integrand(u):
            """Real part of the integrand for Lewis formula."""
            try:
                # Shift to avoid singularity at u = 0
                phi = self._characteristic_function(u - 0.5j, T)
                
                # Lewis integrand
                numerator = np.exp(-1j * u * np.log(K)) * phi
                denominator = u**2 + 0.25
                
                return np.real(numerator / denominator)
            except (RuntimeWarning, FloatingPointError):
                return 0.0
        
        # Numerical integration
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                integral, _ = quad(integrand, 0, 100, limit=200)
            except Exception:
                # Fallback to BS approximation
                return BSPricer(self.S0, self.r, np.sqrt(self.v0)).price(
                    np.array([K]), np.array([T])
                )[0]
        
        # Lewis formula; phi includes log(S0), so the prefactor is sqrt(K)
        price = self.S0 - (np.sqrt(K) * np.exp(-self.r * T) / np.pi) * integral
        
        # Ensure non-negative and bounded
        price = max(price, 0.0)
        price = min(price, self.S0)  # Call price cannot exceed S0
        
        return price
    
    def price(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Compute call option prices via Heston characteristic function.
        
        Args:
            K: Strike prices.
            T: Times to maturity.
            
        Returns:
            Option prices, shape (N,).
        """
        K = np.asarray(K).flatten()
        T = np.asarray(T).flatten()
        
        prices = np.array([self._price_single(k, t) for k, t in zip(K, T)])
        return prices
    
    def delta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute Delta via finite differences on S0."""
        h = 0.01 * self.S0
        
        S0_orig = self.S0
        
        self.S0 = S0_orig + h
        price_up = self.price(K, T)
        
        self.S0 = S0_orig - h
        price_down = self.price(K, T)
        
        self.S0 = S0_orig
        
        return (price_up - price_down) / (2 * h)
    
    def gamma(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute Gamma via finite differences on S0."""
        h = 0.01 * self.S0
        
        S0_orig = self.S0
        
        self.S0 = S0_orig + h
        price_up = self.price(K, T)
        
        self.S0 = S0_orig
        price_mid = self.price(K, T)
        
        self.S0 = S0_orig - h
        price_down = self.price(K, T)
        
        self.S0 = S0_orig
        
        return (price_up - 2 * price_mid + price_down) / (h**2)
    
    def vega(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute Vega via finite differences on v0 (initial variance)."""
        h = 0.001  # Small step for variance
        
        v0_orig = self.v0
        
        self.v0 = v0_orig + h
        price_up = self.price(K, T)
        
        self.v0 = v0_orig - h
        price_down = self.price(K, T)
        
        self.v0 = v0_orig
        
        # Convert to σ-vega: ∂C/∂σ = 2√v0 * ∂C/∂v0
        return (price_up - price_down) / (2 * h) * 2 * np.sqrt(v0_orig)
    
    def theta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute Theta via finite differences on T."""
        h = 0.01
        T = np.asarray(T)
        
        price_curr = self.price(K, T)
        price_shifted = self.price(K, T + h)
        return -(price_shifted - price_curr) / h

    def strike_delta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute strike-Delta (dC/dK) via central finite differences."""
        K = np.asarray(K).flatten()
        T = np.asarray(T).flatten()
        h = 0.001 * np.maximum(np.abs(K), 0.01)
        price_up = self.price(K + h, T)
        price_down = self.price(K - h, T)
        return (price_up - price_down) / (2 * h)

    def strike_gamma(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute strike-Gamma (d2C/dK2) via central finite differences."""
        K = np.asarray(K).flatten()
        T = np.asarray(T).flatten()
        h = 0.001 * np.maximum(np.abs(K), 0.01)
        price_up = self.price(K + h, T)
        price_mid = self.price(K, T)
        price_down = self.price(K - h, T)
        return (price_up - 2 * price_mid + price_down) / (h**2)

    def strike_theta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute strike-Theta (-dC/dT) via finite differences.

        Uses central differences in the interior and one-sided (forward/backward)
        differences at boundary maturities where a symmetric stencil is unavailable.
        """
        K = np.asarray(K).flatten()
        T = np.asarray(T).flatten()
        h = 0.001 * np.maximum(np.abs(T), 0.01)

        dC_dT = np.empty_like(T)

        lower_bnd = T - h <= 0
        upper_bnd = np.zeros_like(T, dtype=bool)

        interior = ~lower_bnd & ~upper_bnd

        if interior.any():
            p_up = self.price(K[interior], T[interior] + h[interior])
            p_dn = self.price(K[interior], T[interior] - h[interior])
            dC_dT[interior] = (p_up - p_dn) / (2 * h[interior])

        if lower_bnd.any():
            p0 = self.price(K[lower_bnd], T[lower_bnd])
            p_up = self.price(K[lower_bnd], T[lower_bnd] + h[lower_bnd])
            dC_dT[lower_bnd] = (p_up - p0) / h[lower_bnd]

        if upper_bnd.any():
            p0 = self.price(K[upper_bnd], T[upper_bnd])
            p_dn = self.price(K[upper_bnd], T[upper_bnd] - h[upper_bnd])
            dC_dT[upper_bnd] = (p0 - p_dn) / h[upper_bnd]

        return -dC_dT


class MonteCarloPricer:
    """
    Monte Carlo option pricer for baseline comparison.
    
    Can handle any SDE model via Euler-Maruyama discretization.
    Supports variance reduction via antithetic variates.
    
    This provides an independent baseline for validating neural network results.
    """
    
    def __init__(
        self,
        S0: float = 1.0,
        r: float = 0.02,
        sigma: float = 0.2,
        n_paths: int = 100000,
        n_steps: int = 100,
        antithetic: bool = True,
        seed: int = 20251,
    ):
        """Initialize MC pricer."""
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.antithetic = antithetic
        self.seed = seed
    
    def price(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Compute call option prices via Monte Carlo.
        
        Uses Euler-Maruyama scheme for GBM:
            S_{t+dt} = S_t * exp((r - σ²/2)dt + σ√dt Z)
        
        Args:
            K: Strike prices.
            T: Times to maturity.
            
        Returns:
            Option prices, shape (N,).
        """
        K = np.asarray(K).flatten()
        T = np.asarray(T).flatten()
        
        np.random.seed(self.seed)
        prices = np.zeros_like(K, dtype=float)
        
        for i, (k, t) in enumerate(zip(K, T)):
            if t < 1e-8:
                prices[i] = max(self.S0 - k, 0.0)
                continue
            
            dt = t / self.n_steps
            sqrt_dt = np.sqrt(dt)
            
            # Generate paths
            n_paths = self.n_paths // 2 if self.antithetic else self.n_paths
            Z = np.random.randn(n_paths, self.n_steps)
            
            # Simulate paths
            S = np.full(n_paths, self.S0)
            drift = (self.r - 0.5 * self.sigma**2) * dt
            
            for j in range(self.n_steps):
                S = S * np.exp(drift + self.sigma * sqrt_dt * Z[:, j])
            
            # Antithetic paths
            if self.antithetic:
                S_anti = np.full(n_paths, self.S0)
                for j in range(self.n_steps):
                    S_anti = S_anti * np.exp(drift - self.sigma * sqrt_dt * Z[:, j])
                S = np.concatenate([S, S_anti])
            
            # Payoff and discounting
            payoff = np.maximum(S - k, 0.0)
            prices[i] = np.exp(-self.r * t) * np.mean(payoff)
        
        return prices
    
    def delta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Delta via pathwise derivative."""
        h = 0.01 * self.S0
        
        S0_orig = self.S0
        
        self.S0 = S0_orig + h
        price_up = self.price(K, T)
        
        self.S0 = S0_orig - h
        price_down = self.price(K, T)
        
        self.S0 = S0_orig
        
        return (price_up - price_down) / (2 * h)
    
    def gamma(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Gamma via finite differences."""
        h = 0.01 * self.S0
        
        S0_orig = self.S0
        
        self.S0 = S0_orig + h
        price_up = self.price(K, T)
        
        self.S0 = S0_orig
        price_mid = self.price(K, T)
        
        self.S0 = S0_orig - h
        price_down = self.price(K, T)
        
        self.S0 = S0_orig
        
        return (price_up - 2 * price_mid + price_down) / (h**2)
    
    def vega(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Vega via finite differences on sigma."""
        h = 0.01
        
        sigma_orig = self.sigma
        
        self.sigma = sigma_orig + h
        price_up = self.price(K, T)
        
        self.sigma = sigma_orig - h
        price_down = self.price(K, T)
        
        self.sigma = sigma_orig
        
        return (price_up - price_down) / (2 * h)
    
    def theta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Theta via finite differences on T."""
        h = 0.01
        T = np.asarray(T)
        price_curr = self.price(K, T)
        price_shifted = self.price(K, T + h)
        return -(price_shifted - price_curr) / h

    def strike_delta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute strike-Delta (dC/dK) via central finite differences."""
        K = np.asarray(K).flatten()
        T = np.asarray(T).flatten()
        h = 0.001 * np.maximum(np.abs(K), 0.01)
        price_up = self.price(K + h, T)
        price_down = self.price(K - h, T)
        return (price_up - price_down) / (2 * h)

    def strike_gamma(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute strike-Gamma (d2C/dK2) via central finite differences."""
        K = np.asarray(K).flatten()
        T = np.asarray(T).flatten()
        h = 0.001 * np.maximum(np.abs(K), 0.01)
        price_up = self.price(K + h, T)
        price_mid = self.price(K, T)
        price_down = self.price(K - h, T)
        return (price_up - 2 * price_mid + price_down) / (h**2)

    def strike_theta(self, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Compute strike-Theta (-dC/dT) via central finite differences."""
        K = np.asarray(K).flatten()
        T = np.asarray(T).flatten()
        h = 0.001 * np.maximum(np.abs(T), 0.01)
        price_up = self.price(K, T + h)
        price_down = self.price(K, T - h)
        dC_dT = (price_up - price_down) / (2 * h)
        return -dC_dT


# Factory function for creating pricers
def create_pricer(model: str, params: dict):
    """
    Factory function to create pricing model.
    
    Args:
        model: Model name ('BS', 'LocalVol', 'Heston', 'MC').
        params: Model parameters dict.
        
    Returns:
        Pricer instance.
        
    Example:
        >>> pricer = create_pricer('Heston', {'S0': 1.0, 'kappa': 2.0, ...})
    """
    if model == "BS":
        return BSPricer(
            S0=params.get("S0", 1.0),
            r=params.get("r", 0.02),
            sigma=params.get("sigma", 0.2),
        )
    elif model == "LocalVol":
        return LocalVolPricer(
            S0=params.get("S0", 1.0),
            r=params.get("r", 0.02),
            sigma0=params.get("sigma0", params.get("sigma", 0.2)),
            alpha=params.get("alpha", 0.0),
            beta=params.get("beta", 0.0),
        )
    elif model == "Heston":
        return HestonPricer(
            S0=params.get("S0", 1.0),
            r=params.get("r", 0.02),
            kappa=params.get("kappa", 2.0),
            theta=params.get("theta", 0.04),
            xi=params.get("xi", 0.5),
            rho=params.get("rho", -0.5),
            v0=params.get("v0", 0.04),
        )
    elif model == "MC":
        return MonteCarloPricer(
            S0=params.get("S0", 1.0),
            r=params.get("r", 0.02),
            sigma=params.get("sigma", 0.2),
            n_paths=params.get("n_paths", 100000),
            n_steps=params.get("n_steps", 100),
        )
    else:
        raise ValueError(f"Unknown model: {model}")
