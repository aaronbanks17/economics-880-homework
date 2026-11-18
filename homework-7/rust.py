
import os
import numpy as np
import pandas as pd
import chex
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import xlogy, comb
import logging


@chex.dataclass
class Data:
    # bus attributes
    x: np.array # mileage at end-of-month
    d: np.array # replacement indicator
    dx: np.array # mileage change month-to-month
    dd: np.array # next-month replacement indicator
    # indices
    ids: np.array # bus ids
    types: np.array # bus types
    years: np.array # years
    months: np.array # months
    is_first: np.array # first observed month by bus
    is_last: np.array # last observed month by bus


def DGP():

    # set directories
    cwd = os.getcwd()
    data = os.path.join(cwd, "data")

    # load bus data
    url = os.path.join(data, "busdata.csv")
    columns = [
        "id", "type", "year", "month",
        "repl", "odo_start",
        "odo_end", "odo_end_total",
        "odo_delta",
    ]
    df = pd.read_csv(url, header=None, names=columns)

    # sort indices
    df.sort_values(["id", "year", "month"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # generate ID variables
    ids = df.id.to_numpy()
    types = df.type.to_numpy()
    years = df.year.to_numpy()
    months = df.month.to_numpy()

    # fetch indices for first and last obs
    is_first = ~df.duplicated("id").to_numpy()
    is_last = ~df.duplicated("id", keep="last").to_numpy()

    # set mileage
    x = df.odo_end.to_numpy(dtype=float)

    # set replacement indicator
    d = df.repl.to_numpy(dtype=float)

    # compute mileage increments given replacement
    x_start = df.odo_start.to_numpy()
    dx = x - (1 - d) * x_start

    # construct next-period replacement indicator
    dd = (
        df.groupby("id")["repl"]
        .shift(-1)
        .to_numpy()
    )

    # collect panel data
    _dict = dict(
        x=x,
        d=d,
        dx=dx,
        dd=dd,
        ids=ids,
        types=types,
        years=years,
        months=months,
        is_first=is_first,
        is_last=is_last
    )

    return Data(_dict)


def odometer_sample(data:Data, exclude_initial:bool=True) -> np.array:
    """Extract mileage increments for lognormal likelihood"""

    # restrict to positive mileage increments
    ix = data.dx > 0

    # optionally exclude initial observations
    if exclude_initial:
        ix = ix & ~data.is_first

    # return mileage sample
    return data.dx[ix]


def replacement_sample(data:Data) -> tuple:
    """Extract state and replacement outcomes for hazard likelihood"""

    # exclude observations without next-period decisions
    ix = ~data.is_last & np.isfinite(data.dd)

    # set mileage states
    x = data.x[ix]

    # set corresponding replacement outcomes
    dd = data.dd[ix].astype(int)

    return x, dd


# -- Question 1 -- #

def estimate_lognormal(
    data:Data,
    exclude_initial:bool=True
    ) -> tuple:
    """Estimate lognormal mileage parameters analytically"""

    # extract mileage sample
    dx = odometer_sample(data, exclude_initial)

    # compute log mileage increments
    log_dx = np.log(dx)

    # estimate normal parameters
    mu_hat = log_dx.mean()
    sigma_hat = log_dx.std(ddof=0)

    return mu_hat, sigma_hat


def estimate_hazards(
    data:Data,
    N:int=40
    ) -> tuple:
    """Estimate discretized replacement hazards analytically"""

    # extract replacement sample
    x, dd = replacement_sample(data)

    # construct mileage grid
    x_bar = data.x.max() + 1
    grid = np.linspace(0, x_bar, N + 1)

    # assign observations to mileage bins
    bin_ids = np.searchsorted(grid, x, side="right") - 1

    # count observations by mileage bin
    counts = np.bincount(bin_ids, minlength=N)

    # count replacements by mileage bin
    replacements = np.bincount(
        bin_ids,
        weights=dd,
        minlength=N
    )

    # estimate replacement hazards
    h_hat = np.divide(
        replacements,
        counts,
        out=np.full(N, np.nan),
        where=counts > 0
    )

    return h_hat, grid, counts, replacements


def odometer_log_likelihood(
    mu:float,
    sigma:float,
    data:Data,
    exclude_initial:bool=True
    ) -> float:
    """Compute lognormal mileage log likelihood"""

    # extract mileage sample
    dx = odometer_sample(data, exclude_initial)

    # check parameter support
    if sigma <= 0 or not np.isfinite(sigma):
        return -np.inf

    # compute odometer log likelihood
    O = np.sum(
        -(np.log(dx) - mu)**2 / (2 * sigma**2)
        - np.log(dx)
        - np.log(sigma)
        - 0.5 * np.log(2 * np.pi)
    )

    return O


def log_likelihood_q1(
    mu:float,
    sigma:float,
    h:np.array,
    data:Data,
    exclude_initial:bool=True
    ) -> float:
    """Compute log likelihood for discretized hazard model"""

    # -- odometer likelihood -- #

    # compute odometer log likelihood
    O = odometer_log_likelihood(
        mu, sigma, data, exclude_initial
    )

    # -- replacement likelihood -- #

    # extract replacement sample
    x, dd = replacement_sample(data)

    # construct mileage grid
    N = len(h)
    grid = np.linspace(0, data.x.max() + 1, N + 1)

    # assign observations to mileage bins
    bin_ids = np.searchsorted(grid, x, side="right") - 1

    # recover replacement probabilities
    probs = h[bin_ids]

    # compute replacement log likelihood
    R = np.sum(
        xlogy(dd, probs)
        + xlogy(1 - dd, 1 - probs)
    )

    return O + R


# -- Question 2 -- #

def bernstein_basis(
    x:np.array,
    n:int,
    x_bar:float
    ) -> np.array:
    """Construct Bernstein polynomial basis for engine mileage"""

    # normalize mileage
    z = np.asarray(x, dtype=float) / x_bar

    # construct polynomial basis
    B = np.column_stack([
        comb(n, k) * z**k * (1 - z)**(n-k)
        for k in range(1, n + 1)
    ])

    return B


def bernstein_hazard(
    f:np.array,
    x:np.array,
    x_bar:float
    ) -> np.array:
    """Compute replacement hazard using Bernstein polynomial"""

    # construct Bernstein polynomial basis
    B = bernstein_basis(x, len(f), x_bar)

    # compute replacement hazards
    return B @ f


def log_likelihood_q2(
    f:np.array,
    data:Data
    ) -> float:
    """Compute replacement log likelihood for Bernstein hazard"""

    # extract replacement sample
    x, dd = replacement_sample(data)

    # set mileage upper bound
    x_bar = data.x.max()

    # compute replacement probabilities
    h = bernstein_hazard(f, x, x_bar)

    # compute replacement log likelihood
    R = np.sum(
        xlogy(dd, h)
        + xlogy(1 - dd, 1 - h)
    )

    return R


def Q2_objective(
    f:np.array,
    B:np.array,
    dd:np.array
    ) -> float:
    """Compute negative replacement log likelihood"""

    # check coefficient constraints
    if not np.all(np.isfinite(f)):
        return np.inf

    if np.any(f < 0) or np.any(f > 1):
        return np.inf

    # compute replacement hazards
    h = B @ f

    # check probability constraints
    if not np.all(np.isfinite(h)):
        return np.inf

    if np.any(h < 0) or np.any(h > 1):
        return np.inf

    # compute negative replacement log likelihood
    q = -np.sum(
        xlogy(dd, h)
        + xlogy(1 - dd, 1 - h)
    )

    return q


def Q2_derivatives(
    f:np.array,
    B:np.array,
    dd:np.array
    ) -> tuple:
    """Compute gradient and Hessian of negative log likelihood"""

    # compute replacement hazards
    h = B @ f

    # construct derivatives of Bernoulli likelihood
    # use separate cases to accommodate valid boundary hazards
    replaced = dd == 1
    weights_grad = np.empty_like(h)
    weights_hess = np.empty_like(h)

    weights_grad[replaced] = -1 / h[replaced]
    weights_grad[~replaced] = 1 / (1 - h[~replaced])

    weights_hess[replaced] = 1 / h[replaced]**2
    weights_hess[~replaced] = 1 / (1 - h[~replaced])**2

    # compute gradient of negative log likelihood
    grad = B.T @ weights_grad

    # compute Hessian of negative log likelihood
    hess = B.T @ (weights_hess[:, None] * B)

    return grad, hess


def projected_gradient(
    f:np.array,
    grad:np.array
    ) -> float:
    """Compute projected gradient norm for box constraints"""

    # project one gradient step onto coefficient bounds
    projected = f - np.clip(f - grad, 0, 1)

    # compute infinity norm
    return np.linalg.norm(projected, ord=np.inf)


def newton_standard(
    f_0:np.array,
    B:np.array,
    dd:np.array,
    tol:float=1e-8,
    max_iter:int=200,
    verbose:bool=False
    ) -> tuple:
    """Maximize replacement likelihood using standard Newton-Raphson"""

    # initialize coefficients
    f = np.asarray(f_0, dtype=float).copy()

    # initialize history
    history = dict(
        iteration=[],
        objective=[],
        error=[],
        step_change=[],
        reason=None
    )

    for iteration in range(max_iter):

        # evaluate objective at current coefficients
        q = Q2_objective(f, B, dd)

        # check parameter support
        if not np.isfinite(q):
            history["reason"] = "Invalid parameter values"
            return f, history, False

        # compute gradient and Hessian
        grad, hess = Q2_derivatives(f, B, dd)

        # check derivative validity
        if not (
            np.all(np.isfinite(grad))
            and np.all(np.isfinite(hess))
        ):
            history["reason"] = "Non-finite derivatives"
            return f, history, False

        # compute convergence criterion
        err = np.linalg.norm(grad, ord=np.inf)

        # check convergence
        if err < tol:
            history["reason"] = "Converged"
            return f, history, True

        # compute Newton step
        try:
            step = np.linalg.solve(hess, grad)

        except np.linalg.LinAlgError:
            history["reason"] = "Singular Hessian"
            return f, history, False

        # update coefficients using full Newton step
        f_new = f - step

        # compute change in coefficients
        step_change = np.linalg.norm(step, ord=np.inf)

        # save iteration information
        history["iteration"].append(iteration)
        history["objective"].append(q)
        history["error"].append(err)
        history["step_change"].append(step_change)

        # verbose output
        if verbose:
            logging.info(
                "Standard Newton Iter %5d | "
                "Objective: %12.6f | "
                "Error: %12.4e | "
                "Step: %12.4e",
                iteration, q, err, step_change
            )

        # check whether Newton step leaves parameter space
        if not np.all(np.isfinite(f_new)):
            history["reason"] = "Non-finite Newton step"
            return f_new, history, False

        if np.any(f_new < 0) or np.any(f_new > 1):
            history["reason"] = "Newton step outside parameter bounds"
            return f_new, history, False

        # check likelihood at proposed coefficients
        q_new = Q2_objective(f_new, B, dd)

        if not np.isfinite(q_new):
            history["reason"] = "Non-finite likelihood after Newton step"
            return f_new, history, False

        # update coefficients
        f = f_new

    # check convergence after final iteration
    grad, _ = Q2_derivatives(f, B, dd)
    converged = np.linalg.norm(grad, ord=np.inf) < tol

    history["reason"] = (
        "Converged" if converged else "Maximum iterations reached"
    )

    return f, history, converged


def newton_raphson(
    f_0:np.array,
    B:np.array,
    dd:np.array,
    tol:float=1e-8,
    max_iter:int=200,
    verbose:bool=False
    ) -> tuple:
    """Maximize replacement likelihood using damped Newton-Raphson"""

    # initialize coefficients
    f = np.asarray(f_0, dtype=float).copy()
    q = Q2_objective(f, B, dd)

    if not np.isfinite(q):
        raise ValueError("Initial guess has non-finite likelihood")

    # initialize history
    history = dict(
        iteration=[],
        objective=[],
        error=[],
        step_change=[]
    )

    for iteration in range(max_iter):

        # compute gradient and Hessian
        grad, hess = Q2_derivatives(f, B, dd)

        # check first order conditions with coefficient bounds
        err = projected_gradient(f, grad)

        if err < tol:
            return f, history, True

        # identify coefficients constrained at boundaries
        active = (
            ((f <= 1e-10) & (grad > 0))
            | ((f >= 1 - 1e-10) & (grad < 0))
        )
        free = ~active

        # initialize Newton direction
        step = np.zeros_like(f)

        if np.any(free):

            # extract Hessian for free coefficients
            H = hess[np.ix_(free, free)]

            # regularize nearly singular Hessian
            H = H + 1e-8 * np.eye(free.sum())

            # compute Newton direction
            try:
                step[free] = -np.linalg.solve(H, grad[free])

            except np.linalg.LinAlgError:
                step[free] = -grad[free]

        # construct fallback gradient direction
        gradient_step = -grad / max(
            1.0, np.max(np.abs(grad))
        )

        # initialize line search
        accepted = False

        for direction in [step, gradient_step]:

            step_size = 1.0

            for _ in range(60):

                # construct feasible coefficient update
                f_new = np.clip(
                    f + step_size * direction, 0, 1
                )

                # compute change in coefficients
                delta = f_new - f

                # evaluate updated objective
                q_new = Q2_objective(f_new, B, dd)

                # check sufficient decrease
                if (
                    grad @ delta < 0
                    and np.isfinite(q_new)
                    and q_new <= q + 1e-4 * (grad @ delta)
                ):
                    accepted = True
                    break

                # reduce step size
                step_size *= 0.5

            if accepted:
                break

        if not accepted:
            return f, history, False

        # compute change in coefficients
        step_change = np.linalg.norm(
            f_new - f, ord=np.inf
        )

        # update coefficients and objective
        f = f_new
        q = q_new

        # compute error at updated coefficients
        grad, _ = Q2_derivatives(f, B, dd)
        err = projected_gradient(f, grad)

        # save iteration information
        history["iteration"].append(iteration)
        history["objective"].append(q)
        history["error"].append(err)
        history["step_change"].append(step_change)

        # verbose output
        if verbose:
            logging.info(
                "Newton Iter %5d | Objective: %12.6f | "
                "Error: %12.4e | Step: %12.4e",
                iteration, q, err, step_change
            )

    # check convergence after final iteration
    grad, _ = Q2_derivatives(f, B, dd)
    converged = projected_gradient(f, grad) < tol

    return f, history, converged


def nelder_mead(
    f_0:np.array,
    B:np.array,
    dd:np.array,
    tol:float=1e-8,
    max_iter:int=1000,
    initial_step:float=0.05,
    verbose:bool=False
    ) -> tuple:
    """Maximize replacement likelihood using Nelder-Mead"""

    # initialize coefficients
    f_0 = np.asarray(f_0, dtype=float)
    n = len(f_0)

    if initial_step <= 0 or initial_step > 1:
        raise ValueError("initial_step must lie in (0,1]")

    # construct initial simplex
    simplex = np.tile(f_0, (n + 1, 1))

    for j in range(n):

        # perturb one coefficient at a time
        shift = (
            initial_step
            if f_0[j] + initial_step <= 1
            else -initial_step
        )
        simplex[j+1, j] += shift

    # compute objective at simplex vertices
    values = np.array([
        Q2_objective(f, B, dd)
        for f in simplex
    ])

    if not np.all(np.isfinite(values)):
        raise ValueError("Initial simplex has non-finite likelihood")

    # initialize history
    history = dict(
        iteration=[],
        objective=[],
        error=[],
        step_change=[]
    )

    for iteration in range(max_iter):

        # order vertices from best to worst
        ix = np.argsort(values)
        simplex = simplex[ix]
        values = values[ix]

        # compute simplex diameter and objective spread
        spread = np.max(
            np.linalg.norm(
                simplex - simplex[0],
                ord=np.inf,
                axis=1
            )
        )
        err = max(spread, np.ptp(values))

        # check convergence
        if err < tol:
            return simplex[0].copy(), history, True

        # compute centroid excluding worst vertex
        centroid = simplex[:-1].mean(axis=0)

        # -- reflection -- #

        # reflect worst vertex through centroid
        reflected = np.clip(
            centroid + (centroid - simplex[-1]), 0, 1
        )
        q_reflected = Q2_objective(reflected, B, dd)

        if q_reflected < values[0]:

            # -- expansion -- #

            # expand beyond reflected vertex
            expanded = np.clip(
                centroid + 2 * (reflected - centroid), 0, 1
            )
            q_expanded = Q2_objective(expanded, B, dd)

            # choose better of expansion and reflection
            if q_expanded < q_reflected:
                simplex[-1] = expanded
                values[-1] = q_expanded
            else:
                simplex[-1] = reflected
                values[-1] = q_reflected

        elif q_reflected < values[-2]:

            # accept reflection
            simplex[-1] = reflected
            values[-1] = q_reflected

        else:

            # -- contraction -- #

            if q_reflected < values[-1]:

                # outside contraction
                contracted = np.clip(
                    centroid + 0.5 * (reflected - centroid), 0, 1
                )
                q_contracted = Q2_objective(
                    contracted, B, dd
                )
                accept = q_contracted <= q_reflected

            else:

                # inside contraction
                contracted = np.clip(
                    centroid + 0.5 * (simplex[-1] - centroid), 0, 1
                )
                q_contracted = Q2_objective(
                    contracted, B, dd
                )
                accept = q_contracted < values[-1]

            if accept:

                # accept contraction
                simplex[-1] = contracted
                values[-1] = q_contracted

            else:

                # -- shrinkage -- #

                # shrink simplex toward best vertex
                for j in range(1, n + 1):

                    simplex[j] = np.clip(
                        simplex[0]
                        + 0.5 * (simplex[j] - simplex[0]),
                        0, 1
                    )
                    values[j] = Q2_objective(
                        simplex[j], B, dd
                    )

        # save iteration information
        history["iteration"].append(iteration)
        history["objective"].append(values.min())
        history["error"].append(err)
        history["step_change"].append(spread)

        # verbose output
        if verbose:
            logging.info(
                "Nelder-Mead Iter %5d | Objective: %12.6f | "
                "Error: %12.4e",
                iteration, values.min(), err
            )

    # recover best vertex
    ix = np.argmin(values)

    return simplex[ix].copy(), history, False


def estimate_q2(
    data:Data,
    method:str="hybrid",
    n:int=5,
    f_0:np.array=None,
    tol:float=1e-8,
    max_iter:int=1000,
    nm_iters:int=30,
    initial_step:float=0.05,
    exclude_initial:bool=True,
    verbose:bool=False
    ) -> dict:
    """Estimate Bernstein hazard and lognormal parameters"""

    # -- estimate odometer parameters -- #

    # estimate lognormal parameters analytically
    mu_hat, sigma_hat = estimate_lognormal(
        data, exclude_initial=exclude_initial
    )

    # compute odometer log likelihood
    O = odometer_log_likelihood(
        mu_hat, sigma_hat, data, exclude_initial
    )

    # -- construct replacement likelihood -- #

    # extract replacement sample
    x, dd = replacement_sample(data)

    # construct Bernstein polynomial basis
    B = bernstein_basis(x, n, data.x.max())

    # initialize coefficients
    if f_0 is None:
        f_0 = np.full(n, 0.5)

    f_0 = np.asarray(f_0, dtype=float)

    # check initial coefficient dimensions and bounds
    if f_0.shape != (n,):
        raise ValueError("Incorrect initial coefficient dimensions")

    if (
        not np.all(np.isfinite(f_0))
        or np.any(f_0 < 0)
        or np.any(f_0 > 1)
    ):
        raise ValueError("Initial coefficients must be in [0,1]")

    # initialize histories
    histories = {}

    # -- choose optimization method -- #

    if method == "newton-standard":

        # estimate using unrestricted Newton-Raphson
        f_hat, history, converged = newton_standard(
            f_0, B, dd,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose
        )

        histories["newton-standard"] = history

    elif method == "newton":

        # estimate using constrained Newton-Raphson
        f_hat, history, converged = newton_raphson(
            f_0, B, dd,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose
        )

        histories["newton"] = history

    elif method == "nelder-mead":

        # estimate using Nelder-Mead
        f_hat, history, converged = nelder_mead(
            f_0, B, dd,
            tol=tol,
            max_iter=max_iter,
            initial_step=initial_step,
            verbose=verbose
        )

        histories["nelder-mead"] = history

    elif method == "hybrid":

        # initial Nelder-Mead iterations
        f_nm, history_nm, _ = nelder_mead(
            f_0, B, dd,
            tol=tol,
            max_iter=nm_iters,
            initial_step=initial_step,
            verbose=verbose
        )

        # switch to constrained Newton-Raphson
        f_hat, history_nr, converged = newton_raphson(
            f_nm, B, dd,
            tol=tol,
            max_iter=max_iter,
            verbose=verbose
        )

        histories["nelder-mead"] = history_nm
        histories["newton"] = history_nr

    else:
        raise ValueError(
            "method must be 'newton-standard', 'newton', "
            "'nelder-mead', or 'hybrid'"
        )

    # -- recover estimation results -- #

    # evaluate objective at estimated coefficients
    q = Q2_objective(f_hat, B, dd)

    # check whether estimated coefficients are feasible
    if np.isfinite(q):

        # evaluate first order conditions
        grad, _ = Q2_derivatives(f_hat, B, dd)

        # use unrestricted gradient for standard Newton
        if method == "newton-standard":
            error = np.linalg.norm(grad, ord=np.inf)
            error_type = "gradient"

        else:
            error = projected_gradient(f_hat, grad)
            error_type = "projected gradient"

        # check optimizer convergence
        converged = (
            converged
            and np.isfinite(error)
            and error < max(tol, 1e-4)
        )

        # compute replacement and full log likelihood
        R = -q
        logL_hat = O + R

    else:

        # invalid estimates from unrestricted optimization
        error = np.nan
        error_type = (
            "gradient"
            if method == "newton-standard"
            else "projected gradient"
        )
        converged = False
        R = np.nan
        logL_hat = np.nan

    # retrieve convergence status
    if method == "newton-standard":

        # record reason for standard Newton outcome
        reason = histories["newton-standard"].get("reason")

    elif converged:
        reason = "Converged"

    else:
        reason = "Convergence criteria not satisfied"

    # collect estimation results
    results = dict(
        mu=mu_hat,
        sigma=sigma_hat,
        f=f_hat,
        logL=logL_hat,
        odometer_logL=O,
        replacement_logL=R,
        converged=converged,
        method=method,
        error=error,
        error_type=error_type,
        reason=reason,
        history=histories
    )

    return results


if __name__=="__main__":

    # make output directory
    output = os.path.join(os.getcwd(), "output")
    os.makedirs(output, exist_ok=True)

    # generate data
    data = DGP()

    # -- Question 1 -- #

    # estimate lognormal parameters
    mu_hat, sigma_hat = estimate_lognormal(data)

    # estimate discretized hazards
    h_hat, grid, counts, replacements = estimate_hazards(
        data=data, N=40
    )

    # evaluate maximized log likelihood
    logL_hat = log_likelihood_q1(
        mu=mu_hat,
        sigma=sigma_hat,
        h=h_hat,
        data=data
    )

    # print estimates
    print(f"Mu: {mu_hat:.6f}")
    print(f"Sigma: {sigma_hat:.6f}")
    print(f"Log likelihood: {logL_hat:.6f}")
    print("Replacement hazards:")
    print(h_hat)

    # -- plot replacement CDF -- #

    # compute cumulative replacement probabilities
    cdf = np.cumsum(replacements) / np.sum(replacements)

    # include initial probability of zero
    cdf = np.concatenate([[0], cdf])

    # plot empirical replacement mileage CDF
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Set2", 2)

    ax.step(
        grid,
        cdf,
        where="post",
        color=colors[0],
        linewidth=2
    )

    ax.set_xlabel("Engine Mileage")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(grid[0], grid[-1])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    sns.despine()

    # save figure
    url = os.path.join(output, "hazard_cdf.pdf")
    plt.savefig(url, bbox_inches="tight")
    plt.close()

    # -- Question 2 -- #

    # set Bernstein polynomial order
    n = 5

    # set initial coefficients
    f_0 = np.full(n, 0.5)

    # estimate using standard Newton-Raphson
    results_standard = estimate_q2(
        data=data,
        method="newton-standard",
        n=n,
        f_0=f_0,
        verbose=False
    )

    # estimate using constrained Newton-Raphson
    results_nr = estimate_q2(
        data=data,
        method="newton",
        n=n,
        f_0=f_0,
        verbose=False
    )

    # estimate using Nelder-Mead
    results_nm = estimate_q2(
        data=data,
        method="nelder-mead",
        n=n,
        f_0=f_0,
        verbose=False
    )

    # estimate using hybrid method
    results_hybrid = estimate_q2(
        data=data,
        method="hybrid",
        n=n,
        f_0=f_0,
        nm_iters=30,
        verbose=False
    )

    # print estimates
    for results in [
        results_standard,
        results_nr,
        results_nm,
        results_hybrid
    ]:

        print(f"\nMethod: {results['method']}")
        print(f"Converged: {results['converged']}")
        print(f"Reason: {results['reason']}")
        print(f"Mu: {results['mu']:.6f}")
        print(f"Sigma: {results['sigma']:.6f}")
        print(f"Bernstein coefficients: {results['f']}")
        print(f"Log likelihood: {results['logL']:.6f}")
        print(
            f"{results['error_type']} error: "
            f"{results['error']:.3e}"
        )

    # -- sensitivity to initial guesses -- #

    # set initial coefficient values
    initial_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # initialize results
    sensitivity = []

    for method in [
        "newton-standard",
        "newton",
        "nelder-mead",
        "hybrid"
    ]:

        for value in initial_values:

            # construct initial coefficients
            f_0 = np.full(n, value)

            # estimate model
            results = estimate_q2(
                data=data,
                method=method,
                n=n,
                f_0=f_0,
                verbose=False
            )

            # collect estimation results
            sensitivity.append(dict(
                method=method,
                initial=value,
                logL=results["logL"],
                converged=results["converged"],
                reason=results["reason"],
                error=results["error"],
                **{
                    f"f_{k+1}": results["f"][k]
                    for k in range(n)
                }
            ))

    # construct results dataframe
    sensitivity = pd.DataFrame(sensitivity)

    # save sensitivity results
    url = os.path.join(output, "initial_guess_sensitivity.csv")
    sensitivity.to_csv(url, index=False)

    print("\nInitial guess sensitivity:")
    print(sensitivity.to_string(index=False))
