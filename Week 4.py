from __future__ import annotations

import math
import os
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Input, Lambda
from keras.models import Model


# ============================================================
# Reproducibility
# ============================================================
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# Config containers
# ============================================================
@dataclass
class TrainingConfig:
    train_paths: int = 4_000
    test_paths: int = 1_500
    batch_size: int = 512
    epochs: int = 10
    learning_rate: float = 1e-3
    hidden_units: tuple[int, ...] = (32, 32)


@dataclass
class MarketConfig:
    s0: float = 100.0
    k: float = 100.0
    t: float = 1.0 / 12.0
    n_steps: int = 12
    r: float = 0.0


# ============================================================
# Market model and payoffs
# ============================================================
def simulate_black_scholes_paths(
    n_paths: int,
    *,
    s0: float,
    sigma: float,
    t: float,
    n_steps: int,
    r: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)],
        axis=1,
    )
    return (s0 * np.exp(log_paths)).astype(np.float32)


def european_call_payoff(paths: np.ndarray, strike: float) -> np.ndarray:
    return np.maximum(paths[:, -1] - strike, 0.0).astype(np.float32).reshape(-1, 1)


def barrier_exercised_payoff(
    paths: np.ndarray,
    *,
    barrier: float,
    strike: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hit_matrix = paths[:, 1:] >= barrier
    hit_any = hit_matrix.any(axis=1)
    first_hit = np.argmax(hit_matrix, axis=1) + 1
    first_hit = np.where(hit_any, first_hit, -1)

    payoff = np.zeros(paths.shape[0], dtype=np.float32)
    if np.any(hit_any):
        payoff[hit_any] = paths[np.where(hit_any)[0], first_hit[hit_any]] - strike
    return payoff.reshape(-1, 1), first_hit, hit_any


# ============================================================
# Black-Scholes reference formulas for Exercise 2
# ============================================================
def normal_cdf(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def bs_call_price(
    s: np.ndarray | float,
    *,
    strike: float,
    sigma: float,
    tau: np.ndarray | float,
    r: float = 0.0,
) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    tau = np.maximum(np.asarray(tau, dtype=np.float64), 1e-10)
    d1 = (np.log(s / strike) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return s * normal_cdf(d1) - strike * np.exp(-r * tau) * normal_cdf(d2)


def bs_call_delta(
    s: np.ndarray | float,
    *,
    strike: float,
    sigma: float,
    tau: np.ndarray | float,
    r: float = 0.0,
) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    tau = np.maximum(np.asarray(tau, dtype=np.float64), 1e-10)
    d1 = (np.log(s / strike) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return normal_cdf(d1)


# ============================================================
# Deep hedging model
# ============================================================
def build_strategy_network(
    input_dim: int,
    *,
    hidden_units: tuple[int, ...],
    output_mode: str,
    position_scale: float = 1.0,
) -> Model:
    inputs = Input(shape=(input_dim,))
    x = inputs
    for width in hidden_units:
        x = Dense(width, activation="tanh")(x)

    if output_mode == "sigmoid":
        outputs = Dense(1, activation="sigmoid")(x)
    elif output_mode == "scaled_tanh":
        raw = Dense(1, activation="linear")(x)
        outputs = Lambda(lambda z: position_scale * tf.tanh(z))(raw)
    else:
        outputs = Dense(1, activation="linear")(x)

    return Model(inputs=inputs, outputs=outputs)


class DeepHedger:
    def __init__(
        self,
        *,
        n_steps: int,
        input_dim: int,
        hidden_units: tuple[int, ...],
        output_mode: str,
        premium_init: float,
        position_scale: float = 1.0,
    ) -> None:
        self.n_steps = n_steps
        self.premium = tf.Variable(
            premium_init,
            trainable=True,
            dtype=tf.float32,
            name="initial_premium",
        )
        self.networks = [
            build_strategy_network(
                input_dim,
                hidden_units=hidden_units,
                output_mode=output_mode,
                position_scale=position_scale,
            )
            for _ in range(n_steps)
        ]

    @property
    def trainable_variables(self) -> list[tf.Variable]:
        variables = [self.premium]
        for network in self.networks:
            variables.extend(network.trainable_variables)
        return variables


# ============================================================
# State features
# ============================================================
def european_state(
    s_t: tf.Tensor,
    *,
    step: int,
    n_steps: int,
    strike: float,
) -> tf.Tensor:
    time_left = tf.ones_like(s_t) * ((n_steps - step) / n_steps)
    return tf.concat([tf.math.log(s_t / strike), s_t / strike - 1.0, time_left], axis=1)


def barrier_state(
    s_t: tf.Tensor,
    *,
    step: int,
    n_steps: int,
    strike: float,
    barrier: float,
    active: tf.Tensor,
) -> tf.Tensor:
    time_left = tf.ones_like(s_t) * ((n_steps - step) / n_steps)
    return tf.concat(
        [
            tf.math.log(s_t / strike),
            tf.math.log(s_t / barrier),
            time_left,
            active,
        ],
        axis=1,
    )


# ============================================================
# Strategy rollout
# ============================================================
def rollout_european_call(
    hedger: DeepHedger,
    paths: tf.Tensor,
    *,
    strike: float,
    training: bool,
    return_positions: bool = False,
) -> dict[str, tf.Tensor]:
    batch_size = tf.shape(paths)[0]
    wealth = tf.ones((batch_size, 1), dtype=tf.float32) * hedger.premium
    positions = []

    for step, network in enumerate(hedger.networks):
        s_t = paths[:, step : step + 1]
        s_next = paths[:, step + 1 : step + 2]
        phi_t = network(
            european_state(s_t, step=step, n_steps=hedger.n_steps, strike=strike),
            training=training,
        )
        wealth = wealth + phi_t * (s_next - s_t)
        if return_positions:
            positions.append(phi_t)

    payoff = tf.nn.relu(paths[:, -1:] - strike)
    error = wealth - payoff

    results = {"wealth": wealth, "payoff": payoff, "error": error}
    if return_positions:
        results["positions"] = tf.concat(positions, axis=1)
    return results


def rollout_barrier_option(
    hedger: DeepHedger,
    paths: tf.Tensor,
    *,
    strike: float,
    barrier: float,
    training: bool,
    return_positions: bool = False,
) -> dict[str, tf.Tensor]:
    batch_size = tf.shape(paths)[0]
    wealth = tf.ones((batch_size, 1), dtype=tf.float32) * hedger.premium
    active = tf.ones((batch_size, 1), dtype=tf.float32)
    payoff = tf.zeros((batch_size, 1), dtype=tf.float32)
    positions = []

    for step, network in enumerate(hedger.networks):
        s_t = paths[:, step : step + 1]
        s_next = paths[:, step + 1 : step + 2]

        raw_phi = network(
            barrier_state(
                s_t,
                step=step,
                n_steps=hedger.n_steps,
                strike=strike,
                barrier=barrier,
                active=active,
            ),
            training=training,
        )
        phi_t = raw_phi * active
        wealth = wealth + phi_t * (s_next - s_t)

        hit_now = active * tf.cast(s_next >= barrier, tf.float32)
        payoff = payoff + hit_now * (s_next - strike)
        active = active * (1.0 - hit_now)

        if return_positions:
            positions.append(phi_t)

    error = wealth - payoff

    results = {"wealth": wealth, "payoff": payoff, "error": error}
    if return_positions:
        results["positions"] = tf.concat(positions, axis=1)
    return results


# ============================================================
# Training and evaluation helpers
# ============================================================
def train_hedger(
    hedger: DeepHedger,
    train_paths: np.ndarray,
    rollout_fn,
    *,
    config: TrainingConfig,
    rollout_kwargs: dict[str, float],
) -> list[float]:
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
    dataset = (
        tf.data.Dataset.from_tensor_slices(train_paths.astype(np.float32))
        .shuffle(train_paths.shape[0], seed=42, reshuffle_each_iteration=True)
        .batch(config.batch_size)
    )

    history = []
    for epoch in range(config.epochs):
        epoch_losses = []
        for batch_paths in dataset:
            with tf.GradientTape() as tape:
                outputs = rollout_fn(
                    hedger,
                    batch_paths,
                    training=True,
                    return_positions=False,
                    **rollout_kwargs,
                )
                loss = tf.reduce_mean(tf.square(outputs["error"]))

            gradients = tape.gradient(loss, hedger.trainable_variables)
            optimizer.apply_gradients(zip(gradients, hedger.trainable_variables))
            epoch_losses.append(float(loss.numpy()))

        mean_loss = float(np.mean(epoch_losses))
        history.append(mean_loss)

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == config.epochs:
            print(
                f"Epoch {epoch + 1:2d}/{config.epochs} | "
                f"mean squared hedge error = {mean_loss:.6f}",
                flush=True,
            )

    return history


def evaluate_hedger(
    hedger: DeepHedger,
    test_paths: np.ndarray,
    rollout_fn,
    *,
    rollout_kwargs: dict[str, float],
) -> dict[str, np.ndarray | float]:
    outputs = rollout_fn(
        hedger,
        tf.convert_to_tensor(test_paths.astype(np.float32)),
        training=False,
        return_positions=True,
        **rollout_kwargs,
    )

    errors = outputs["error"].numpy().reshape(-1)
    payoffs = outputs["payoff"].numpy().reshape(-1)
    wealth = outputs["wealth"].numpy().reshape(-1)
    positions = outputs["positions"].numpy()

    return {
        "premium": float(hedger.premium.numpy()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean_error": float(np.mean(errors)),
        "mean_payoff": float(np.mean(payoffs)),
        "wealth_mean": float(np.mean(wealth)),
        "errors": errors,
        "payoffs": payoffs,
        "positions": positions,
    }


# ============================================================
# Plot helpers
# ============================================================
def plot_barrier_results(
    *,
    paths: np.ndarray,
    barrier: float,
    loss_history: list[float],
    errors: np.ndarray,
    out_file: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    max_paths = min(30, paths.shape[0])
    for idx in range(max_paths):
        axes[0].plot(paths[idx], alpha=0.6)
    axes[0].axhline(barrier, color="black", ls="--", lw=2, label="Barrier")
    axes[0].set_title("Sample barrier paths")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Stock price")
    axes[0].legend()
    axes[0].grid(True, ls="--", alpha=0.3)

    axes[1].plot(loss_history, color="crimson", lw=2)
    axes[1].set_title("Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE hedge error")
    axes[1].grid(True, ls="--", alpha=0.3)

    axes[2].hist(errors, bins=40, color="darkorange", alpha=0.85)
    axes[2].set_title("Terminal hedge error")
    axes[2].set_xlabel("Wealth - payoff")
    axes[2].set_ylabel("Count")
    axes[2].grid(True, ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_constrained_call_results(
    *,
    market: MarketConfig,
    sigma: float,
    loss_history: list[float],
    hedger: DeepHedger,
    out_file: str,
) -> None:
    price_grid = np.linspace(70.0, 130.0, 250, dtype=np.float32).reshape(-1, 1)
    chosen_steps = [0, market.n_steps // 2, market.n_steps - 1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(loss_history, color="navy", lw=2)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE hedge error")
    axes[0].grid(True, ls="--", alpha=0.3)

    for step in chosen_steps:
        tau = market.t - step * (market.t / market.n_steps)
        state = european_state(
            tf.convert_to_tensor(price_grid),
            step=step,
            n_steps=market.n_steps,
            strike=market.k,
        )
        learned_delta = hedger.networks[step](state, training=False).numpy().reshape(-1)
        exact_delta = bs_call_delta(
            price_grid.reshape(-1),
            strike=market.k,
            sigma=sigma,
            tau=tau,
            r=market.r,
        )

        label = f"step {step + 1}"
        axes[1].plot(price_grid, learned_delta, lw=2, label=f"NN delta ({label})")
        axes[1].plot(
            price_grid,
            exact_delta,
            ls="--",
            lw=2,
            label=f"BS delta ({label})",
        )

    axes[1].set_title("Constrained hedge ratio")
    axes[1].set_xlabel("Stock price")
    axes[1].set_ylabel("Position in stock")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, ls="--", alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Exercises
# ============================================================
def solve_exercise_1(market: MarketConfig, training: TrainingConfig) -> None:
    print("\n" + "=" * 70)
    print("Exercise 1: Barrier exercised option")
    print("=" * 70, flush=True)

    sigma = 0.40
    barrier = 110.0
    strike = market.k

    train_paths = simulate_black_scholes_paths(
        training.train_paths,
        s0=market.s0,
        sigma=sigma,
        t=market.t,
        n_steps=market.n_steps,
        r=market.r,
        seed=1,
    )
    test_paths = simulate_black_scholes_paths(
        training.test_paths,
        s0=market.s0,
        sigma=sigma,
        t=market.t,
        n_steps=market.n_steps,
        r=market.r,
        seed=2,
    )

    train_payoff, _, _ = barrier_exercised_payoff(
        train_paths,
        barrier=barrier,
        strike=strike,
    )
    test_payoff, _, hit_any = barrier_exercised_payoff(
        test_paths,
        barrier=barrier,
        strike=strike,
    )

    hedger = DeepHedger(
        n_steps=market.n_steps,
        input_dim=4,
        hidden_units=training.hidden_units,
        output_mode="scaled_tanh",
        premium_init=float(np.mean(train_payoff)),
        position_scale=2.0,
    )

    loss_history = train_hedger(
        hedger,
        train_paths,
        rollout_barrier_option,
        config=training,
        rollout_kwargs={"strike": strike, "barrier": barrier},
    )

    stats = evaluate_hedger(
        hedger,
        test_paths,
        rollout_barrier_option,
        rollout_kwargs={"strike": strike, "barrier": barrier},
    )

    print(f"Chosen model           : Black-Scholes with sigma = {sigma:.2f}", flush=True)
    print(f"Barrier level B        : {barrier:.2f}", flush=True)
    print(f"Estimated option price : {stats['premium']:.4f}", flush=True)
    print(f"Test MC price          : {np.mean(test_payoff):.4f}", flush=True)
    print(f"Barrier hit rate       : {np.mean(hit_any):.2%}", flush=True)
    print(f"Test hedge RMSE        : {stats['rmse']:.4f}", flush=True)
    print(f"Mean hedge error       : {stats['mean_error']:.4f}", flush=True)

    plot_barrier_results(
        paths=test_paths[:40],
        barrier=barrier,
        loss_history=loss_history,
        errors=stats["errors"],
        out_file="week4_ex1_barrier.png",
    )


def solve_exercise_2(market: MarketConfig, training: TrainingConfig) -> None:
    print("\n" + "=" * 70)
    print("Exercise 2: Constrained strategy")
    print("=" * 70, flush=True)

    sigma = 0.20
    strike = market.k

    train_paths = simulate_black_scholes_paths(
        training.train_paths,
        s0=market.s0,
        sigma=sigma,
        t=market.t,
        n_steps=market.n_steps,
        r=market.r,
        seed=11,
    )
    test_paths = simulate_black_scholes_paths(
        training.test_paths,
        s0=market.s0,
        sigma=sigma,
        t=market.t,
        n_steps=market.n_steps,
        r=market.r,
        seed=12,
    )

    train_payoff = european_call_payoff(train_paths, strike)
    exact_price = float(
        bs_call_price(
            market.s0,
            strike=strike,
            sigma=sigma,
            tau=market.t,
            r=market.r,
        )
    )

    hedger = DeepHedger(
        n_steps=market.n_steps,
        input_dim=3,
        hidden_units=training.hidden_units,
        output_mode="sigmoid",
        premium_init=float(np.mean(train_payoff)),
    )

    loss_history = train_hedger(
        hedger,
        train_paths,
        rollout_european_call,
        config=training,
        rollout_kwargs={"strike": strike},
    )

    stats = evaluate_hedger(
        hedger,
        test_paths,
        rollout_european_call,
        rollout_kwargs={"strike": strike},
    )

    print(f"Volatility sigma       : {sigma:.2f}", flush=True)
    print(f"Learned option price   : {stats['premium']:.4f}", flush=True)
    print(f"Black-Scholes price    : {exact_price:.4f}", flush=True)
    print(f"Test hedge RMSE        : {stats['rmse']:.4f}", flush=True)
    print(f"Mean hedge error       : {stats['mean_error']:.4f}", flush=True)
    print(
        "Observed strategy range: "
        f"[{np.min(stats['positions']):.4f}, {np.max(stats['positions']):.4f}]",
        flush=True,
    )

    plot_constrained_call_results(
        market=market,
        sigma=sigma,
        loss_history=loss_history,
        hedger=hedger,
        out_file="week4_ex2_constrained.png",
    )


def main() -> None:
    print("Week 4 deep hedging solutions", flush=True)
    print("Assumption: r = 0, so we hedge in discounted units.", flush=True)
    print("Tip: increase TrainingConfig sizes if you want a tighter hedge.", flush=True)

    market = MarketConfig()
    training = TrainingConfig()

    solve_exercise_1(market, training)
    solve_exercise_2(market, training)

    print("\nSaved figures:", flush=True)
    print("  - week4_ex1_barrier.png", flush=True)
    print("  - week4_ex2_constrained.png", flush=True)


if __name__ == "__main__":
    main()

