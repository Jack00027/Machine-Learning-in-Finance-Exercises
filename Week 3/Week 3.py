# Exercise 2

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

# ============================================================
# Reproducibility
# ============================================================
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================
# True function and derivative (only for checking / plotting)
# ============================================================
def f_true(u):
    return u**2 - 2*u + 2

def df_true(u):
    return 2*u - 2

# ============================================================
# Generate Monte Carlo training data
# For each u_i, simulate M samples from Exp(1)
# ============================================================
def generate_data(N=400, M=50):
    # Sample input points u uniformly on [0,2]
    u = np.random.uniform(0.0, 2.0, size=(N, 1)).astype("float32")

    # Simulate X ~ Exp(1), shape = (N, M)
    X = np.random.exponential(scale=1.0, size=(N, M)).astype("float32")

    # Monte Carlo estimate of f(u) = E[(X-u)^2]
    y = np.mean((X - u)**2, axis=1, keepdims=True).astype("float32")

    # Monte Carlo estimate of f'(u) = E[2(u-X)]
    z = np.mean(2.0 * (u - X), axis=1, keepdims=True).astype("float32")

    return u, y, z

# Training and test data
u_train, y_train, z_train = generate_data(N=400, M=50)
u_test,  y_test,  z_test  = generate_data(N=200, M=200)

# Convert to tensors
u_train_tf = tf.convert_to_tensor(u_train)
y_train_tf = tf.convert_to_tensor(y_train)
z_train_tf = tf.convert_to_tensor(z_train)

u_test_tf = tf.convert_to_tensor(u_test)
y_test_tf = tf.convert_to_tensor(y_test)
z_test_tf = tf.convert_to_tensor(z_test)

# ============================================================
# Build model
# I use tanh instead of ReLU because here we also want a smooth derivative
# ============================================================
def build_model():
    inp = Input(shape=(1,))
    x = Dense(32, activation="tanh")(inp)
    x = Dense(32, activation="tanh")(x)
    out = Dense(1, activation="linear")(x)
    return Model(inputs=inp, outputs=out)

model = build_model()
optimizer = Adam(learning_rate=1e-3)

# Weight of derivative loss
lambda_deriv = 1.0

# For plotting loss curves
value_losses = []
deriv_losses = []
total_losses = []

# ============================================================
# Custom training loop
# We need this because the loss depends on g(u) and on g'(u)
# ============================================================
epochs = 1500

for epoch in range(epochs):
    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(u_train_tf)
            g_pred = model(u_train_tf, training=True)   # g(u)

        dg_pred = inner_tape.gradient(g_pred, u_train_tf)  # g'(u)

        value_loss = tf.reduce_mean((g_pred - y_train_tf)**2)
        deriv_loss = tf.reduce_mean((dg_pred - z_train_tf)**2)
        total_loss = value_loss + lambda_deriv * deriv_loss

    grads = outer_tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    value_losses.append(value_loss.numpy())
    deriv_losses.append(deriv_loss.numpy())
    total_losses.append(total_loss.numpy())

    if epoch % 300 == 0:
        print(f"Epoch {epoch:4d} | "
              f"value loss = {value_loss.numpy():.6f} | "
              f"derivative loss = {deriv_loss.numpy():.6f} | "
              f"total loss = {total_loss.numpy():.6f}")

# ============================================================
# Evaluate on a grid
# ============================================================
u_grid = np.linspace(0.0, 2.0, 400).reshape(-1, 1).astype("float32")
u_grid_tf = tf.convert_to_tensor(u_grid)

with tf.GradientTape() as tape:
    tape.watch(u_grid_tf)
    g_grid = model(u_grid_tf, training=False)
dg_grid = tape.gradient(g_grid, u_grid_tf)

g_grid = g_grid.numpy().flatten()
dg_grid = dg_grid.numpy().flatten()

# Exact curves
f_grid = f_true(u_grid.flatten())
df_grid = df_true(u_grid.flatten())

# ============================================================
# Test errors
# ============================================================
with tf.GradientTape() as tape:
    tape.watch(u_test_tf)
    g_test = model(u_test_tf, training=False)
dg_test = tape.gradient(g_test, u_test_tf)

test_value_mse = tf.reduce_mean((g_test - y_test_tf)**2).numpy()
test_deriv_mse = tf.reduce_mean((dg_test - z_test_tf)**2).numpy()

print("\nTest MSE for function values     :", test_value_mse)
print("Test MSE for derivative values   :", test_deriv_mse)

# ============================================================
# Plots
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

# Losses
axes[0].semilogy(value_losses, label="value loss")
axes[0].semilogy(deriv_losses, label="derivative loss")
axes[0].semilogy(total_losses, label="total loss")
axes[0].set_title("Training losses")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, ls="--", alpha=0.4)

# Function fit
axes[1].scatter(u_train[:80], y_train[:80], s=15, alpha=0.4, label="MC train labels")
axes[1].plot(u_grid, f_grid, color="black", lw=2, label="true f(u)")
axes[1].plot(u_grid, g_grid, "--", color="crimson", lw=2, label="NN g(u)")
axes[1].set_title("Function approximation")
axes[1].set_xlabel("u")
axes[1].set_ylabel("value")
axes[1].legend()
axes[1].grid(True, ls="--", alpha=0.4)

# Derivative fit
axes[2].scatter(u_train[:80], z_train[:80], s=15, alpha=0.4, label="MC derivative labels")
axes[2].plot(u_grid, df_grid, color="black", lw=2, label="true f'(u)")
axes[2].plot(u_grid, dg_grid, "--", color="darkorange", lw=2, label="NN g'(u)")
axes[2].set_title("Derivative approximation")
axes[2].set_xlabel("u")
axes[2].set_ylabel("derivative")
axes[2].legend()
axes[2].grid(True, ls="--", alpha=0.4)

plt.tight_layout()
plt.show()
