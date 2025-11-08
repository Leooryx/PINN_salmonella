import jinns
import jax
from jax import random, vmap
import jax.numpy as jnp
import equinox as eqx

import jinns.loss
import matplotlib.pyplot as plt

key = random.PRNGKey(2)
key, subkey = random.split(key)
eqx_list = (
    (eqx.nn.Linear, 1, 20),
    (jax.nn.tanh,),
    (eqx.nn.Linear, 20, 20),
    (jax.nn.tanh,),
    (eqx.nn.Linear, 20, 20),
    (jax.nn.tanh,),
    (eqx.nn.Linear, 20, 1),
    # (jnp.exp,)
)
key, subkey = random.split(key)

u, init_nn_params = jinns.nn.PINN_MLP.create(key=subkey, eqx_list=eqx_list, eq_type="ODE")

nt = 320
batch_size = 32
method = 'uniform'
tmin = 0
tmax = 1

Tmax = 1
key, subkey = random.split(key)
train_data = jinns.data.DataGeneratorODE(
    key=subkey,
    nt=nt,
    tmin=tmin,
    tmax=tmax,
    temporal_batch_size=batch_size,
    method=method
)

# initial conditions and growth
t0 = 0
u0 = 1.
a = 1.

init_params = jinns.parameters.Params(
    nn_params=init_nn_params,
    eq_params={"a": a},
)

vectorized_u_init = vmap(lambda t: u(t, init_params), (0), 0)

def u_true(t):
    return u0 * jnp.exp(a * (t-t0) )

ts = train_data.times.sort(axis=0)
plt.plot(ts * Tmax, vmap(u_true, 0)(ts * Tmax), label="True solution")
plt.plot(ts * Tmax, vectorized_u_init(ts), label="Init solution")
plt.legend()

from jinns.loss import ODE


class LinearFODE(ODE):

    def equation(self, t, u, params):
        # in log-space
        u_ = lambda t, p: u(t, p)[0]
        du_dt = jax.grad(u_, 0)(t, params)
        return du_dt - params.eq_params.a


fo_loss = LinearFODE(Tmax=Tmax)

loss_weights = jinns.loss.LossWeightsODE(dyn_loss=2.0, initial_condition=1.0)

loss = jinns.loss.LossODE(
    u=u,
    loss_weights=loss_weights,
    dynamic_loss=fo_loss,
    initial_condition=(float(tmin), jnp.log(u0)),
    params=init_params
)
# Testing the loss function
train_data, batch = train_data.get_batch()

losses_and_grad = jax.value_and_grad(loss.evaluate, 0, has_aux=True)
losses, grads = losses_and_grad(
    init_params,
    batch
)
l_tot, d = losses
print(f"total loss: {l_tot}")
print(f"Individual losses: { {key: f'{val:.2f}' for key, val in d.items()} }")

params = init_params
# Optimizer
import optax
tx = optax.adam(learning_rate=1e-4)
n_iter = int(10000)
key, subkey = random.split(key)
params, total_loss_list, loss_by_term_dict, data, loss, _, _ , _, _, _ = jinns.solve(
    init_params=params,
    data=train_data,
    optimizer=tx,
    loss=loss,
    n_iter=n_iter
)
for loss_name, loss_values in loss_by_term_dict.items():
    plt.plot(jnp.log10(loss_values), label=loss_name)
plt.plot(jnp.log10(total_loss_list), label="total loss")
plt.legend()

u_est_fp = vmap(lambda t:u(t, params), (0), 0)


key, subkey = random.split(key, 2)
val_data = jinns.data.DataGeneratorODE(key=subkey, nt=nt, tmin=tmin, tmax=tmax, temporal_batch_size=batch_size, method=method)

import pandas as pd
ts = val_data.times.sort(axis=0).squeeze()
df = pd.DataFrame(
    {
        "t": ts * Tmax, # rescale time for plotting
        "u_true": vmap(u_true)(ts),
        "u_pinn": jnp.exp(u_est_fp(ts).squeeze()),
        "Method": "PINN"
    },
)
df.plot(x="t", style=["-", "--"])

plt.show()
