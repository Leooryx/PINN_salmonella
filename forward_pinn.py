import pandas as pd
import matplotlib.pyplot as plt
import jinns
import jax
from jax import random, vmap
import jax.numpy as jnp
import equinox as eqx
import optax
from jinns.loss import ODE


## PINN architecture
key = random.PRNGKey(2)
key, subkey = random.split(key)
eqx_list = ( #neural network defined using equinox and JAX
    (eqx.nn.Linear, 1, 20),
    (jax.nn.tanh,),
    (eqx.nn.Linear, 20, 40),
    (jax.nn.tanh,),
    (eqx.nn.Linear, 40, 20),
    (jax.nn.tanh,),
    (eqx.nn.Linear, 20, 1),
    # (jnp.exp,)
)
key, subkey = random.split(key)

u, init_nn_params = jinns.nn.PINN_MLP.create(key=subkey, eqx_list=eqx_list, eq_type="ODE")

## Define time collocation points
nt = 320
batch_size = 32
method = 'uniform'
tmin = 0
tmax = 1    # temps max normalisé
Tmax = 27   # durée totale de la simulation en jours

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
# u0 = [100.0, 100.0, 80.0, 0.8, 0.2] # valeur initiale de u(t0)
u0 = [100.0]  # état initial identique à state0 en R
u0 = jnp.array(u0)

# Parameters
init_params = jinns.parameters.Params(
    nn_params=init_nn_params,
    eq_params={
        'mu1': 2.0, 'K_S': 300.0, 'C_SA': 0.5,  'C_SB': 0.9,
        'mu2': 1.5, 'K_A': 400.0, 'C_AS': 0.6,  'C_AB': 0.2,
        'mu3': 1.5, 'K_B': 300.0, 'C_BS': 1.1,  'C_BA': 0.2,
        'beta_S': 0.7, 'beta_A': 1.2, 'beta_B': 1.3,
        'gamma1': 0.3, 'gamma2': 0.1, 'gamma3': 0.1,
        'alpha_S': 6.0, 'alpha_A': 3.0, 'K_I': 0.1, 'n_I': 20.0,
        'delta_i': 0.1, 'C_iB': 0.001, 'C_im': 0.8,
        'rho': 0.025
    },
)


vectorized_u_init = vmap(lambda t: u(t, init_params), (0), 0)

ts = train_data.times.sort(axis=0)
# plt.plot(ts * Tmax, vectorized_u_init(ts), label="Init solution")
# plt.legend()



class LinearFODE(ODE):
    def equation(self, t, u, params):
        # Décomposition des variables d'état à partir de u(t, params)
        #u_ = lambda t, p: u(t, p)[0:5]  # On suppose que u retourne un tableau de 5 éléments
        S_val = lambda t, p: u(t, p)[0]
        # A_val = lambda t, p: u(t, p)[1]
        # B_val = lambda t, p: u(t, p)[2]
        # i_val = lambda t, p: u(t, p)[3]
        # m_val = lambda t, p: u(t, p)[4]

########################################################################################### OBSOLETE
        #S_val, dS_dt = jax.value_and_grad(get_S, 0)(t, params)
        #A_val, dA_dt = jax.value_and_grad(get_A, 0)(t, params)
        #B_val, dB_dt = jax.value_and_grad(get_B, 0)(t, params)
        #i_val, di_dt = jax.value_and_grad(get_i, 0)(t, params)
        #m_val, dm_dt = jax.value_and_grad(get_m, 0)(t, params)
############################################################################################

        dS_dt = jax.grad(S_val, 0)(t, params)
        # dA_dt = jax.grad(A_val, 0)(t, params)
        # dB_dt = jax.grad(B_val, 0)(t, params)
        # di_dt = jax.grad(i_val, 0)(t, params)
        # dm_dt = jax.grad(m_val, 0)(t, params)

        # # 4. Paramètres du système
        # mu1, K_S, C_SA, C_SB = params.eq_params['mu1'], params.eq_params['K_S'], params.eq_params['C_SA'], params.eq_params['C_SB']
        # mu2, K_A, C_AS, C_AB = params.eq_params['mu2'], params.eq_params['K_A'], params.eq_params['C_AS'], params.eq_params['C_AB']
        # mu3, K_B, C_BS, C_BA = params.eq_params['mu3'], params.eq_params['K_B'], params.eq_params['C_BS'], params.eq_params['C_BA']
        # beta_S, beta_A, beta_B = params.eq_params['beta_S'], params.eq_params['beta_A'], params.eq_params['beta_B']
        # gamma1, gamma2, gamma3 = params.eq_params['gamma1'], params.eq_params['gamma2'], params.eq_params['gamma3']
        # alpha_S, alpha_A = params.eq_params['alpha_S'], params.eq_params['alpha_A']
        # K_I, n_I = params.eq_params['K_I'], params.eq_params['n_I']
        # delta_i, C_iB, C_im = params.eq_params['delta_i'], params.eq_params['C_iB'], params.eq_params['C_im']
        # rho = params.eq_params['rho']

        # # 5. Calcul des dérivées du système
        # # Modulation par i et m
        # fS = i_val * (beta_S - m_val)
        # fA = i_val * (beta_A - m_val)
        # fB = i_val * (beta_B - m_val)

        # # Dynamique écologique (valeurs que doivent avoir les dérivées produites par jax et qui servent à calculer la loss)
        # dS = S_val * Tmax * (mu1 * (1 - (S_val + C_SA * A_val + C_SB * B_val) / K_S) + gamma1 * fS)
        # dA = A_val *  Tmax * (mu2 * (1 - (A_val + C_AS * S_val + C_AB * B_val) / K_A) - gamma2 * fA)
        # dB = B_val *  Tmax * (mu3 * (1 - (B_val + C_BS * S_val + C_BA * A_val) / K_B) - gamma3 * fB)

        # # Inflammation
        # H_S = S_val**n_I / (K_I**n_I + S_val**n_I)
        # H_A = A_val**n_I / (K_I**n_I + A_val**n_I)
        # di =  Tmax * alpha_S * H_S +  Tmax * alpha_A * H_A -  Tmax * (delta_i + C_iB * B_val + C_im * m_val) * i_val

        # # Maturation
        # dm =  Tmax * rho * i_val * (1 - m_val)

        # 6. Forme finale : du/dt - f(u) = 0
        # return jnp.array([
        #     dS_dt - dS,
        #     dA_dt - dA,
        #     dB_dt - dB,
        #     di_dt - di,
        #     dm_dt - dm
        # ])
        # print(self.Tmax)
        return jnp.array([
            dS_dt - params.eq_params.mu1 # * self.Tmax
            #dA_dt - params.eq_params['mu1'],
            #dB_dt - params.eq_params['mu1'],
            #di_dt - params.eq_params['mu1'],
            #dm_dt - params.eq_params['mu1']
        ])

# def data_loss(u, params, t_data, Y_obs):
#     """
#     u : réseau PINN
#     params : paramètres du PINN
#     t_data : temps des observations (shape N,1)
#     Y_obs : observations (S_obs, A_obs, B_obs) shape (N,3)
#     """
#     # Prédictions du PINN aux temps observés
#     Y_pred = vmap(lambda t: u(t, params))(t_data)  # shape (N, 5)

#     # On ne garde que les 3 premières colonnes (S, A, B)
#     Y_pred_SAB = Y_pred[:, :3]

#     # MSE sur les données observées
#     return jnp.mean((Y_pred_SAB - Y_obs) ** 2)


# def system(t, y, p):
#     S, A, B, i, m = y
#     # Parameters unpacking
#     mu1, K_S, C_SA, C_SB = p['mu1'], p['K_S'], p['C_SA'], p['C_SB']
#     mu2, K_A, C_AS, C_AB = p['mu2'], p['K_A'], p['C_AS'], p['C_AB']
#     mu3, K_B, C_BS, C_BA = p['mu3'], p['K_B'], p['C_BS'], p['C_BA']
#     beta_S, beta_A, beta_B = p['beta_S'], p['beta_A'], p['beta_B']
#     gamma1, gamma2, gamma3 = p['gamma1'], p['gamma2'], p['gamma3']
#     alpha_S, alpha_A = p['alpha_S'], p['alpha_A']
#     K_I, n_I = p['K_I'], p['n_I']
#     delta_i, C_iB, C_im = p['delta_i'], p['C_iB'], p['C_im']
#     rho = p['rho']

#     # Linear modulation attenuated by m
#     fS = i * (beta_S - m)
#     fA = i * (beta_A - m)
#     fB = i * (beta_B - m)

#     # Ecological  dynamics 
#     dS = S * (mu1 * (1 - (S + C_SA * A + C_SB * B) / K_S) + gamma1 * fS)
#     dA = A * (mu2 * (1 - (A + C_AS * S + C_AB * B) / K_A) - gamma2 * fA)
#     dB = B * (mu3 * (1 - (B + C_BS * S + C_BA * A) / K_B) - gamma3 * fB)

#     # Inflammation
#     H_S = S**n_I / (K_I**n_I + S**n_I)
#     H_A = A**n_I / (K_I**n_I + A**n_I)
#     di = alpha_S * H_S + alpha_A * H_A - (delta_i + C_iB * B + C_im * m) * i

#     # Maturation
#     dm = rho * i * (1 - m)

#     return [dS, dA, dB, di, dm]



fo_loss = LinearFODE(Tmax=Tmax) # loss dynamique


# Calcul Loss Totale
loss_weights = jinns.loss.LossWeightsODE(dyn_loss=1000.0, initial_condition=1.0)

loss = jinns.loss.LossODE(
    u=u,
    loss_weights=loss_weights,
    dynamic_loss=fo_loss,
    initial_condition=(float(tmin), u0),
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
tx = optax.adam(learning_rate=1e-4)
n_iter = int(100000)
key, subkey = random.split(key)
params, total_loss_list, loss_by_term_dict, data, loss, _, _ , _, _, _ = jinns.solve(
    init_params=params,
    data=train_data,
    optimizer=tx,
    loss=loss,
    n_iter=n_iter
)
# for loss_name, loss_values in loss_by_term_dict.items():
#     plt.plot(jnp.log10(loss_values), label=loss_name)
# plt.plot(jnp.log10(total_loss_list), label="total loss")
# plt.legend()


# #Plot Prediction PINN pré-entrainement
# plt.figure(figsize=(8,5))
# plt.plot(ts * Tmax, vectorized_u_init(ts))
# plt.title("Prediction par PINN (pré-entrainement) des solutions en fonction du temps")
# plt.xlabel("Temps")
# plt.ylabel("Valeur des variables d'état")
# plt.legend(["S", "A", "B", "i", "m"])
# plt.grid(True)

key, subkey = random.split(key, 2)
val_data = jinns.data.DataGeneratorODE(key=subkey, nt=nt, tmin=tmin, tmax=tmax, temporal_batch_size=batch_size, method=method)
ts = val_data.times.sort(axis=0).squeeze() #vecteur 1D contenant les 320 pas de temps normalisés sur [0,1]

u_est_fp = vmap(lambda t:u(t, params), (0), 0)
u_pinn_all = u_est_fp(ts)  # shape (320, 5) sans squeeze ici pour indexer proprement

import pandas as pd
df = pd.DataFrame(
    {
        "t": ts * Tmax, # rescale time for plotting
        "S_pinn": u_pinn_all[:, 0],  # composante 1 : S
        "A_pinn": u_pinn_all[:, 1],  # composante 2 : A
        "B_pinn": u_pinn_all[:, 2],  # composante 3 : B
        "i_pinn": u_pinn_all[:, 3],  # composante 4 : i
        "m_pinn": u_pinn_all[:, 4],  # composante 5 : m
        #"u_pinn": u_est_fp(ts).squeeze(),  #exp à retirer ? si la PINN prédit en log-space
        #"u_true": vmap(u_true)(ts),
        "Method": "PINN"
    },
)

plt.figure(figsize=(8,5))
plt.plot(ts * Tmax, u_pinn_all[:, 0], label="S")
plt.plot(ts * Tmax, u_pinn_all[:, 1], label="A")
plt.plot(ts * Tmax, u_pinn_all[:, 2], label="B")
plt.plot(ts * Tmax, u_pinn_all[:, 3], label="i")
plt.plot(ts * Tmax, u_pinn_all[:, 4], label="m")
plt.title("Prediction par PINN (post-entraînement) des solutions en fonction du temps")
plt.xlabel("Temps")
plt.ylabel("Valeur des variables d'état")
plt.legend()
plt.grid(True)


# Plot Courbes de loss pendant l'entraînement
plt.figure(figsize=(8,5))
for loss_name, loss_values in loss_by_term_dict.items():
    plt.plot(jnp.log10(loss_values), label=loss_name)
plt.plot(jnp.log10(total_loss_list), label="Loss totale", linewidth=2, color="black")
plt.title("Évolution des pertes d'entraînement (échelle log₁₀)")
plt.xlabel("Itération")
plt.ylabel("log₁₀(Loss)")
plt.legend()
plt.grid(True)
plt.show()