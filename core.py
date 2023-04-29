import optax
import jax.numpy as jnp
import jax
import flax.linen as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler


##################
# Model
##################
class NN(nn.Module):
    n_hidden_layer_neurons: list
    output_shape: int
    activation: callable
    kernel_first_layer_init: callable
    kernel_other_layers_init: callable
    bias_first_layer_init: callable
    bias_other_layers_init: callable
    scaling_factor: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.n_hidden_layer_neurons[0],
            kernel_init=self.kernel_first_layer_init,
            bias_init=self.bias_first_layer_init,
        )(x)
        x = self.activation(x)
        for i in range(1, len(self.n_hidden_layer_neurons)):
            x = nn.Dense(
                self.n_hidden_layer_neurons[i],
                kernel_init=self.kernel_other_layers_init,
                bias_init=self.bias_other_layers_init,
            )(x)
            x = self.activation(x)
        x = nn.Dense(self.output_shape)(x)
        return x


##################
# Weights initialization
##################
def siren_first_layer_init(key, shape, dtype):
    input_shape = shape[0]
    return jax.random.uniform(
        key, shape, dtype, minval=-1.0 / input_shape, maxval=1.0 / input_shape
    )


def siren_other_layers_init(key, shape, dtype, scaling_factor):
    input_shape = shape[0]
    return jax.random.uniform(
        key,
        shape,
        dtype,
        minval=-jnp.sqrt(6.0 / input_shape) / scaling_factor,
        maxval=jnp.sqrt(6.0 / input_shape) / scaling_factor,
    )


he_normal_init = nn.initializers.he_normal()
he_uniform_init = nn.initializers.he_uniform()
lecun_uniform_init = nn.initializers.lecun_uniform()
lecun_normal_init = nn.initializers.lecun_normal()
zeros_init = nn.initializers.zeros_init()


##################
# Utils
##################


def fit(key, model, train_x, train_y, lr, batch_size, iterations):
    train_x = jnp.asarray(train_x)
    train_y = jnp.asarray(train_y)

    # initialize params
    params = model.init(key, jnp.ones((1, train_x.shape[1])))

    # loss fun
    def loss_fn(params, x, y, key):
        y_hat = model.apply(params, x, rngs={"dropout": key})
        loss = jnp.mean((y - y_hat) ** 2)
        return loss, y_hat

    value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    optimizer = optax.adam(lr)
    state = optimizer.init(params)

    # lax scan loop
    @jax.jit
    def one_step(params_and_state, key):
        params, state = params_and_state
        if batch_size < train_y.shape[0]:
            batch_idx = jax.random.choice(
                key, jnp.arange(train_x.shape[0]), shape=(batch_size,), replace=False
            )
            x = train_x[batch_idx]
            y = train_y[batch_idx]
        else:
            x = train_x
            y = train_y
        (value, grads), out = value_and_grad_fn(params, x, y, key)
        updates, state = optimizer.update(grads, state)
        params = optax.apply_updates(params, updates)
        return (params, state), (value, out)

    keys = jax.random.split(key, iterations)
    (params, state), (losses, outs) = jax.lax.scan(one_step, (params, state), xs=keys)
    return params, losses, outs


def predict(model, params, x):
    return model.apply(params, x)
