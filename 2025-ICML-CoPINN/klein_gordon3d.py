import argparse
import os
import time

import jax
import numpy as np
import optax
from networks.hessian_vector_products import *
from tqdm import trange
from utils.data_generators import generate_test_data, generate_train_data
from utils.eval_functions import setup_eval_function
from utils.training_utils import *
from utils.visualizer import show_solution

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

@partial(jax.jit, static_argnums=(2,))
def apply_model_copinn(epoch, num_epochs, apply_fn, params, *train_data):
    def residual_loss(epoch, num_epochs, params, t, x, y, source_term):
        def compute_loss(params, t, x, y, source_term, mean=True):
            # calculate u
            u = apply_fn(params, t, x, y)
            # tangent vector dx/dx
            v_t = jnp.ones(t.shape)
            v_x = jnp.ones(x.shape)
            v_y = jnp.ones(y.shape)
            # 2nd derivatives of u
            utt = hvp_fwdfwd(lambda t: apply_fn(params, t, x, y), (t,), (v_t,))
            uxx = hvp_fwdfwd(lambda x: apply_fn(params, t, x, y), (x,), (v_x,))
            uyy = hvp_fwdfwd(lambda y: apply_fn(params, t, x, y), (y,), (v_y,))
            loss = (utt - uxx - uyy + u**2 - source_term)**2

            if mean:
                return jnp.mean(loss)
            else:
                return loss
            
        # # self-paced learning 
        loss = compute_loss(params, t, x, y, source_term, mean=False)
        grad_fn = jax.grad(compute_loss, argnums=(1, 2, 3))
        grad_t, grad_x, grad_y = grad_fn(params, t, x, y, source_term)

        grad_t_broadcasted = grad_t[:, None, None]  # Extend to (Nc, 1, 1)
        grad_x_broadcasted = grad_x[None, :, None]  # Extend to (1, Nc, 1)
        grad_y_broadcasted = grad_y[None, None, :]   # Extend to (1, 1, Nc)

        flattened_loss = loss.flatten()

        V = get_SPL_V(grad_t_broadcasted, grad_x_broadcasted, grad_y_broadcasted, epoch, num_epochs)

        new_loss = flattened_loss * V

        return jnp.mean(new_loss) 
           
    def get_SPL_V(grad_x, grad_y, grad_z, epoch, num_epochs):
        def get_gamma_d(epoch, num_epochs, beta):
            ve = 1 - (epoch - 1)/num_epochs
            vh = (epoch - 1)/num_epochs
            gamma_d = (ve - vh)*beta
            return gamma_d
        
        difficult = jnp.sqrt(grad_x**2 + grad_y**2 + grad_z**2).squeeze()
        
        flattened_difficult = difficult.flatten()

        difficult_min = flattened_difficult.min()
        difficult_max = flattened_difficult.max()
        normalized_difficult = (flattened_difficult - difficult_min) / (difficult_max - difficult_min)

        beta = 0.01
        gamma_d = get_gamma_d(epoch, num_epochs, beta)

        V = 1 - (1 / num_epochs) * (epoch - 1) - gamma_d * normalized_difficult
        
        return V
    
    def initial_loss(params, t, x, y, u):
        return jnp.mean((apply_fn(params, t, x, y) - u)**2)

    def boundary_loss(params, t, x, y, u):
        loss = 0.
        for i in range(4):
            loss += jnp.mean((apply_fn(params, t[i], x[i], y[i]) - u[i])**2)
        return loss

    # unpack data
    tc, xc, yc, uc, ti, xi, yi, ui, tb, xb, yb, ub = train_data

    # isolate loss func from redundant arguments
    loss_fn = lambda params: residual_loss(epoch, num_epochs, params, tc, xc, yc, uc) + \
                        initial_loss(params, ti, xi, yi, ui) + \
                        boundary_loss(params, tb, xb, yb, ub)

    loss, gradient = jax.value_and_grad(loss_fn)(params)

    return loss, gradient

if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='Training configurations')

    # model and equation
    parser.add_argument('--model', type=str, default='copinn', help='model name')
    parser.add_argument('--equation', type=str, default='klein_gordon3d', help='equation to solve')
    
    # input data settings
    parser.add_argument('--nc', type=int, default=32, help='the number of training points for each axis: 16 32 64 128 256')
    parser.add_argument('--nc_test', type=int, default=100, help='the number of test points for each axis')

    # training settings
    parser.add_argument('--seed', type=int, default=113, help='random seed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=50000, help='training epochs')

    # model settings
    parser.add_argument('--mlp', type=str, default='modified_mlp', choices=['mlp', 'modified_mlp'], help='type of mlp')
    parser.add_argument('--n_layers', type=int, default=4, help='the number of layer')
    parser.add_argument('--features', type=int, default=64, help='feature size of each layer')
    parser.add_argument('--r', type=int, default=32, help='rank of the approximated tensor')
    parser.add_argument('--out_dim', type=int, default=1, help='size of model output')
    parser.add_argument('--pos_enc', type=int, default=0, help='size of the positional encoding (zero if no encoding)')

    # PDE settings
    parser.add_argument('--k', type=int, default=2, help='temporal frequency of the solution')
    # log settings
    parser.add_argument('--log_iter', type=int, default=100, help='print log every...')
    parser.add_argument('--plot_iter', type=int, default=100, help='plot result every...')

    args = parser.parse_args()

    # random key
    key = jax.random.PRNGKey(args.seed)

    # make & init model forward function
    key, subkey = jax.random.split(key, 2)
    apply_fn, params = setup_networks(args, subkey)

    # count total params
    args.total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    # name model
    name = name_model(args)

    # result dir
    root_dir = os.path.join(os.getcwd(), 'results', args.equation, args.model)
    result_dir = os.path.join(root_dir, name)

    # make dir
    os.makedirs(result_dir, exist_ok=True)

    # optimizer
    optim = optax.adam(learning_rate=args.lr)
    state = optim.init(params)

    # dataset
    key, subkey = jax.random.split(key, 2)
    train_data = generate_train_data(args, subkey, result_dir=result_dir)
    test_data = generate_test_data(args, result_dir)

    # evaluation function
    eval_fn = setup_eval_function(args.model, args.equation)

    # save training configuration
    save_config(args, result_dir)

    # log
    logs = []
    if os.path.exists(os.path.join(result_dir, 'log (loss, error).csv')):
        os.remove(os.path.join(result_dir, 'log (loss, error).csv'))
    if os.path.exists(os.path.join(result_dir, 'best_error.csv')):
        os.remove(os.path.join(result_dir, 'best_error.csv'))
    best = 100000.
    best_error = 100000.

    # start training
    for e in trange(1, args.epochs + 1):
        if e == 2:
            # exclude compiling time
            start = time.time()

        if e % 100 == 0:
            # sample new input data
            key, subkey = jax.random.split(key, 2)
            train_data = generate_train_data(args, subkey)

        loss, gradient = apply_model_copinn(e, args.epochs, apply_fn, params, *train_data)

        params, state = update_model(optim, gradient, params, state)

        error, rmse = eval_fn(apply_fn, params, *test_data)
        if error < best:
            best = error
            best_error, best_rmse = error, rmse

        if e % args.log_iter == 0:
            error, rmse = eval_fn(apply_fn, params, *test_data)
            print(f'Epoch: {e}/{args.epochs} --> total loss: {loss:.8f}, error: {error:.8f}, best error {best_error:.8f}, rmse: {rmse:.8f}')
            with open(os.path.join(result_dir, 'log (loss, error).csv'), 'a') as f:
                # f.write(f'{loss}, {error}, {rmse}\n')
                f.write(f'{loss}, {error}, {best_error}, {rmse}, {best_rmse}\n')


        # visualization
        if e % args.plot_iter == 0:
            show_solution(args, apply_fn, params, test_data, result_dir, e, resol=50)


    # training done
    runtime = time.time() - start
    print(f'Runtime --> total: {runtime:.2f}sec ({(runtime/(args.epochs-1)*1000):.2f}ms/iter.)')
    jnp.save(os.path.join(result_dir, 'params.npy'), params)
        
    # save runtime
    runtime = np.array([runtime])
    np.savetxt(os.path.join(result_dir, 'total runtime (sec).csv'), runtime, delimiter=',')

    # save total error
    with open(os.path.join(result_dir, 'best_error.csv'), 'a') as f:
        f.write(f'best error: {best_error}\n')