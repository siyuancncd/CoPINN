import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from utils.vorticity import velocity_to_vorticity_fwd
from utils.data_utils import helmholtz3d_exact_u, klein_gordon3d_exact_u
from utils.vorticity import vorx, vory, vorz
import matplotlib
from matplotlib.ticker import FuncFormatter

import pdb

def format_ticks(x, pos):
    if x == 10:
        return '10.00'
    
    if x == 0:
        return '0.000'
    
    if x >= 1:
        order = jnp.floor(jnp.log10(abs(x)))
        decimal_places = int(max(0, 3 - order))  # 确保总位数为4
    elif x >= 0.1: 
        order = jnp.floor(jnp.log10(abs(x)))
        decimal_places = int(max(0, 2 - order))  # 确保总位数为4
    elif x >= 0.01: 
        order = jnp.floor(jnp.log10(abs(x)))
        decimal_places = int(max(0, 1 - order))  # 确保总位数为4
    else:
        order = jnp.floor(jnp.log10(abs(x)))
        decimal_places = int(max(0, - order))
        
    format_str = f"{{:.{decimal_places}f}}"
    return format_str.format(x)

def _diffusion3d(args, apply_fn, params, test_data, result_dir, e, resol):
    print("visualizing solution...")

    nt = 11 # number of time steps to visualize
    t = jnp.linspace(0., 1., nt)
    x = jnp.linspace(-1., 1., resol)
    y = jnp.linspace(-1., 1., resol)
    xd, yd = jnp.meshgrid(x, y, indexing='ij')  # for 3-d surface plot
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')
    if args.model == 'pinn':
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
    else:
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    u_ref = test_data[-1]
    ref_idx = 0

    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)
    u = apply_fn(params, t, x, y)
    if args.model == 'pinn':
        u = u.reshape(nt, resol, resol)
        u_ref = u_ref.reshape(-1, resol, resol)

    for tt in range(nt):
        fig = plt.figure(figsize=(18, 6))

        # reference solution (hard-coded; must be modified if nt changes)
        ax1 = fig.add_subplot(131, projection='3d')
        im = ax1.plot_surface(xd, yd, u_ref[ref_idx], cmap='jet', linewidth=0, antialiased=False)
        ref_idx += 10
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u')
        ax1.set_title(f'Reference $u(x, y)$ at $t={tt*(1/(nt-1)):.1f}$', fontsize=15)
        ax1.set_zlim(jnp.min(u_ref), jnp.max(u_ref))
        ax1.tick_params(axis='both', which='major', labelsize=6)

        # predicted solution
        ax2 = fig.add_subplot(132, projection='3d')
        im = ax2.plot_surface(xd, yd, u[tt], cmap='jet', linewidth=0, antialiased=False)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('u')
        ax2.set_title(f'Predicted $u(x, y)$ at $t={tt*(1/(nt-1)):.1f}$', fontsize=15)
        ax2.set_zlim(jnp.min(u_ref), jnp.max(u_ref))
        ax2.tick_params(axis='both', which='major', labelsize=6)

        # absolute error
        abs_error = jnp.abs(u[tt]-u_ref[ref_idx])
        ax3 = fig.add_subplot(133, projection='3d')
        im = ax3.plot_surface(xd, yd, abs_error, cmap='jet', linewidth=0, antialiased=False)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_zlabel('u')
        ax3.set_title(f'Absolute error $u(x, y)$ at $t={tt*(1/(nt-1)):.1f}$', fontsize=15)
        ax3.set_zlim(jnp.min(u_ref), jnp.max(u_ref))
        ax3.tick_params(axis='both', which='major', labelsize=6)

        cbar_ax = fig.add_axes([0.95, 0.3, 0.01, 0.4])
        fig.colorbar(im, cax=cbar_ax)

        plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred_{tt*(1/(nt-1)):.1f}.png'),dpi=600)
        plt.close()

def _helmholtz3d(args, apply_fn, params, result_dir, e, resol):
    print("visualizing solution...")

    x = jnp.linspace(-1., 1., resol)
    y = jnp.linspace(-1., 1., resol)
    z = jnp.linspace(-1., 1., resol)
    xm, ym, zm = jnp.meshgrid(x, y, z, indexing='ij')
    if args.model == 'pinn':
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
        z = zm.reshape(-1, 1)
    else:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        z = z.reshape(-1, 1)

    u_ref = helmholtz3d_exact_u(args.a1, args.a2, args.a3, xm, ym, zm)

    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)
    u_pred = apply_fn(params, x, y, z)
    if args.model == 'pinn':
        u_pred = u_pred.reshape(resol, resol, resol)
        u_ref = u_ref.reshape(resol, resol, resol)

    fig = plt.figure(figsize=(14, 8))
    plt.suptitle(f"Epoch = {e}", fontsize=20, y=0.98)  
    plt.axis('off')
    
    
    # reference solution
    ax1 = fig.add_subplot(231, projection='3d')
    im1 = ax1.scatter(xm, ym, zm, c=u_ref, cmap = 'seismic', s=0.5)
    ax1.set_xlabel('x', labelpad=10)
    ax1.set_ylabel('y', labelpad=10)
    ax1.set_zlabel('z', labelpad=10)
    ax1.set_title(f'Reference $u(x, y, z)$', fontsize=15)
    
    fig.colorbar(im1,pad=0.15,fraction=0.03, shrink=0.8, aspect=10)
    
    norm = matplotlib.colors.Normalize(vmin=u_ref.min(),vmax=u_ref.max()) 

    # prediction solution
    ax2 = fig.add_subplot(232, projection='3d')
    im2 = ax2.scatter(xm, ym, zm, c=u_pred, cmap = 'seismic', s=0.5, norm=norm)
    ax2.set_xlabel('x', labelpad=10)
    ax2.set_ylabel('y', labelpad=10)
    ax2.set_zlabel('z', labelpad=10)
    ax2.set_title(f'Predicted $u(x, y, z)$', fontsize=15)
    
    fig.colorbar(im2,pad=0.15,fraction=0.03, shrink=0.8, aspect=10)

    vmax_norm2 = jnp.abs(u_ref-u_pred).max()
    norm2 = matplotlib.colors.Normalize(vmin=0.00,vmax=vmax_norm2) 

    # absolute error
    ax3 = fig.add_subplot(233, projection='3d')
    im3 = ax3.scatter(xm, ym, zm, c=jnp.abs(u_ref-u_pred), cmap = 'seismic', s=0.5, norm=norm2)
    ax3.set_xlabel('x', labelpad=10)
    ax3.set_ylabel('y', labelpad=10)
    ax3.set_zlabel('z', labelpad=10)
    ax3.set_title(f'Absolute error $u(x, y, z)$', fontsize=15)

    bar3 = fig.colorbar(im3,pad=0.15,fraction=0.03, shrink=0.8, aspect=10)
    bar3.ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))

    faces = {
        'y = -1.0': (xm[:, 0, :], zm[:, 0, :], jnp.abs(u_ref-u_pred)[:, 0, :]),  
        'y = 0.0': ((xm[:, 24, :]+xm[:, 25, :])/2., (zm[:, 24, :]+zm[:, 25, :])/2., (jnp.abs(u_ref-u_pred)[:, 24, :]+jnp.abs(u_ref-u_pred)[:, 25, :])/2.),  
        'y = 1.0': (xm[:, 49, :], zm[:,49, :], jnp.abs(u_ref-u_pred)[:, 49, :]),  
    }

    for i, (face_name, (x_coords, y_coords, face_error)) in enumerate(faces.items()):
        ax = fig.add_subplot(2, 3, i + 4)
        scatter = ax.scatter(x_coords, y_coords, c=face_error, cmap='seismic', s=18, norm=norm2)
        ax.set_title(f'Absoulte error: {face_name} slice', fontsize=15)
        ax.set_xlabel('x' if 'x' not in face_name else 'y', labelpad=10)
        ax.set_ylabel('y' if 'y' not in face_name else 'z', labelpad=10)
        ax.set_position([ax.get_position().x0, ax.get_position().y0, 0.25, ax.get_position().height*0.8]) 

        ax.set_aspect('equal')  
        
        bar = fig.colorbar(scatter,pad=0.15,fraction=0.03, shrink=0.8, aspect=10)
        bar.ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))

    fig.subplots_adjust(wspace=0.4, hspace=0.2, left=0.05, right=0.95, top=0.9, bottom=0.1)
    # plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred.png'), bbox_inches='tight')
    plt.savefig(os.path.join(result_dir, f'vis/{e:05d}.png'), bbox_inches='tight') 
    
    plt.close()


def _klein_gordon3d(args, apply_fn, params, result_dir, e, resol):
    print("visualizing solution...")

    t = jnp.linspace(0., 10., resol)
    x = jnp.linspace(-1., 1., resol)
    y = jnp.linspace(-1., 1., resol)
    tm, xm, ym = jnp.meshgrid(t, x, y, indexing='ij')
    if args.model == 'pinn':
        t = tm.reshape(-1, 1)
        x = xm.reshape(-1, 1)
        y = ym.reshape(-1, 1)
    else:
        t = t.reshape(-1, 1)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    u_ref = klein_gordon3d_exact_u(tm, xm, ym, args.k)

    os.makedirs(os.path.join(result_dir, f'vis/{e:05d}'), exist_ok=True)
    u_pred = apply_fn(params, t, x, y)
    if args.model == 'pinn':
        u_pred = u_pred.reshape(resol, resol, resol)
        u_ref = u_ref.reshape(resol, resol, resol)

    fig = plt.figure(figsize=(14, 8))
    plt.suptitle(f"Epoch = {e}", fontsize=20, y=0.98)  
    plt.axis('off')
    
    
    # reference solution
    ax1 = fig.add_subplot(231, projection='3d')
    im1 = ax1.scatter(tm, xm, ym, c=u_ref, cmap = 'seismic', s=0.5)
    ax1.set_xlabel('t', labelpad=10)
    ax1.set_ylabel('x', labelpad=10)
    ax1.set_zlabel('y', labelpad=10)
    ax1.set_title(f'Reference $u(t, x, y)$', fontsize=15)
    
    fig.colorbar(im1,pad=0.25,fraction=0.03, shrink=0.8, aspect=10)
    
    norm = matplotlib.colors.Normalize(vmin=u_ref.min(),vmax=u_ref.max()) 

    # prediction solution
    ax2 = fig.add_subplot(232, projection='3d')
    im2 = ax2.scatter(tm, xm, ym, c=u_pred, cmap = 'seismic', s=0.5, norm=norm)
    ax2.set_xlabel('t', labelpad=10)
    ax2.set_ylabel('x', labelpad=10)
    ax2.set_zlabel('y', labelpad=10)
    ax2.set_title(f'Predicted $u(t, x, y)$', fontsize=15)
    
    fig.colorbar(im2,pad=0.25,fraction=0.03, shrink=0.8, aspect=10)

    norm2 = matplotlib.colors.Normalize(vmin=0.00,vmax=jnp.abs(u_ref-u_pred).max()) 

    # absolute error
    ax3 = fig.add_subplot(233, projection='3d')
    im3 = ax3.scatter(tm, xm, ym, c=jnp.abs(u_ref-u_pred), cmap = 'seismic', s=0.5, norm=norm2)
    ax3.set_xlabel('t', labelpad=10)
    ax3.set_ylabel('x', labelpad=10)
    ax3.set_zlabel('y', labelpad=10)
    ax3.set_title(f'Absolute error $u(t, x, y)$', fontsize=15)

    bar3 = fig.colorbar(im3,pad=0.25,fraction=0.03, shrink=0.8, aspect=10)
    bar3.ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))

    faces = {
        'y = -1.0': (tm[:, :, 0], xm[:, :, 0], jnp.abs(u_ref-u_pred)[:, :, 0]),  
        'y = 0.0': ((tm[:, :, 24]+tm[:, :, 25])/2., (xm[:, :, 24]+xm[:, :, 25])/2., (jnp.abs(u_ref-u_pred)[:, :, 24]+jnp.abs(u_ref-u_pred)[:, :, 25])/2.),  
        'y = 1.0': (tm[:, :, 49], xm[:, :, 49], jnp.abs(u_ref-u_pred)[:, :, 49]),  
    }

    for i, (face_name, (t_coords, x_coords, face_error)) in enumerate(faces.items()):
        ax = fig.add_subplot(2, 3, i + 4)
        scatter = ax.scatter(t_coords, x_coords, c=face_error, cmap='seismic', s=18, norm=norm2)
        ax.set_title(f'Absoulte error: {face_name} slice', fontsize=15)
        ax.set_xlabel('t' if 't' not in face_name else 'x', labelpad=10)
        ax.set_ylabel('x' if 'x' not in face_name else 'y', labelpad=10)
        # ax.set_aspect('equal') 
        ax.set_aspect(5) 
        
        bar = fig.colorbar(scatter,pad=0.25,fraction=0.03, shrink=0.8, aspect=10)
        bar.ax.yaxis.set_major_formatter(FuncFormatter(format_ticks))

    fig.subplots_adjust(wspace=0.4, hspace=0.2, left=0.05, right=0.95, top=0.9, bottom=0.1)
    # plt.savefig(os.path.join(result_dir, f'vis/{e:05d}/pred.png'), bbox_inches='tight')
    plt.savefig(os.path.join(result_dir, f'vis/{e:05d}.png'), bbox_inches='tight') #只是为了画视频
    
    plt.close()

def show_solution(args, apply_fn, params, test_data, result_dir, e, resol=50):
    if args.equation == 'diffusion3d':
        _diffusion3d(args, apply_fn, params, test_data, result_dir, e, resol)
    elif args.equation == 'helmholtz3d':
        _helmholtz3d(args, apply_fn, params, result_dir, e, resol)
    elif args.equation == 'klein_gordon3d':
        _klein_gordon3d(args, apply_fn, params, result_dir, e, resol)
    else:
        raise NotImplementedError