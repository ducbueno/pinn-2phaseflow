#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

data = np.load('data/2phaseflow_smooth.npz')
x0 = data['x0']
tb = data['tb']
lb = data['lb']
ub = data['ub']
P_pred = data['P_pred']
S_pred = data['S_pred']
Exact_p = data['Exact_p']
Exact_s = data['Exact_s']
t = data['t']
x = data['x']

error_p = np.linalg.norm(Exact_p.flatten() - P_pred.T.flatten(), 2)/np.linalg.norm(Exact_p.flatten(), 2)
print('Error p: %e' % (error_p))

error_s = np.linalg.norm(Exact_s.flatten() - S_pred.T.flatten(), 2)/np.linalg.norm(Exact_s.flatten(), 2)
print('Error s: %e' % (error_s))

X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
X_u_train = np.vstack([X0, X_lb, X_ub])

########################################
####### Pressure #######################
########################################

fig, ax = plt.subplots()
ax.axis('off')

####### Row 0: p(t,x) ##################    
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(P_pred.T, interpolation='nearest', cmap='viridis',
              extent=[lb[1], ub[1], lb[0], ub[0]],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx',
        label = 'Data (%d points)' % (X_u_train.shape[0]),
        markersize = 4, clip_on = False)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[15]*np.ones((2, 1)), line, 'k--', linewidth = 1)
ax.plot(t[60]*np.ones((2, 1)), line, 'k--', linewidth = 1)
ax.plot(t[120]*np.ones((2, 1)), line, 'k--', linewidth = 1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
leg = ax.legend(frameon=False, loc = 'best')
ax.set_title('$p(t,x)$', fontsize = 10)

####### Row 1: p(t,x) slices ##################    
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=1.0)

ax = plt.subplot(gs1[0, 0])
ax.plot(x, Exact_p[:, 15], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x, P_pred[15, :], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$p(t,x)$')
ax.set_title('$t = %.2f$' % (t[15]), fontsize = 10)
ax.axis('square')
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-1.1, 0.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact_p[:, 60], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x, P_pred[60, :], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$p(t,x)$')
ax.axis('square')
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-1.1, 0.1])
ax.set_title('$t = %.2f$' % (t[60]), fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8),
          ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x, Exact_p[:, 120], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x, P_pred[120, :], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$p(t,x)$')
ax.axis('square')
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-1.1, 0.1])
ax.set_title('$t = %.2f$' % (t[120]), fontsize = 10)

plt.savefig('img/pressure_spe10_sineK.jpeg', dpi=200)  

########################################
####### Saturation #####################
########################################

fig, ax = plt.subplots()
ax.axis('off')

####### Row 0: s(t,x) ##################    
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])

h = ax.imshow(S_pred.T, interpolation='nearest', cmap='viridis',
              extent=[lb[1], ub[1], lb[0], ub[0]],
              origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)

ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx',
        label = 'Data (%d points)' % (X_u_train.shape[0]),
        markersize = 4, clip_on = False)

line = np.linspace(x.min(), x.max(), 2)[:, None]
ax.plot(t[15]*np.ones((2, 1)), line, 'k--', linewidth = 1)
ax.plot(t[60]*np.ones((2, 1)), line, 'k--', linewidth = 1)
ax.plot(t[120]*np.ones((2, 1)), line, 'k--', linewidth = 1)

ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
leg = ax.legend(frameon=False, loc = 'best')
ax.set_title('$S(t,x)$', fontsize = 10)

####### Row 1: s(t,x) slices ##################    
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
ax.plot(x, Exact_s[:, 15], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x, S_pred[15, :], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$S(t,x)$')
ax.set_title('$t = %.2f$' % (t[15]), fontsize = 10)
ax.axis('square')
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])

ax = plt.subplot(gs1[0, 1])
ax.plot(x, Exact_s[:, 60], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x, S_pred[60, :], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$S(t,x)$')
ax.axis('square')
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])
ax.set_title('$t = %.2f$' % (t[60]), fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8),
          ncol=5, frameon=False)

ax = plt.subplot(gs1[0, 2])
ax.plot(x, Exact_s[:, 120], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x, S_pred[120, :], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$S(t,x)|$')
ax.axis('square')
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])
ax.set_title('$t = %.2f$' % (t[120]), fontsize = 10)

plt.savefig('img/saturation_spe10_sineK.jpeg', dpi=200)  