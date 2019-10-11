function [S, m, f] = Upstream(Grid, S, Fluid, V, q, T)

Nx = Grid.Nx;                                          % number of grid points
N = Nx;                                                % number of unknowns
pv = Grid.V.*Grid.por;                                 % pore volume=cell volume*porosity

fi = max(q,0);                                         % inflow from wells
XP = max(V, 0); XN = min(V, 0);                        % influx and outflux, x-faces

Vi = XP(1:Nx)-XN(2:Nx+1);                              % total flux into each gridblock
pm = min(pv./(Vi(:)+fi));                              % estimate of influx
cfl = ((1-Fluid.swc-Fluid.sor)/3)*pm;                  % CFL restriction
Nts = ceil(T/cfl);                                     % number of local time steps
dtx = (T/Nts)./pv;                                     % local time steps

A = GenA(Grid, V, q);                                  % system matrix
A = spdiags(dtx, 0, N, N)*A;                           % A * dt/|Omega_i|
fi = max(q, 0).*dtx;                                   % injection

for t = 1:Nts
    [mw, mo] = RelPerm(S, Fluid);                      % compute mobilities
    fw = mw./(mw+mo);                                  % compute fractional flow
    S = S+(A*fw+fi);                                   % update saturation
end

m = mw + mo;
f = fw;