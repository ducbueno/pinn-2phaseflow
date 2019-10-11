function [P, V] = Pres(Grid, S, Fluid, q)

% Compute K*lambda(S)
[Mw, Mo] = RelPerm(S, Fluid);
Mt = Mw+Mo;
KM = Mt'.*Grid.K;

% Compute pressure and extract fluxes
[P, V] = TPFA(Grid, KM, q);