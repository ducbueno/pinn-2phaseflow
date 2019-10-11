Grid.Nx = 220; Grid.hx = 10*.3048;                       % Dimension in x-direction
N = Grid.Nx;                                             % Total number of grid blocks
load('K_spe10.mat');
load('phi_spe10.mat');
Grid.K = K;
Grid.por = phi';
Grid.V = Grid.hx;                                        % Cell volumes

Q = zeros(N, 1); Q([1 N]) = [9.3529 -9.3529];            % Production/injection

Fluid.vw = 3e-4; Fluid.vo = 3e-3;                        % Viscosities
Fluid.swc = 0.0; Fluid.sor = 0.0;                        % Irreducible saturations

S = zeros(N, 1);                                         % Initial saturation
m = zeros(N, 1);
f = zeros(N, 1);
nt = 200; dt = 0.032;                                    % Time steps

x = linspace(0, 1, Grid.Nx);
tt = linspace(0, 0.255, nt);

S_history = zeros(N, nt);
P_history = zeros(N, nt);

for t=1:nt
    [P, V] = Pres(Grid, S, Fluid, Q);                    % pressure solver
    [S, m, f] = Upstream(Grid, S, Fluid, V, Q, dt);      % saturation solver
    S_history(:, t) = S;
    P_history(:, t) = P;
    
    plot(P)
    axis([0 N -1 0.1])
    drawnow;                                            % force update of plot
end

save('../2phaseflow_spe10.mat', 'x', 'tt', 'P_history', 'S_history', 'K', 'phi');
