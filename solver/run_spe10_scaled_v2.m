load('K_spe10.mat');
load('phi_spe10.mat');
% K = K./max(K);
% phi = phi'/max(phi);

Pc = 7.6148e+10;
L = 670.56;
mu = 3e-4;
mu_o = 3e-3;
beta = mu_o/mu;
Kc = max(K);
lambda_c = Kc/mu;
tc = L^2/(lambda_c*Pc);
epsilon = 9.3529*tc;

Grid.Nx = 220; Dx = 1; Grid.hx = Dx/Grid.Nx;             % Dimension in x-direction
N = Grid.Nx;                                             % Total number of grid blocks
Grid.K =  K/Kc;
Grid.por = phi';
Grid.V = Grid.hx;                                        % Cell volumes

Q = zeros(N, 1); Q([1 N]) = [epsilon -epsilon];          % Production/injection

Fluid.vw = 1; Fluid.vo = beta;                           % Viscosities
Fluid.swc = 0.0; Fluid.sor = 0.0;                        % Irreducible saturations

S = zeros(N, 1);                                         % Initial saturation
m = zeros(N, 1);
f = zeros(N, 1);
nt = 128; dt = 1/nt;                                     % Time steps

x = linspace(0, 1/L, Grid.Nx);
tt = linspace(0, 0.245*Pc/(L^2), nt);

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
