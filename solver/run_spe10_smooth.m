load('K_spe10.mat');
load('phi_spe10.mat');

%K = 1 + 0.25*sin(linspace(0, 1, 220)*2*pi);
%K = smoothdata(K, 'loess', 150);
%K = smoothdata(K, 'gaussian', 20);
K = smoothdiff(K, 50);
%phi = smoothdata(phi, 'gaussian', 10);

Pc = 13.0853;
%K = sqrt(Pc)*K./max(K);
K = K./max(K);
phi = 13.5*phi';
%phi = 13.5*phi';

Grid.Nx = 220; Dx = 1; Grid.hx = Dx/Grid.Nx;             % Dimension in x-direction
N = Grid.Nx;                                             % Total number of grid blocks
Grid.K =  K;
Grid.por = phi;
Grid.V = Grid.hx;                                        % Cell volumes

Q = zeros(N, 1); Q([1 N]) = [1 -1];                      % Production/injection
%Q = zeros(N, 1); Q([1 N]) = [1 -1];

Fluid.vw = 0.1; Fluid.vo = 1;                            % Viscosities
Fluid.swc = 0.0; Fluid.sor = 0.0;                        % Irreducible saturations

S = 0.1 - 0.1*linspace(0, 1, N)';                        % Initial saturation
m = zeros(N, 1);
f = zeros(N, 1);
nt = 128; dt = 1/nt;                                     % Time steps

x = linspace(0, 1, Grid.Nx);
tt = linspace(0, 1, nt);

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

save('../data/2phaseflow_spe10_smooth.mat', 'x', 'tt', 'P_history', 'S_history', 'K', 'phi');