Grid.Nx = 256; Dx = 1; Grid.hx = Dx/Grid.Nx;            % Dimension in x-direction
N = Grid.Nx;                                             % Total number of grid blocks
Grid.V = Grid.hx;                                        % Cell volumes
Grid.K = ones(3, Grid.Nx);                               % Unit permeability
Grid.por = ones(Grid.Nx, 1)/0.375;                             % Unit porosity

Q = zeros(N, 1); Q([1 N]) = [1 -1];                      % Production/injection

Fluid.vw = 0.09;  Fluid.vo = 0.9;                         % Viscosities
Fluid.swc = 0.0; Fluid.sor = 0.0;                        % Irreducible saturations

%S = zeros(N, 1);                                         % Initial saturation
S = 0.1 - 0.1*linspace(0, 1, N)';
m = zeros(N, 1);
f = zeros(N, 1);
nt = 128; dt = 1/nt;                                     % Time steps

x = linspace(0, Dx, Grid.Nx);
tt = linspace(0, 1, nt);

S_history = zeros(N, nt);
P_history = zeros(N, nt);
m_history = zeros(N, nt);
f_history = zeros(N, nt);

for t=1:nt
    [P, V] = Pres(Grid, S, Fluid, Q);                    % pressure solver
    [S, m, f] = Upstream(Grid, S, Fluid, V, Q, dt);      % saturation solver
    S_history(:, t) = S;
    P_history(:, t) = P;
    m_history(:, t) = m;
    f_history(:, t) = f;

    plot(P)
    axis([0 N -1 0.1])
    drawnow;                                            % force update of plot
end

save('../2phaseflow.mat', 'x', 'tt', 'P_history', 'S_history');
