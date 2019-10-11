function [P, V] = TPFA(Grid, K, q)

% Compute transmissibilities by harmonic averaging
Nx = Grid.Nx; N = Nx;
hx=Grid.hx;
L = K.^(-1);
tx = 2/hx; TX = zeros(Nx+1, 1);
TX(2:Nx) = tx./(L(1:Nx-1)+L(2:Nx));

% Assemble TPFA discretization matrix.
x1 = TX(1:Nx); x2 = TX(2:Nx+1);
DiagVecs = [-x2, x1+x2, -x1];
DiagIndx = [-1, 0, 1];
A = spdiags(DiagVecs, DiagIndx, N, N);
A(1, 1) = A(1, 1) + sum(Grid.K(:, 1));

% Solve linear system and extract interface fluxes.
P = A\q;
V = zeros(Nx+1, 1);
V(2:Nx) = (P(1:Nx-1)-P(2:Nx)).*TX(2:Nx);