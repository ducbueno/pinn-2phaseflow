function A = GenA(Grid, V, q)

Nx = Grid.Nx;
N = Nx;                                                     % number of unknowns
fp = min(q, 0);                                             % production

XN = min(V, 0); x1 = XN(1:Nx);                              % separate flux into positive coordinate
XP = max(V, 0); x2 = XP(2:Nx+1);                            % separate flux into negative coordinate

DiagVecs = [x2, fp+x1-x2, -x1];                             % diagonal vectors
DiagIndx = [-1, 0, 1];                                      % diagonal index
A = spdiags(DiagVecs, DiagIndx, N, N);                      % matrix with upwind FV stencil
