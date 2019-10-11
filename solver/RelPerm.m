function [Mw, Mo, dMw, dMo] = RelPerm(s, Fluid)

S = (s-Fluid.swc)/(1-Fluid.swc-Fluid.sor);    % Rescale saturations
Mw = S.^2/Fluid.vw;                           % Water mobility
Mo =(1-S).^2/Fluid.vo;                        % Oil mobility

if (nargout == 4)
    dMw = 2*S/Fluid.vw/(1-Fluid.swc-Fluid.sor);
    dMo = -2*(1-S)/Fluid.vo/(1-Fluid.swc-Fluid.sor);
end