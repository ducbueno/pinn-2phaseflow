function K_smooth = smoothdiff(K, wlen)
K = [K K(end)];
dK = diff(K);

dK_smooth = smoothdata(dK, 'loess', wlen);
K_smooth = cumsum(dK_smooth);

K_smooth = K_smooth - min(K_smooth) + min(K);