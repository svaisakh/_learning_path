function W = weights(x, X, tau)
    
    W=diag(sum(exp(-(X-x).^2/(2*tau^2)),2));
end

