function y = lwlr(X_train, y_train, x, tau)
    
    n = size(X_train,2);
    m = length(y_train);
    lambda = 1.0e-4;
    
    t0 = rand(n+1,1);
    
    w = exp(-sum((X_train-x.').^2,2)/(2*tau));
    
    X_train = [ones(m,1) X_train];
% the main task

    t = nropt(t0,100,@grad,@hess);
    
    y = t'*[1; x] > 0.5;
    
    function H = hess(t)
        % hessian
        H=X_train'*diamatrix(t)*X_train-lambda;
    end

    function G = grad(t)
        % gradient
        G=X_train'*z(t)-lambda*t;
    end

    function D = diamatrix(t)
        D=diag(-w.*hyp(t).*(1-hyp(t)));
    end

    function z = z(t)
        z=w.*(y_train-X_train*t);
    end

    function h = hyp(t)
        h=X_train*t;
    end
end