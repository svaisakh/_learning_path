function t = logreg(X,y,varargin)

% parse all the arguments
    p=inputParser;
    addRequired(p,'X',@isnumeric);
    addRequired(p,'y',@isnumeric);
    addParameter(p,'iter',200,@isnumeric);
    addParameter(p,'p',[],@isnumeric);
    addParameter(p,'plot',false,@islogical);
    parse(p,X,y,varargin{:});

    iter=p.Results.iter;
    powers=p.Results.p;
    plot=p.Results.plot;

% validate input    
    
    m = length(y);
    if(m~=size(X,1))
        disp('Oops, training set dimension mismatch. Try again');
        return;
    end

% bookkeeping
    Xsimple = X;
    X = enhance(X,powers);
    n = size(X,2);
    X = [ones(m,1) X];
    
    t = rand(n+1,1);

% the main task
  
    if(plot)
        for i=1:iter
            clf;
            plotdata(Xsimple,y);
            hold on;
            axis manual;
            [t, nri] = nropt(t,1,@grad,@hess);
            plotreg(t,'powers',powers);
            drawnow;
            if(nri<1)
                break;
            end
        end
    else
        t = nropt(t,iter,@grad,@hess);
    end
    
% sub functions

    function H = hess(t)
        % hessian
        h=hyp(t);
        H=(h.*X)'*((1-h).*X);
    end

    function G = grad(t)
        % gradient
        G=-X'*(y-hyp(t));
    end

    function h = hyp(t)
        % hypothesis
        h=sigmoid(X*t);
    end

end

