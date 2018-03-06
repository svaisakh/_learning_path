function t = linreg(X,y,varargin)

    p = inputParser;
    addRequired(p,'X',@isnumeric);
    addRequired(p,'y',@isnumeric);
    addOptional(p,'W',[],@isnumeric);
    addOptional(p,'lambda',0,@isnumeric);
    parse(p,X,y,varargin{:});
    
    X=p.Results.X;
    y=p.Results.y;
    W=p.Results.W;
    lambda=p.Results.lambda;
    
    if(isempty(W))
        W=eye(length(y));
    end
    
    m = size(X,1);
    n = size(X,2);
    
    if(m~=length(y))
        error('Dimension mismatch');
    end
    
    X=[ones(m,1) X];
    
    t = ((X'*W*X + lambda*eye(n+1))^-1)*X'*W*y;
end

