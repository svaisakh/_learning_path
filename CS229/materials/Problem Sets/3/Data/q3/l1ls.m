function theta = l1ls(X,y,lambda)

%%% YOUR CODE HERE
    
    n = size(X,2);
    
    theta = rand(n,1);
    
    for iter = 1:1000
        for i=1:n
            theta(i) = update(i)-lambda;
            costplus = cost(theta);
            theta(i) = theta(i)+2*lambda;
            costminus = cost(theta);
            if costplus<costminus
                theta(i) = max(theta(i)-2*lambda,0);
            else
                theta(i) = min(theta(i),0);
            end
        end
    end
    
    function J = cost(t)
        J = 0.5*norm(X*t-y)+lambda*norm(t,1);
    end

    function u=update(i)
        temp = theta(i);
        theta(i) = 0;
        u = -(X(:,i)'*X(:,i))^(-1)* X(:,i)' * (X*theta-y);
        theta(i) = temp;
    end

end