function W = ica(X)

    %%% YOUR CODE HERE
    batch_size = 1;
    learning_rate = 0.0005;
    iterations = 10;

    m = size(X,1);
    n = size(X,2);

    W = rand(n,n);

    for iter = 1:iterations
        for i=1:m/batch_size
            X_temp = X(i:i+batch_size-1,:);

            dW = learning_rate*((1-2*sigmoid(W*X_temp')) * X_temp + W'^(-1));

            if norm(dW,1)<0.001
                break;
            end

            W = W + dW;
        end
    end

    function g = sigmoid(z)
        g = 1./(1+exp(-z));
    end

end

