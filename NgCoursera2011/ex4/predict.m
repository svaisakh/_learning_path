function p = predict(thetaVector, sizeVector, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

 % Infer parameters implicit from arguments
    m = size(X,2);
    L = length(sizeVector);
    
    % Roll up thetaVector into new matrices.

    thetaMatrices = cell(1, L-1);

    startIndex = 1;
    for l=1:L-1
        matrixSize = [sizeVector(l+1) (sizeVector(l)+1)];
        thetaMatrices{l} = reshape(thetaVector(startIndex : startIndex + prod(matrixSize) - 1),matrixSize(1), matrixSize(2));
        startIndex = startIndex + prod(matrixSize);
    end
    
    % Feed-forward calculation of activations
    
    activationMatrices = cell(1, L);
    
    activationMatrices{1} = [ones(1, m); X];
    for l=2:L
        activationMatrices{l} = sigmoid(thetaMatrices{l-1}*activationMatrices{l-1});
        activationMatrices{l} = [ones(1, m); activationMatrices{l}];
    end
    activationMatrices{L}(1,:)=[];
    
    [~, p]=max(activationMatrices{L});
    p=p'
end
