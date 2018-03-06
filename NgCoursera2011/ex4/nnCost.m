function [cost, grad] = nnCost(thetaVector, sizeVector, X, y, regularizationParameter)
%nnCost Returns the cost and gradient associated with a particular neural
%network
%   thetaVector is the rolled up vector of all the parameters of the
%   network starting from the first layer in chronological order (i.e.
%   (1,0) (1,1)... upto (L-1, K).
%   
%   sizeVector is a vector of the number of neurons in each layer.
%
%   X is the input matrix
%
%   y is the trained output in a logical classifier form.
%
%   regularizationParameter is an optional parameter to avoid overfitting.

    % Infer parameters implicit from arguments
    m = size(y,2);
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
    
    % Backpropagation
    
    deltaMatrices = cell(1, L);
    gradientMatrices = cell(1, L-1);
    
    deltaMatrices{L} = activationMatrices{L} - y;
    
    for l=L-1:-1:2
        deltaMatrices{l} = (thetaMatrices{l}'*deltaMatrices{l+1}).*activationMatrices{l}.*(1-activationMatrices{l});
        deltaMatrices{l}(1,:) = [];
        
        gradientMatrices{l} = deltaMatrices{l+1}*activationMatrices{l}'/m;
    end
    gradientMatrices{1}=deltaMatrices{2}*activationMatrices{1}'/m;
    
    % Calculation of regularization error
    
    regularizationError=0;
    for l=1:L-1
        regularizationError = regularizationError + sum(sum(thetaMatrices{l}.^2))-sum(sum(thetaMatrices{l}(:,1).^2));
    end
    regularizationError = (regularizationParameter/(2*m))*regularizationError;
    
    % Calculate cost and regularize gradient
    
    cost = (-1/m)*sum(sum(y.*log(activationMatrices{L})+(1-y).*log(1-activationMatrices{L}))) + regularizationError;
    
    for l=1:L-1
        gradientMatrices{l}=gradientMatrices{l}+(regularizationParameter/m)*thetaMatrices{l};
        gradientMatrices{l}(:,1)=gradientMatrices{l}(:,1)-(regularizationParameter/m)*thetaMatrices{l}(:,1);
    end
    
    % Unroll gradient
    
    grad=zeros(size(thetaVector));
    
    startIndex = 1;
    for l=1:L-1
        grad(startIndex : startIndex+numel(gradientMatrices{l})-1)=reshape(gradientMatrices{l},1,numel(gradientMatrices{l}));
        startIndex = startIndex+numel(gradientMatrices{l});
    end

end

