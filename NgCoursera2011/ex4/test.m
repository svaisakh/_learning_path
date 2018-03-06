load('ex4data1.mat');
hiddenLayer1Neurons=30;
hiddenLayer2Neurons=15;
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 50);

%  You should also try different values of lambda
lambda = 1;

epsilon_init=0.08;
initial_nn_params= rand(hiddenLayer1Neurons*401+hiddenLayer2Neurons*(hiddenLayer1Neurons+1)+10*(hiddenLayer2Neurons+1), 1) * 2 * epsilon_init - epsilon_init;

yMatrix = zeros(10,numel(y));
for k=1:10
    yMatrix(k,:) = (y==k);
end
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCost(p, [400 hiddenLayer1Neurons hiddenLayer2Neurons 10], X', yMatrix, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1=reshape(nn_params(1:hiddenLayer1Neurons*401),hiddenLayer1Neurons,401);
Theta2=reshape(nn_params(hiddenLayer1Neurons*401+1:hiddenLayer1Neurons*401+hiddenLayer2Neurons*(hiddenLayer1Neurons+1)),hiddenLayer2Neurons,hiddenLayer1Neurons+1);
pred = predict(nn_params, [400 hiddenLayer1Neurons hiddenLayer2Neurons 10], X');
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));