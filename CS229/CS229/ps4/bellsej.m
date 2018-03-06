%------------------------------------------------------------
% ICA

load mix.dat	% load mixed sources
Fs = 11025; %sampling frequency being used

% listen to the mixed sources
normalizedMix = 0.99 * mix ./ (ones(size(mix,1),1)*max(abs(mix)));

% handle writing in both matlab and octave
v = version;
if (v(1) <= '3') % assume this is octave
  audiowrite('mix1.wav', normalizedMix(:, 1), Fs, 16);
  audiowrite('mix2.wav', normalizedMix(:, 2), Fs, 16);
  audiowrite('mix3.wav', normalizedMix(:, 3), Fs, 16);
  audiowrite('mix4.wav', normalizedMix(:, 4), Fs, 16);
  audiowrite('mix5.wav', normalizedMix(:, 5), Fs, 16);
else
  audiowrite('mix1.wav', normalizedMix(:, 1), Fs);
  audiowrite('mix2.wav', normalizedMix(:, 2), Fs);
  audiowrite('mix3.wav', normalizedMix(:, 3), Fs);
  audiowrite('mix4.wav', normalizedMix(:, 4), Fs);
  audiowrite('mix5.wav', normalizedMix(:, 5), Fs);
end

W=eye(5);	% initialize unmixing matrix
m=size(normalizedMix,1);

% this is the annealing schedule I used for the learning rate.
% (We used stochastic gradient descent, where each value in the 
% array was used as the learning rate for one pass through the data.)
% Note: If this doesn't work for you, feel free to fiddle with learning
%  rates, etc. to make it work.
anneal = [0.1 0.1 0.1 0.05 0.05 0.05 0.02 0.02 0.01 0.01 ...
      0.005 0.005 0.002 0.002 0.001 0.001];

for iter=1:length(anneal)
   
   %%%% here comes your code part (should not be much, ours was about 10 lines of code)
   
   order = randperm(m);
   
   for i=1:m
       x = normalizedMix(order(m),:)';
       W = W + anneal(iter)*((1-2*sigmoid(W*x))*x'+W'^(-1));
   end

end;


%%%% After finding W, use it to unmix the sources.  Place the unmixed sources 
%%%% in the matrix S (one source per column).  (Your code.) 

S = W*normalizedMix;

S=0.99 * S./(ones(size(mix,1),1)*max(abs(S))); 	% rescale each column to have maximum absolute value 1 

% now have a listen --- You should have the following five samples:
% * Godfather
% * Southpark
% * Beethoven 5th
% * Austin Powers
% * Matrix (the movie, not the linear algebra construct :-) 

v = version;
if (v(1) <= '3') % assume this is octave
  audiowrite('unmix1.wav', S(:, 1), Fs, 16);
  audiowrite('unmix2.wav', S(:, 2), Fs, 16);
  audiowrite('unmix3.wav', S(:, 3), Fs, 16);
  audiowrite('unmix4.wav', S(:, 4), Fs, 16);
  audiowrite('unmix5.wav', S(:, 5), Fs, 16);
else
  audiowrite(S(:, 1), Fs, 16, 'unmix1.wav');
  audiowrite(S(:, 2), Fs, 16, 'unmix2.wav');
  audiowrite(S(:, 3), Fs, 16, 'unmix3.wav');
  audiowrite(S(:, 4), Fs, 16, 'unmix4.wav');
  audiowrite(S(:, 5), Fs, 16, 'unmix5.wav');
end
