pause_time = 0.000;
min_trial_length_to_start_display = 1000;
min_attempts_to_start_display = 0;
display_started=0;
learn = true;

NUM_STATES = 163;
NUM_ACTIONS = 3;

GAMMA=0.995;
TAU = 5;
LAMBDA = 0.00; 

TOLERANCE=0.0000000000000001;

NO_LEARNING_THRESHOLD = 20;

%%%%%%%%%%  End parameter list   %%%%%%%%%%

% Time cycle of the simulation
time=0;

% These variables perform bookkeeping (how many cycles was the pole
% balanced for before it fell). Useful for plotting learning curves.

max_failures=100000; % You should reach convergence well before this.  

time_steps_to_failure=zeros(max_failures,1);
num_failures=0;
time_at_start_of_current_trial=0;

% Starting state is (0 0 0 0)
% x, x_dot, theta, theta_dot represents the actual continuous state vector
x = -1.1 + rand(1)*2.2; x_dot = 0.0; theta = (rand()*2-1)*1*pi/180; theta_dot = 0.0;

% state is the number given to this state - you only need to consider
% this representation of the state
state = [x, x_dot, theta, theta_dot]';

if min_trial_length_to_start_display==0 || display_started==1
    show_cart(x, x_dot, theta, theta_dot, pause_time);
end

%%% CODE HERE: Perform all your initializations here %%%

% Assume no transitions or rewards have been observed
% Initialize the value function array to small random values (0 to 0.10,
% say)
% Initialize the transition probabilities uniformly (ie, probability of
% transitioning for state x to state y using action a is exactly
% 1/NUM_STATES). Initialize all state rewards to zero.

values = rand(NUM_STATES,1)*0.10;
transitionprob = ones(NUM_STATES,NUM_STATES,NUM_ACTIONS)/NUM_STATES;
rewards = zeros(NUM_STATES,1);
transitionprobest = zeros(NUM_STATES,NUM_STATES,NUM_ACTIONS);
rewardsest = zeros(NUM_STATES,1);
statecount = zeros(NUM_STATES,1);

chooserrandaction = false;
if (learn)
    param = zeros(5,1);
    rng('shuffle');
    final_epsilon = 0.01;
    pruning_rate = 1-final_epsilon^(1/max_failures);
    epsilon = 1;
else
    epsilon = 0;
%     param = [1 -1 -0.1 -10 -1]';%mine
%     param = [105.5220    5.8031    0.9593  -28.7187   -6.5822]';%norma
param=1.0e+03 *[4.9459    0.0004   -0.0005   -0.0175    0.0005]';
%     param = [172.6868   92.5331   10.1836  -30.9302  -14.9861]';%fast
%     param = 1.0e+04 *[2.6054    2.0427    0.2373   -0.8941   -0.4231]';%superfast
end
no_learning_trials=0;
statematrix=[];
tau = [];
num_examples = 0;
y=zeros(2,1);


%%%% END YOUR CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%





%%% CODE HERE (while loop condition) %%%
% This is the criterion to end the simulation
% You should change it to terminate when the previous
% 'NO_LEARNING_THRESHOLD' consecutive value function computations all
% converged within one value function iteration. Intuitively, it seems
% like there will be little learning after this, so end the simulation
% here, and say the overall algorithm has converged. 

while num_failures<max_failures
% while (number of consecutive no learning trials < NO_LEARNING_THRESHOLD)
    if (no_learning_trials>=NO_LEARNING_THRESHOLD)
        break;
    end

  %%% CODE HERE: Write code to choose action (1 or 2) %%%
  % This action choice algorithm is just for illustration. It may
  % convince you that reinforcement learning is nice for control
  % problems!  Replace it with your code to choose an action that is
  % optimal according to the current value function, and the current MDP
  % model.
  if (~chooserrandaction)
      expectation = zeros(NUM_ACTIONS,1);
      for a = 1:NUM_ACTIONS
          [new_x, new_x_dot, new_theta, new_theta_dot]=cart_pole(a, x, x_dot, theta, theta_dot);
          expectation(a) = param'*abs([1 new_x, new_x_dot, new_theta, new_theta_dot])';
      end
      [~,action] = max(expectation);
  else
      action = randi([1 NUM_ACTIONS]);
  end

  %if num_failures<-20
  %  if (rand(1) < 0.5)
  %    action=1;
  %  else
  %    action=2;
  %  end
  %end


  %%% END YOUR CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  

  % Reward function to use - do not change this!
  R = get_reward(state);
  
  % Get the next state by simulating the dynamics
  [x, x_dot, theta, theta_dot] = cart_pole(action, x, x_dot, theta, theta_dot);

  % Increment simulation time
  time = time + 1;
  
  % Get the state number corresponding to new state vector
  new_state = [1 x, x_dot, theta, theta_dot]';
  statematrix= [statematrix; new_state'];

  if display_started==1
    show_cart(x, x_dot, theta, theta_dot, pause_time);
  end
  

  %%% CODE HERE: Perform updates %%%%%%%%%

  % A transition from 'state' to 'new_state' has just been made using
  % 'action'. The reward observed in 'new_state' (note) is 'R'.
  % Write code to update your statistics about the MDP - i.e., the
  % information you are storing on the transitions and on the rewards
  % observed. Do not change the actual MDP parameters, except when the
  % pole falls (the next if block)!
  
  % Recompute MDP model whenever pole falls
  % Compute the value function V for the new model
  if (abs(x) > 2.4 || abs(theta) > 12*pi/180) && learn

      y=zeros(size(statematrix,1),1);
    % Update MDP model using the current accumulated statistics about the
    % MDP - transitions and rewards.
    % Make sure you account for the case when total_count is 0, i.e., a
    % state-action pair has never been tried before, or the state has
    % never been visited before. In that case, you must not change that
    % component (and thus keep it at the initialized uniform distribution).
    for i = 1:size(statematrix,1)
        q = zeros(NUM_ACTIONS,1);
        for a = 1:NUM_ACTIONS
            [new_x, new_x_dot, new_theta, new_theta_dot] = cart_pole(a, statematrix(i,1), statematrix(i,2), statematrix(i,3), statematrix(i,4));
            transstate = abs([1 new_x new_x_dot new_theta new_theta_dot])';
            q(a) = get_reward(statematrix(i,:)) + GAMMA*param'*transstate;
        end
        y(i) = max(q);
    end
    
    [newparam,tau] = linregadapt(param, tau, abs(statematrix),y)
    num_examples = num_examples + size(statematrix,1);
    if max(abs(newparam-param)) < TOLERANCE
        no_learning_trials = no_learning_trials+1;
    end
    param=newparam;
    
    % Perform value iteration using the new estimated model for the MDP
    % The convergence criterion should be based on TOLERANCE as described
    % at the top of the file.
    % If it converges within one iteration, you may want to update your
    % variable that checks when the whole simulation must end

    pause(pause_time); % You can use this to stop for a while!
    
  end
    

  %%% END YOUR CODE %%%%%%%%%%%%%%%%%%%
  

  % Dont change this code: Controls the simulation, and handles the case
  % when the pole fell and the state must be reinitialized
  if (abs(x) > 2.4 || abs(theta) > 12*pi/180)
    num_failures = num_failures+1;
    time_steps_to_failure(num_failures) = time - time_at_start_of_current_trial;
    disp([num_failures    time_steps_to_failure(num_failures)]);
    time_at_start_of_current_trial = time;
    
    if learn
        epsilon = epsilon *(1-pruning_rate);
        if (rand() < epsilon)
            chooserrandaction = true;
        else
            chooserrandaction = false;
        end
  end

    if ((time_steps_to_failure(num_failures)> ...
	min_trial_length_to_start_display) && num_failures> ...
    min_attempts_to_start_display)
      display_started=1;
    end
    
    % Reinitialize state
    x = -1.1 + rand(1)*2.2;
    %x=0.0;
    x_dot = 0.0; theta = (rand()*2-1)*1*pi/180; theta_dot = 0.0;
    state = [x, x_dot, theta, theta_dot time-time_at_start_of_current_trial]';
    statematrix = [];
  else 
    state=new_state;
  end
end

% Plot the learning curve (time balanced vs trial)
plot_learning_curve

function t = regress(x,y,theta,alpha,start_index,iterations)
        t = theta;
        for iter = 1:iterations
            for j=start_index:start_index+size(x,1)-1
                i = j-start_index+1;%randi([1 size(x,1)]);
                t = t - (alpha/sqrt(j))*[1; x(i,:)']*(theta'*[1; x(i,:)'] - y(i));
            end
        end
end

function t = regress2(x,y,theta,alpha,start_index,iterations)
        t = theta;
        for iter = 1:iterations
            t = t - (alpha/sqrt(start_index))*[ones(1,size(x,1)); x']*([ones(1, size(x,1)); x']'*theta - y);
        end
end
