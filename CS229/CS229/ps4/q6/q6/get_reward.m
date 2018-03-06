function R = get_reward(state)
    x = state(1);
    x_dot = state(2);
    theta = state(3);
    theta_dot = state(4);
    
%     if abs(x) < 0.5  && abs(theta) < 1*pi/180
%         R = 1;
%     else
%         R = -0.01;
%     end
    if abs(x) >2.4 || abs(theta)>12*pi/180
        R=-1;
    else
        R=0;
    end
end