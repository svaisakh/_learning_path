function [t, i] = nropt(t0,n,G,H)
    warning off;
    sufficiency=1.0e-10;
    t=t0;

    
    for i=1:n
        dt=((H(t))^(-1)) * G(t);
        if(sum(abs(dt)>sufficiency))
            t = t - dt;
        else
            i=i-1;
            break;
        end
    end
end

