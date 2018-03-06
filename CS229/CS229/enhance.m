function Y = enhance(X, powers)
    m = size(X,1);
    n = size(X,2);
    
    if (isempty(powers))
        powers=ones(1,n);
    elseif (length(powers)==1)
        powers=powers*ones(1,n);
    elseif(length(powers)>n)
        disp('Error. More powers asked than number of features!');
    end
    
    Y=zeros(m,sum(powers));
    if(isa(X,'sym'))
        Y=sym(Y);
    end
    count=0;
    for i=1:n
        Y(:, count+1:count+powers(i))=X(:,i).^(1:powers(i));
        count = count+powers(i);
    end
    
end

