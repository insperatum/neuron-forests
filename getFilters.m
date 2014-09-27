function [fs] = getFilters(sigmas)
    d = {1, [1; -1], [1, -1], permute([1;-1], [2,3,1])};
    coreFilters = {d{1}, d{2}, d{3}, d{4}, ...
                   convn(d{2}, d{2}), convn(d{2}, d{3}), convn(d{2}, d{4}), ...
                   convn(d{3}, d{3}), convn(d{3}, d{4}), convn(d{4}, d{4})};
    
    function [f] = getNormFilter(sigma)
        if(sigma == 0)
            f = 1;
        else
            r = ceil(2 * sigma);
            n = 2*r + 1;
            f1 = normpdf(-r : r, 0, sigma);
            f = reshape(kron(f1'*f1,f1),[n,n,n]);
        end
    end
    
    fs = cell(length(sigmas), 3);
    for i = 1:length(sigmas)
        sigma = sigmas(i);
        normFilter = getNormFilter(sigma);
        for j = 1:length(coreFilters)
            fs{i, j} = convn(normFilter, coreFilters{j});
        end
    end
    fs = fs(:);
end

