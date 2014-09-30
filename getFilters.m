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
    
    %todo:
    %getPlaneDerivativeFilters(...);
end
                
function [fs] = getPlaneDerivativeFilters(sigma1, sigma23)
    r = max(sigma1, sigma23);
    [Y, X, Z] = meshgrid(-2*r:2*r, -2*r:2*r, -2*r:2*r);

    s2 = 1/sqrt(2); s3=1/sqrt(3);
    directions = [0 0 1; 0 s2 -s2; 0 1 0; 0 s2 s2; ...
                  s3 -s3 -s3; s2 -s2 0; s3 -s3 s3; ...
                  s2 0 -s2; 1 0 0; s2 0 s2; ...
                  s3 s3 -s3; s2 s2 0; s3 s3 s3 ];
              
    fs = cell(1, size(directions,1));
    for i = 1:size(directions,1)
        v = directions(i,:)';
        V = [v null(v')];
        D = diag([sigma1 sigma23*ones(1, length(v)-1)]);
        C = V * D^2 * V';
        gFil = reshape(mvnpdf([X(:) Y(:) Z(:)], [0 0 0], C), length(X), length(Y), length(Z));
        dFil = cat(3, [-(v(1)+v(2)+v(3))/2 v(2)/2; v(1)/2 0], [v(3)/2 0; 0 0]);
        fs{i} = convn(gFil, dFil);
    end
end

