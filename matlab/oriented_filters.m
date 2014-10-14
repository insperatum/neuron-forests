sigma1 = 3;
sigma2 = 6;
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

for i = 1:size(directions,1)
    m = max(fs{i}(:));
    subplot(2, 4, 1); imshow(fs{i}(:,:,floor(4*r/4)), [-m m], 'InitialMagnification', 1000);
    subplot(2, 4, 2); imshow(fs{i}(:,:,floor(5*r/4)), [-m m], 'InitialMagnification', 1000);
    subplot(2, 4, 3); imshow(fs{i}(:,:,floor(6*r/4)), [-m m], 'InitialMagnification', 1000);
    subplot(2, 4, 4); imshow(fs{i}(:,:,floor(7*r/4)), [-m m], 'InitialMagnification', 1000);
    subplot(2, 4, 5); imshow(fs{i}(:,:,floor(8*r/4)), [-m m], 'InitialMagnification', 1000);
    subplot(2, 4, 6); imshow(fs{i}(:,:,floor(9*r/4)), [-m m], 'InitialMagnification', 1000);
    subplot(2, 4, 7); imshow(fs{i}(:,:,floor(10*r/4)), [-m m], 'InitialMagnification', 1000);
    subplot(2, 4, 8); imshow(fs{i}(:,:,floor(11*r/4)), [-m m], 'InitialMagnification', 1000);
    k = waitforbuttonpress;
    disp('hi');
end