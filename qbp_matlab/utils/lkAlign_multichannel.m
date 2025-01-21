function uv = lkAlign_multichannel(im0, im1, iters, uv0)
%LKALIGN Align two images (2D translation) using Lucas-Kanade for both single and multi-channel images

if nargin < 4 || isempty(uv0)
    uv = zeros(1, 1, 2, class(uv0));
else
    uv = reshape(uv0, [1, 1, 2]);
end

for i = 1:iters
    It = 0; Ix = 0; Iy = 0;
    for c=1:size(im0,3)
        [It_, Ix_, Iy_] = partial_deriv_patch(im0(:,:,c), im1(:,:,c), uv);
        It = It + It_;
        Ix = Ix + Ix_;
        Iy = Iy + Iy_;
    end
    
    % If images are multi-channel, sum over channels
    if ndims(im0) == 3
        It = sum(It, 3);
        Ix = sum(Ix, 3);
        Iy = sum(Iy, 3);
    end
    
    A = [Ix(:), Iy(:)];
    b = -It(:);
    
    if rank(A) < 2
        break
    end
    
    x = A \ b;
    if norm(x) > 1
        x = x / norm(x);
    end
    
    uv = uv + reshape(x, size(uv));
end

uv = reshape(uv, [1, 2]);

end
