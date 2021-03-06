function [gold] = getGold(segmented, type)
    if(nargin > 1 && strcmp(type, 'affinity'))
        seg1 = segmented(1:end-1, 1:end-1, 1:end-1);
        gold(:,:,:,1) = (seg1 ~= 0) & (seg1 == segmented(2:end, 1:end-1, 1:end-1));
        gold(:,:,:,2) = (seg1 ~= 0) & (seg1 == segmented(1:end-1, 2:end, 1:end-1));
        gold(:,:,:,3) = (seg1 ~= 0) & (seg1 == segmented(1:end-1, 1:end-1, 2:end));
    else
        gold = segmented(1:end-1, 1:end-1, 1:end-1) ~= 0;
    end
end