function [ind_keep] = Denoise_jEAR_updated(x_event,y_event,T_event,tau,r,c)

%initialize
spatial = nan(r+2,c+2); %pad a couple extra rows and columns
ind_keep = false(1,length(T_event));

% Find the 1D indices corresponding to 2D location (x,y):
idx = sub2ind([r+2 c+2], y_event+1, x_event+1);
ind=([-r 0 r]+[-1 0 1]'); ind=ind(:);
indq = ind;
indq(5) = []; %dont use the same pixel

for i=1:length(T_event)
    
    if isnan(spatial(idx(i))) %no history - filter event but let next ones pass
        spatial(idx(i)+ind) = T_event(i);
        
    else %have a history
        neighbor = T_event(i) - spatial(idx(i)+indq);
        
        if sum(neighbor<=tau) >= 1
            spatial(idx(i)+ind) = T_event(i);
            ind_keep(i) = true;
        end
    end
    
end
