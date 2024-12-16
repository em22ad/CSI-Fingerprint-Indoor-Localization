function nbr_list=find_nbr(ad_ch_csi_col,x,y,res_x,res_y)
nbr_list=[];
if (x+res_x <= size(ad_ch_csi_col,1)) && (y+res_y <= size(ad_ch_csi_col,2)) 
    if (size(ad_ch_csi_col{x+res_x,y+res_y},1) > 0)
       nbr_list=[nbr_list;x+res_x y+res_y];
    end
end

if (x+res_x <= size(ad_ch_csi_col,1)) 
    if (size(ad_ch_csi_col{x+res_x,y},1) > 0)
       nbr_list=[nbr_list;x+res_x y];
    end
end

if (x+res_x <= size(ad_ch_csi_col,1)) && (y-res_y >= 1) 
    if (size(ad_ch_csi_col{x+res_x,y-res_y},1) > 0)
       nbr_list=[nbr_list;x+res_x y-res_y];
    end    
end

if (y+res_y <= size(ad_ch_csi_col,2)) 
    if (size(ad_ch_csi_col{x,y+res_y},1) > 0)
       nbr_list=[nbr_list;x y+res_y];
    end
end

if (y+res_y <= size(ad_ch_csi_col,2)) && (x-res_x >= 1) 
    if (size(ad_ch_csi_col{x-res_x,y+res_y},1) > 0)
       nbr_list=[nbr_list;x-res_x y+res_y];
    end    
end

if (y-res_y >= 1) 
    if (size(ad_ch_csi_col{x,y-res_y},1) > 0)
       nbr_list=[nbr_list;x y-res_y];
    end
end

if (x-res_x >= 1) 
    if (size(ad_ch_csi_col{x-res_x,y},1) > 0)
       nbr_list=[nbr_list;x-res_x y];
    end
end

if (x-res_x >= 1) && (y-res_y >= 1) 
    if (size(ad_ch_csi_col{x-res_x,y-res_y},1) > 0)
       nbr_list=[nbr_list;x-res_x y-res_y];
    end
end