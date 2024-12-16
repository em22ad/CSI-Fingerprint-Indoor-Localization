function fp=compact_fp(fp)
sz_x=size(fp,1);
sz_y=size(fp,2);
%for ch=1:size(fp,3)
ch=1;
idxs=[];
for x=1:sz_x
    dims=size(find(fp(x,:,ch) == -1),2);
    if (dims == sz_y)
        idxs=[idxs x];
    end
end
fp(idxs(:),:,:)=[];

sz_x=size(fp,1);
sz_y=size(fp,2);
idxs=[];
for y=1:sz_y
    dims=size(find(fp(:,y,ch) == -1),1);
    if (dims == sz_x)
        idxs=[idxs y];
    end
end
fp(:,idxs,:)=[];
end