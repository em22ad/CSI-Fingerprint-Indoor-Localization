function fp_csi=interp_missing_pixels(fp_csi)
for ch=1:size(fp_csi,3)
    clear X;
    M=fp_csi(:,:,ch);
    %M = [fliplr(M(:,floor(size(M,2)/2)+2:end)) M(:,floor(size(M,2)/2)+1:end)];
    [X(:,1),X(:,2)] = find(M~=-1);
    V=zeros(size(X,1),1);
    for i=1:size(X(:,1),1)
        V(i)=M(X(i,1),X(i,2));
    end
    F = scatteredInterpolant(X,V);
    F.Method = 'natural';
    O=zeros(size(M));
    for i=1:size(M,1)
        for j=1:size(M,2)
            O(i,j)=F([i j]);
        end
    end
    fp_csi(:,:,ch)=O;
end