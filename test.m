function blocks=slide_window(path,block_szx,block_szy,jump_r, jump_c)
img=imread(path);
[rows,cols]=size(img);
%ri = imresize(img,[row row]);
%block_sz = 200;  % assumed square
%jump_r=200;
%jump_c=200;
%imageSize = row; % assumed square
blocks = cell(rows - block_szx+1,cols - block_szy+1);
%Now iterate over each 31x31 block as follows
 for r=1:jump_r:size(blocks,1)
     for c=1:jump_c:size(blocks,2)
         blocks{r,c} = ri(r:r - 1 + block_szx, ...
                                   c:c - 1 + block_szy);
         imshow(blocks{r,c})
         pause(5);
     end
 end