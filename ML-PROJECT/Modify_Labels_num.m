function finalLabel=Modify_Labels_num(X)

finalLabel=zeros(size(X,1),1);
X=X';
[maxval maxind]=max(X);
finalLabel=maxind';
end
