% This function takes the adjacency matrix of an undirected graph and
% returns a CONNECTED dominating set computed.
% Reference: Guha and Khuller, "Approximation Algorithms for Connected Dominating Sets", Algorithmica, 1998
%-------------------------------------------------------------------------
function Dom = Conn_Dom_Guha_Kehler(A)

N = size(A,1);
d = sum(A,2);
max_degree = max(d);

W = ones(N,1);       % Vertices that are not yet dominated
G = zeros(N,1);      % Vertices that are already dominated
B = zeros(N,1);      % Vertices in the Dominating set
e = zeros(N,1);

%------ Making initial node (one with the max degree) Black ---------
Z = find(d == max(d));
z = Z(1);
B(z) = 1;
temp = find(A(:,z)==1);
for i = 1:length(temp)
    G(temp(i))=1;
end
%-------------------------------------
BG = or(B,G);
W = ~BG;
check = sum(BG);
while (check < N)

%--------------------------------------------------------------------------
% Below 'for' loop determines, out of all the vertices, the vertex that has maximum number of
% non-dominating vertices in its closed neighborhood.
    
    for k=1:N
     if (G(k)==1)
        s = 0;
    
        for l = 1:N
        
            if (A(k,l)==1)&&(W(l)==1)
                s = s+1;
            end
        
        end
    
        e(k) = s+W(k);
     end
    end

b = find(e==max(e));   % b array contains the indices with maximum number 
                       % of non-dominating vertices in their closed
                       % neighborhoods.
% ------------------------------------------------------------------------
B(b(1))=1;             % Including b(1) into the dominating set.

W(b(1))=0;

%--------------------------------------------------------------------------
% Below, we update the status of vertices in the closed neighborhood of
% b(1), as dominated ones.

temp = find(A(:,b(1))==1);
for i = 1:length(temp)
    G(temp(i))=1;
end

BG = or(B,G);
W = ~BG;
% -------------------------------------------------------------------------

check = sum(BG);   % Terminating condition, (All vertices are dominated).

end

Dom = find(B==1);  % Set of Dominating vertices.    
%Range = [ceil((length(Dom)/log(max_degree))), length(Dom)];
Dom = Dom';

