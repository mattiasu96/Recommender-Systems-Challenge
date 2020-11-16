Repo for challenge
**TODO:** Implement  p3Alpha recommender with ItemContentBased using the merging Similarity matrix (use Maurizio Hybrid model, it's already there)

**Useful:** This commit (https://github.com/nicolo-felicioni/recsys-polimi-2019/commit/2e71e99abe58cc8d99f3b5bcb51b05f0b7680f65) gives some useful hints on how to build an hybrid recommender and manage the various classes ecc... 

Score -> Matrice dei risultati. Ottenuta facendo il prodotto dell'user profile per la similarity matrix (W_matrix)
W_Matrix -> similarity matrix.

Nel hybrid del prof lui fa il mix delle similarity matrix, infatti poi W_matrix è la somma pesata delle due similarity.
Nel recommender dei vincitori dell'anno scorso fanno (invece) la somma pesata delle score matrix! 
