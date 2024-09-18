
## TODO visualization

- [ ] Understand reasons behind loss spikes -> one is due to crossing at zero of vectors that get normalized.

#### Vivien's TODO

We have settle on a random seed, 
- show fine-grained visualization of test acc vs sweep axis
- get visualization, and analyze them manually

- Code Amelioration
    - [ ] Write subfunctions for the visualization to be able to pick the ones we want. Load a json config file for the visualization.

- Fine-tuning
    - [ ] write some fine-tuning experiments.
    - [ ] Write visualization when $p=3$ with triangles for the sequence associated to $y = 2$.

Clean this file.
Read Nanda, LLaMa3, MuP.

## Visualize memorizaton vs. algorithm
- In factorization.ipynb: 
    - visualize attention matrices along training with corresponding sequence
    - visuzalize "pruning" graph along iterations
    - visualize vector in $R^d$ (e.g., $d=32$) after attention (attn @ value @ x)
    - do the same after first layer and second layer of MLP

## Experience on a two layer transformers

*TODO:*
- generate data with alpha uniform + (1- alpha) deterministic, start from random state

We probably do not care about SGD much, LayerNorm probably change the picture a little bit.
Would be nice to check the influence of the weight initialization.

Experience 0: Verifier qu'on apprends le circuit
- Measure if the first attention is focused on the sub-diagonal.
- Measure if the second attention is focused on toekn that follows that token we are processing.

- Retrouver le induction head circuit
    - Check the attention matrix.
    - Set the code to evaluate the different score to measure what the three individual memories are learning.

Experience 1: Montrer l'apparition du circuit au cours du temps
- Montrer les connexions entre les differentes "register" (les $x_{t, l}$ pour different $t$ et different layer $l$) avec la largeur qui depend des poids de l'attention.
montrer comment ca evolue au cours du training.
    - Petite sequences du type 16 tokens. on montre les connextions due a la premiere et la deuxieme attention. ca veut dire qu'on a 16 + 16 + 1 noeuds, et 16 x 16 + 16 edges, et on fait grossir les edges en fonction des poids de l'attention.

Experience 2: Montrer le pruning de facon plus explicit
- Reflechir a comment illustrer le mechanisme que gradient descent essaye de pousser tous les circuits qui peuvent aider, et que cela cree du pruning.
Peut etre montrer en vers les connexion qui sont renforces de t a t+1 et en rouge celles qui sont diminues. Peut etre prendre un batch size petit pour que cela se voit bien.
Ou alors le montrer pour chaque donnees dans le batch puis montrer la moyenne.
- Essayer de voir si on peut expliquer les maths avec notre comprehension des mecanismes de memoires.
    - Meme truc mais on colorie en vert quand les poids grossissent et en rouge quand ils diminuent.

Experience 3: Etude de l'effet de differents parametres
- Si on comprends comment c'est implementer a l'interieur, on peut comprendre quand est-ce qu'on va etre a la limite de la capacite d'apprentissage.
- Montrer que le circuit change si on a "mal conditionne" les donnees par rapport aux parametres du transformer.

## Note regarding the study for a single layer

$N$: number of inputs
$M$: number of outputs $(M \leq N)$
$d$: dimension of the embedding

Case $N = M = 2$

On peut prendre $d=2$:
W est de dimension 4.
Mais dans le cas SGD, on s'interesse qu'a $(u_1 - u_2) e_i^\top$ pour $i \in \{1, 2\}$, ce qui permets de se ramener a un probleme de dimension 2, et on plot le landscape.

Je suis pas sur que ca soit aussi le cas pour Adam.

**Experience Zero:** Validation empirique que Adam + LN > Adam > LN > SGD pour differente valeure de N, M, d.

**Premiere experience:** verifier si les trajectoires pour Adam sont sur un plan ou non pour
    - LayerNorm -> Non
    - Adam with different beta -> Non
    - NormGD -> Non
Test planaire: W_t - W_s peut etre reconstruit Construire la base avec Gram-Schmidt (W_1 - W_2, W_{-2} - W_{-1}).

Autre precision: Layer norm ca s'assure que We_x reste sur la sphere, mais ca dit pas que W reste sur la sphere. C'est pas clair l'effet que ca a sur le landscape. Notons cependant que
\[
    We_x / \|We_x\| = (W / \|W\|) (e_x / \|e_x\|) / \| (W / \|W\|) (e_x / \|e_x\|) \|.
\]
Est-ce qu'on peut creer des classes d'equivalence tel que $C(W_0) := \{W | \forall\, x; We_x / \|We_x\| = W_0e_x\}$ (et qui sont pas les memes que $\{W | W / \|W\| = W_0})$?
Idee generale: avec layernorm on perd quelques degree de liberte.

% Autre question: est-ce qu'il existe un espace compact convexe (la sphere), qui intersecte tous les $C(W_0)$
Autre precision: Dans la classe d'equivalence, on a envie de dire qu'elle commute avec les gradient updates? $\exists W_1; \forall W \in C(W_0), W_{+1} = W - \eta \nabla L(W) \in C(W_1)$.
Mais c'est pas sur, donc pas sur qu'on puisse n'etudier la dynamique qu'a partir des classes d'equivalence.

**Deuxieme experience:** est-ce qu'avec LayerNorm, la norm de W divergence vers l'infini si on fait des pas constants de gradient updates quand $p(y|x) \in \{0, 1\}$ -> Pas clair, probable. (il faut comparer avec la divergence de SGD, car la divergence est tres lente, et peut etre dur a voir experimentalement).
[Est-ce qu'on visualiser le loss landscape?]
On peut commencer avec une seule particule (ou alors des particules decouplees ($e_x \perp e_{x'}$))

1. Etude de layernorm: Minimum, hidden convexity
Etude a faire sur le loss landscape quand on utilise layernorm: c'est plus convexe par rapport a W, mais toujours par rapport aux We_x. Est-ce que ca nous permet de dire des choses sur le nombre de minimum locaux,...
On peut regarder le landscape as a function of $p_W(y|x)$, sachant que notre model ne parametrise un sous-ensemble des probabilites $(p(y|x)) \in \Delta_M^N$ (c'est un peu comme un optimization sous contraintes).

**Troisieme experience:**
Si on veut faire des plot en 2D, on a naturellement deux dimensions:
- Une qui est la direction de convergence (l'accuracy ne depend pas de la norme de W). On peut regarder $f_1(t) = W_t^\top W_\infty / \|W_\infty\|$
- L'autre composante doit quantifier des mechanismes transitoires (peut etre oscillatoires), grosso un truc du type $f_2(t) = \|\Pi(W_t; W_\infty^\perp)\|$
On peut dire si f_1 est grand et f_2 est petit on a une guarantie (upper bound) sur la valeur de la loss de training (cross-entropy) et l'accuracy (0-1 loss). On peut aussi regarder les losses directement.


*Question experimentale*
Comment initialiser les poids? Est-ce que c'est important?

