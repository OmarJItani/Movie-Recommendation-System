# Movie Recommendation System

## What is in this repository

In this repository, I present my implementation of a movie recommendation system using the matrix completion approach. I first present how I modeled the system as an optimization problem. Then I present how I solve the optimization problem using a stochastic gradient descent. Afterwards, I implement the system in Matlab and present the results.

## Mathematical Modelling: Matrix Completion

Collaborative filtering makes predictions by
looking at user-item relations. In simpler words, if the system identifies two similar users, then
items preferred by the first user are recommended to the second user and vice versa.

In a typical collaborative filtering setting, the recommendation system can be modeled as a
matrix completion problem. Given a list of $m$ users $(u_1,u_2,\dots,u_m)$ and $n$ items $(i_1,i_2,\dots,i_n)$, the ratings/preferences of users toward the items can be represented as an incomplete $m \times n$
matrix $M$, where $(i, j)-$ th entry either represents the rating of user $i$ to movie $j$ or is missing.

Let $\Omega \triangleq${ $(i, j): M_{i,j}$ is observed } be the set of observed (non-missing) entries. Our goal is to recover a low-rank matrix using the observed data $\Omega$. More specifically, we would like to solve the following optimization problem:

$\underset{X \in R^{m \times n}}{\min }{rank(X)} ~~~ s.t. ~~~ X_{i,j}=M_{i,j}$ for all $(i,j) \in \Omega$.

The rank of the matrix is the number of non-zero eigenvalues. Hence, we need to recover a
matrix with the lowest possible rank that matches the observed rankings in $\Omega$. Unfortunately,
this problem is extremely hard to solve. Substituting the rank (number of non-zero eigenvalues)
with the nuclear norm yields the following convex optimization problem

$` \underset{X \in R^{m \times n}}{\min }{||X||_*} \text{ s.t. } X_{i,j} = M_{i,j} \text{ for all } (i,j) \in \Omega `$,

which is more tractable. To allow for some flexibility in the constraint, a more common formulation of the problem is as follows

$` \underset{X \in R^{m \times n}}{\min }{||X||_*} +\lambda \underset{(i,j) \in \Omega}{\sum}{(X_{i,j}-M_{i,j})^2} `$.

Despite convexity, our formulated problem can have an extremely high dimension (m×n). Moreover, estimating the nuclear norm requires computing the singular value decomposition at every
iteration which is also computationally expensive.

More recently, matrix factorization method has shown to solve many of the discussed issues. This methods forces a low rank matrix by expressing $X$ as a product of two matrices $U \in R^{m \times r}$
and $V \in R^{n \times r}$ , i.e.

$X = UV^T$.

This forces the rank of $X$ to be at most $r$ and yields the following optimization problem

$` \underset{U \in R^{m \times r} , V \in R^{n \times r}}{\min }{ \underset{(i,j) \in \Omega}{\sum}{( u_i{v_j}^T -M_{i,j})^2}} `$,

where $u_i$ is the $i^{th}$ row of $U$ and $v_j$ corresponds the $j^{th}$ row of $V$. This problem aims at
recovering a matrix with rank at most $r$ having similar observed entries. To prevent overfitting
a regularization term is introduced to the squared error which yields

$` \underset{U \in R^{m \times r} , V \in R^{n \times r}}{\min }{ \underset{(i,j) \in \Omega}{\sum}{( u_i{v_j}^T -M_{i,j})^2 + \lambda (||u_i||^2 + ||v_j||^2)}} `$.

Note that the number of variables in the problem above is $(m + n)r$ which is much less than the dimension of the original problem $mn$ when $r << m, n$. In our problem, each row $u_i$ represents a vector of features for user $i$ and each row $v_i$ represents a vector of features for item $i$. Hence $u_i v_j^T$ is our prediction of the rating of user $i$ to item $j$.


## Optimization Algorithm: Stochastic Gradient Descent

Different optimization algorithms can be applied to solve the optimization problem at hand. Due to the form of the optimization problem, that is, since the given cost function is a summation of several sub-cost functions, one can apply the Stochastic Gradient Descent algorithm (SGD), in which at each iteration, the SGD randomly chooses a sub-cost function to minimize by moving on its negative gradient direction with a step size $\alpha$. In our work, an SGD with constant step size $\alpha$ is to be used. The following Pseudo-code explains the steps of the algorithm to be developed.

Algorithm:
- Step 1: Initialize $U$ and $V$ randomly with a uniform distribution over $[0,1]$.
- Step 2: Generate a random index $stoch_{ind}$ over $[1, size(dataset)]$ with a uniform distribution.
- Step 3: Compute the $i$ and $j$ associated with the selected data point.
- Step 4: Update $u_i$ and $v_j$ according to the following update rule:

    $` u_i = u_i - \alpha [(u_iv_j^T - M_{i,j})v_j + \lambda u_i] `$

    $` v_j = v_j - \alpha [(u_iv_j^T - M_{i,j})u_i + \lambda v_j] `$

- Repeat Steps 2-4 for 𝑇 iterations

## Results

To be added.
