## **Contextualization of TDA**

Topological data analysis (TDA) is an approach to the analysis of datasets using techniques from topology that is insensitive to the particular metric chosen. TDA general framework also provides techniques for dimensionality reduction and robustness to noise. It provides a concrete link between algebraic topology, metric geometry, and geometric topology by translating metric information from data into algebraic invariants that capture its topological structure.

- Algebraic topology is a branch of mathematics that uses tools from abstract algebra to study topological spaces. The basic goal is to find algebraic invariants that classify topological spaces up to homeomorphism, though usually most classify up to homotopy equivalence.

- Metric geometry is a branch of geometry with metric spaces as the main object of study. It is applied mostly to Riemannian geometry and group theory.

- Geometric topology is the study of manifolds and maps between them, particularly embeddings of one manifold into another.

## **Persistent Homology**

The main tool of TDA is persistent homology, an adaptation of homology to point cloud data.

TDA is premised on the idea that the shape of data sets contains relevant information. Real high-dimensional data is typically inherently sparse (independently of sampling effects), and tends to have relevant low dimensional features. One task of TDA is to provide a precise characterization of this fact.

Many algorithms for data analysis, including those used in TDA, require setting various parameters. Without prior domain knowledge, the correct collection of parameters for a data set is difficult to choose.

> \[!NOTE]
>In the context of TDA (specifically *persistent homology*) **the parameter** refers to the **scale parameter** (or *filtration value*) used to build a family of topological spaces.

The main insight of persistent homology is to use the information obtained from all parameter values by encoding this huge amount of information into an understandable and easy-to-represent form. With TDA, there is a mathematical interpretation when the information is a homology group. In general, the assumption is that features that persist for a wide range of parameters are "true" features. Features persisting for only a narrow range of parameters are presumed to be noise, although the theoretical justification for this is unclear.

### **What does Persistent Homology provide?**

**At the topological level:**

Persistent Homology captures the informally called features, concretelly, these are **topological invariants describing the shape of the data**, each corresponding to the presence of a hole whose **boundary has a specific intrinsic dimension**. In other words:

> A *feature* in persistent homology is a topological hole whose intrinsic dimensionality equals the dimension of the manifold that bounds it (regardless of the ambient dimension where the data lives).

In algebraic topology:

- A 0-dimensional feature is a connected component (a set of points distributed around one centroid: dimension 0)
- A 1-dimensional feature is a hole bounded by a loop (a set of points distributed around a closed curve: dimension 1)
- A 2-dimensional feature is a void (a bubble) (a set of points distributed around a hollow sphere-like cavity: bounded by a 2D surface)

Finding these topological features within the data set requires building candidate point agrupations and "testing" their fitness. Concretely, this agrupations are taken as the sublevel family of complexes obtained by thickening the data at scale $r$. In plain words, merging into the same agrupation pairs of points with a nonempty intersection of the $r$-balls built around them.

> The scale ($r$) is the radius of the balls artificially placed around each data point.

The *persistent* qualifiyer in *Persistent Homology* means that it does not study topology at a single scale; instead it studies how the topology of the data changes as we the scale is continuously varied, and it keeps only the features that survive for a long range of scales.

> The *scale* controls the spatial resolution at which the data is probed by thickening the points.

It's exactly like zooming in and out:

- At very fine resolution $\to$ lots of details, noise
- At very coarse resolution $\to$ only the big structure remains

**At the practical level:**

This features are encoded into visualy representable mathematical artifacts. Mainly persistence barcodes and persistence diagrams.

- **persistence barcode**: multiset of intervals in $\mathbb{R}$

- **persistence diagram**: multiset of points in $\Delta := {(u,v) \in \mathbb{R}^2 \mid u,v \ge 0,\ u \le v}$.

## **Relevant Simplicial Complexes**

**Simplicial Complex**

In mathematics, a simplicial complex is a structured set of simplices (for example, points, line segments, triangles, and their n-dimensional counterparts) such that all the faces and intersections of the elements are also included in the set (see illustration).

[Wikipedia Image]

**Vietoris-Rips Complex**

In topology, the **Vietoris–Rips complex**, also called the **Vietoris complex** or **Rips complex**, is a way of forming a topological space from distances in a set of points. It is an abstract simplicial complex that can be defined from any metric space $M$ and distance $\delta$ by forming a simplex for every finite set of points that has diameter at most $\delta$. That is, it is a family of finite subsets of $M$, in which we think of a subset of $k$ points as forming a $(k-1)$-dimensional simplex (an edge for two points, a triangle for three points, a tetrahedron for four points, etc.). If a finite set $S$ has the property that the distance between every pair of points in $S$ is at most $\delta$, then we include $S$ as a simplex in the complex.

**Čech Complex**

In algebraic topology and topological data analysis, the Čech complex is an abstract simplicial complex constructed from a point cloud in any metric space which is meant to capture topological information about the point cloud or the distribution it is drawn from. Given a finite point cloud $X$ and an $\varepsilon > 0$, we construct the Čech complex $\check{C}_\varepsilon(X)$ as follows: Take the elements of $X$ as the vertex set of $\check{C}*\varepsilon(X)$. Then, for each $\sigma \subset X$, let $\sigma \in \check{C}*\varepsilon(X)$ if the set of $\varepsilon$-balls centered at points of $\sigma$ has a nonempty intersection. In other words, the Čech complex is the nerve of the set of $\varepsilon$-balls centered at points of $X$. By the nerve lemma, the Čech complex is homotopy equivalent to the union of the balls, also known as the offset filtration.

[Wikipedia Image]

The Čech complex is a subcomplex of the Vietoris-Rips complex. While the Čech complex is more computationally expensive than the Vietoris–Rips complex, since we must check for higher order intersections of the balls in the complex, the nerve theorem provides a guarantee that the Čech complex is homotopy equivalent to union of the balls in the complex. The Vietoris-Rips complex may not be.

---

## **Instrumental Constructs**

Some widely used concepts are introduced below. Note that some definitions may vary from author to author.

A **point cloud** is often defined as a finite set of points in some Euclidean space, but may also be taken to be any finite metric space.

The **Čech complex** of a point cloud is the nerve of the cover of balls of a fixed radius around each point in the cloud.

A **persistence module** $\mathbb{U}$ indexed by $\mathbb{Z}$ is a vector space $U_t$ for each $t \in \mathbb{Z}$, and a linear map
$u_t^s : U_s \to U_t$ whenever $s \le t$, such that:

- $u_t^t = 1$ for all $t$, and
- $u_t^s, u_s^r = u_t^r$ whenever $r \le s \le t$.

An equivalent definition is a functor from $\mathbb{Z}$, considered as a partially ordered set, to the category of vector spaces.

The **persistent homology group** $PH$ of a point cloud is the persistence module defined as
$PH_k(X) = \prod H_k(X_r)$,
where $X_r$ is the Čech complex of radius $r$ of the point cloud $X$ and $H_k$ is the $k$-th homology group.

A **persistence barcode** is a multiset of intervals in $\mathbb{R}$, and a **persistence diagram** is a multiset of points in
$\Delta := {(u,v) \in \mathbb{R}^2 \mid u,v \ge 0,\ u \le v}$.

The **Wasserstein distance** between two persistence diagrams $X$ and $Y$ is defined as:

$$
W_p[L_q](X,Y)
:= \inf_{\varphi : X \to Y}
\left[
\sum_{x \in X}
(\lVert x - \varphi(x) \rVert_q)^p
\right]^{1/p}
$$

where $1 \le p, q \le \infty$ and $\varphi$ ranges over bijections between $X$ and $Y$.
See Figure 3.1 in Munch [14] for an illustration.

The **bottleneck distance** between $X$ and $Y$ is:

$$
W_\infty[L_q](X,Y)
:= \inf_{\varphi : X \to Y}
\sup_{x \in X}
\lVert x - \varphi(x) \rVert_q.
$$

This is a special case of the Wasserstein distance obtained by setting $p = \infty$.

---

## **Instrumental Properties**

**Structure theorem**

The first classification theorem for persistent homology appeared in 1994 [11] via Barannikov's canonical forms. The classification theorem interpreting persistence in the language of commutative algebra appeared in 2005 [10]:

For a finitely generated persistence module $C$ with field $F$ coefficients,

$$
H(C;F)
\simeq
\bigoplus_i x^{t_i}\cdot F[x]
;\oplus;
\left(
\bigoplus_j
x^{r_j}\cdot \left(
F[x]/(x^{s_j}\cdot F[x])
\right)
\right).
$$

Intuitively:

- The **free parts** correspond to the homology generators that appear at filtration level $t_i$ and **never disappear**.
- The **torsion parts** correspond to those that appear at filtration level $r_j$ and persist for $s_j$ steps of the filtration (equivalently, they disappear at filtration level $s_j + r_j$). [11]

Persistent homology is visualized through a **barcode** or **persistence diagram**. Barcodes have roots in abstract mathematics: the category of finite filtered complexes over a field is **semi-simple**, meaning any filtered complex is isomorphic to its **canonical form**, a direct sum of one- and two-dimensional simple filtered complexes.

**Stability**

Stability is desirable because it provides robustness against noise. If $X$ is any space homeomorphic to a simplicial complex, and $f,g : X \to \mathbb{R}$ are continuous tame [15] functions, then the persistence vector spaces

- ${H_k(f^{-1}([0,r]))}$
- ${H_k(g^{-1}([0,r]))}$

are finitely presented, and

$$
W_\infty(D(f), D(g))
\le
\lVert f - g \rVert_\infty,
$$

where:

- $W_\infty$ denotes the **bottleneck distance** [16],
- $D$ is the map taking a continuous tame function to the persistence diagram of its $k$-th homology.

---
