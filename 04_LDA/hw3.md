---
fontfamily: "libertine"
mainfont: "GFS Artemisa"
title: "Homework #3"
author: [Tae Geun Kim]
date: 2020-05-19
subject: "Markdown"
keywords: [Markdown, Example]
subtitle: "Linear Discriminant"
titlepage: true
toc-own-page: true
header-includes:
    - \usepackage{xcolor}
...

# Least Square

Implement Least square for Linear discriminant as follows.

1. Generate two groups of 2D random data. (Each group has 150 samples)
    * $(x_1,y_1) \sim (\mathcal{N}(3, 1^2),\,\mathcal{N}(1, 3^2))$

    * $(x_2,y_2) \sim (\mathcal{N}(-3, 1^2), \, \mathcal{N}(-1, 3^2))$

2. Use Least square, find $\widetilde{\mathbf{W}}$.

3. Plot data & decision boundary

4. Add some outliers (10 samples) - $(x_3, y_3) \sim (\mathcal{N}(5, 1^2), \mathcal{N}(3, 1^2))$ and repeat 2, 3.

# Fisher's LDA

Use Fisher's linear discriminant, repeat above.

**Hint**: We know $\mathbf{w} \propto \mathbf{S_w^{-1}}(\mathbf{m_2} - \mathbf{m_1})$. With normalization, we can find true $\mathbf{w}$.

**Helpful reference**

* Bishop Chap 4
* [\textcolor{blue}{https://adnoctum.tistory.com/442}](https://adnoctum.tistory.com/442)


# RANSAC

RANSAC menas Random Sample Consensus. ([\textcolor{blue}{Wiki}](https://en.wikipedia.org/wiki/Random_sample_consensus))  
It is much more robust than Least square.

Implement RANSAC and apply to parabola problem in [\textcolor{blue}{https://darkpgmr.tistory.com/61}](https://darkpgmr.tistory.com/61).
