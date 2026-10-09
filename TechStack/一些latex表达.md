花体：$\mathcal{a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z}$
```latex
\mathcal{}
```

黑板加粗体：$\mathbb{a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z}$
```latex
\mathbb{}
```

大于等于号，小于等于号：$\geq,\leq$，g是greater，l是less

```
\geq,\leq
```

加^号 输入\hat  或 \widehat

加横线 输入 \overline

加波浪线 输入 \widetilde

不等号：$\neq$

```latex
\neq
```

任意和存在：$\forall,\exist$

```latex
\forall,\exist
```

偏导数和梯度：$\partial,\nabla$

```latex
\partial,\nabla
```



在使用LaTex进行排版时，一个常见的需求是要把下标放在某个文字或者符号的正下方：



LaTex的数学模式下提供了\limits命令，形如

```expr1\limits_{expr2}^{expr3}latex
expr1\limits_{expr2}^{expr3}
```

中expr2会出现在expr1的正下方，而expr3会出现在expr1的正上方，例如命令

```latex
$\sum\limits_{i=0}^n {x_i}$
```

$\sum\limits_{i=0}^n {x_i}$
会生成效果

大括号括起来多行公式的方式，lrc分别表示该列内容居左、居中、居右：
```
\begin{equation}
\left\{
             \begin{array}{lr}
             x=\dfrac{3\pi}{2}(1+2t)\cos(\dfrac{3\pi}{2}(1+2t)), &  \\
             y=s, & 0\leq s\leq L,|t|\leq1.\\
             z=\dfrac{3\pi}{2}(1+2t)\sin(\dfrac{3\pi}{2}(1+2t)), &  
             \end{array}
\right.
\end{equation}
```
效果：
$$
\begin{equation}
\left\{
             \begin{array}{lr}
             x=\dfrac{3\pi}{2}(1+2t)\cos(\dfrac{3\pi}{2}(1+2t)), &  \\
             y=s, & 0\leq s\leq L,|t|\leq1.\\
             z=\dfrac{3\pi}{2}(1+2t)\sin(\dfrac{3\pi}{2}(1+2t)), &  
             \end{array}
\right.
\end{equation}
$$


在写论文的过程中，编辑公式部分会有一些希腊字母需要用LaTeX形式，将其汇总如下，方便查阅。

| 希腊字母小写 | 大写  | LaTeX形式                |
| ------ | --- | ---------------------- |
| α      | A   | `\alpha`               |
| β      | B   | `\beta`                |
| γ      | Γ   | `\gamma`               |
| δ      | Δ   | `\delta`               |
| ϵ      | ε   | `\epsilon`             |
| ζ      | Z   | `\zeta`                |
| η      | H   | `\eta`                 |
| θ      | ϑ   | `\theta` / `\vartheta` |
| υ      | Υ   | `\upsilon`             |
| ω      | Ω   | `\omega`               |
| ϕ      | φ   | `\phi` / `\varphi`     |
| κ      | K   | `\kappa`               |
| λ      | Λ   | `\lambda`              |
| μ      | M   | `\mu`                  |
| ξ      | Ξ   | `\xi`                  |
| π      | Π   | `\pi`                  |
| ρ      | ϱ   | `\rho` / `\varrho`     |
| σ      | Σ   | `\sigma`               |
| τ      | T   | `\tau`                 |
| χ      | X   | `\chi`                 |
| ι      |     | `\iota`                |

