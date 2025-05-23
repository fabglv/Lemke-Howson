# Lemke-Howson
An implementation the Lemke-Howson algorithm, a method for finding a Nash equilibrium in two-player, non-degenerate games.

The Lemke-Howson algorithm and its implementation through integer pivoting are described in sections 3.4 and 3.5 of [1]. The modifications described in section 3.6 to handle degenerate games have not been implemented.

## File structure

```
lemke_howson.py           # Core implementation of the Lemke-Howson algorithm
demo.py                   # Script to demonstrate usage of the Lemke-Howson algorithm
```

## References
[1] Nisan N, Roughgarden T, Tardos E, Vazirani VV, eds. Algorithmic Game Theory. Cambridge University Press; 2007.
