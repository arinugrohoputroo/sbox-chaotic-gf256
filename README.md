# S-box Chaotic Map GF(2^8)

Konstruksi **S-box 8×8** di **GF(2^8)** menggunakan **Logistic Map** dengan pengujian kriptografi: NL, SAC, BIC-NL, BIC-SAC, LAP, dan DAP.

## Pipeline

1. **Logistic Map** — `x_{n+1} = r · x_n · (1 - x_n)` menghasilkan urutan kaotik
2. **Permutasi bijektif** — rank-order sorting dari 256 nilai kaotik
3. **Multiplicative inverse** — operasi di GF(2^8) dengan polinom irreducibel AES `0x11B`
4. **Affine transform** — `S(x) = M·x ⊕ c` di GF(2)

## Struktur

```
src/
  gf256.py          # Aritmetika GF(2^8)
  chaotic.py        # Logistic Map
  sbox_builder.py   # Pipeline konstruksi S-box
  metrics.py        # NL, SAC, BIC-NL, BIC-SAC, LAP, DAP
app/
  streamlit_app.py  # Demo interaktif
tune_parameters.py  # Grid search parameter r dan x0
```

## Metrik Kriptografi

| Metrik | Deskripsi |
|--------|-----------|
| NL | Nonlinearity (Walsh transform) |
| SAC | Strict Avalanche Criterion |
| BIC-NL | Bit independence — nonlinearity |
| BIC-SAC | Bit independence — SAC |
| LAP | Linear Approximation Probability |
| DAP | Differential Approximation Probability |

Parameter optimal hasil grid search: **r = 4.0**, **x0 = 0.5** — 10/10 metrik setara AES S-box.
