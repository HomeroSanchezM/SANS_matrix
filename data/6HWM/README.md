# 6HWM — protéine cible de l'analyse matricielle SANS

## Fichiers

| Fichier | Rôle | Taille | md5 |
|---|---|---|---|
| `6HWM_H.pdb` | **Entrée du pipeline matrice** — protonée, tous les H explicites | 1,7 Mo | `69a5ab3f745b878bdbfaef81715686a9` |
| `6HWM_original.pdb` | Structure déposée RCSB, telle quelle (sans H) | 1,7 Mo | `4f9cd7467bfc9ee58463d0d47d06e4aa` |
| `6HWM.cif` | Même structure au format mmCIF (provenance) | 1,9 Mo | `862f63b2ce34098c9e7954a6cef91a17` |

## Contenu de `6HWM_H.pdb` (vérifié)

- 20 664 atomes `ATOM` dont **10 452 hydrogènes explicites**
- 7 chaînes (A→G), ~185 résidus chacune, **uniquement les 20 acides aminés standards**
  (aucune sélénométhionine ni résidu modifié → rien à patcher)
- 422 atomes `HETATM` : `BO2` (371, borate) et `PEG` (51) — sans hydrogène, ignorés par
  la deutération (ils ne figurent dans aucun groupe d'AA) et tolérés par Pepsi-SANS
- pas de `MODEL`/`ENDMDL` (mono-modèle) → une seule copie lue par `gemmi`

## Provenance

Dérivé de `Documents/Phd/OptiSANS/data/` :

```
data/6hwm/6HWM.cif                 → 6HWM.cif (copie)
data/6hwm/6HWM.pdb                 → 6HWM_original.pdb (copie)
data/6HWM_H.pdb                    → 6HWM_H.pdb (copie)   ← protoné, prêt à l'emploi
```

Les intermédiaires de nettoyage/ajout d'hydrogènes (`6HWM_cleaned.pdb`,
`6HWM_cleaned_patched.pdb`, `6HWM_from_cif.pdb`, `work_addh/`, `work_pdbfixer/`) sont
restés dans `Documents/Phd/OptiSANS/data/6hwm/` et peuvent être régénérés avec
`OptiSANS/data/clean_pdb.py`, `clean_with_pdbfixer.py` et `add_h.sh`.

**Prérequis du pipeline** : l'entrée doit avoir tous les H explicites et protonés
(c'est le cas de `6HWM_H.pdb`) ; le nom de fichier est libre, il est passé via `--pdb`.

## Utilisation

Étape 1 — matrice brute (à lancer depuis la racine de ce dépôt, cf. `pixi.toml`) :

```bash
pixi run matrix --pdb data/6HWM/6HWM_H.pdb \
  --mode complete --d2o-step 5 \
  --output 6HWM_18_aa_21_d2o.h5 \
  --threads 16 \
  --pepsi /home/homero/Documents/Phd/OptiSANS/Pepsi-SANS-Linux/Pepsi-SANS
```

Ajouter `--test-mode` pour un premier essai sur **un seul** patron (fichiers temporaires
conservés) : indispensable avant les 5 505 024 simulations du run complet.

Ordre de grandeur mesuré sur cette machine (16 cœurs, Pepsi-SANS 0,27 s par courbe sur
cette structure) : ~26 h de CPU parallèle, plus la partie série (écriture de 21 PDB par
patron, soit ~9,4 To cumulés) → compter 1,5 à 3 jours.

Suite du pipeline : `bspline_fit.py` (n-basis **18** obligatoire) → `spline_clustering.py`
(UMAP + HDBSCAN, GPU requis) → `compute_cluster_average.py`.
