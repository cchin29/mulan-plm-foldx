# Vendored FoldX energies

A byte-for-byte copy of [skempi-foldx](https://github.com/cchin29/skempi-foldx) at its **v0.1.0**
tag. It is pinned there deliberately: every number in this repository was produced against v0.1.0,
and skempi-foldx 0.2.0 rebuilt the store — 1850 of 6003 values differ. Adopting the newer store
means re-running what was scored against these energies, not swapping files.

## Terms

**These files do not carry this repository's CC BY-NC-SA licence.** They are skempi-foldx's
artifact and keep its terms:

* the **code** that produced them is MIT;
* the **data** is an adaptation of **SKEMPI 2.0**, which is CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/) — each record carries SKEMPI's verbatim
  `cleaned` mutation string. The per-complex mutation *lists* are SKEMPI's for 1,744 of the 4,238
  single-point records and the **S4169** selection for the other 2,494 — each record names its
  origin in `_source`, and the distinction matters because a value depends on its list (see
  `../../OPEN_QUESTIONS.md`). S4169 is mCSM-PPI2's selection, obtained here from GeoPPI's
  verbatim copy; both are credited in `../../NOTICE`;
* the energies were computed with **FoldX**, licensed software from the CRG, which is not
  included, wrapped or redistributed here or upstream.

A CC BY 4.0 grant cannot be narrowed by copying the files into a non-commercial repository, so
non-commercial terms do **not** attach to them. Read skempi-foldx's own `NOTICE` before
redistributing, and cite SKEMPI:

> Jankauskaite, J., Jimenez-Garcia, B., Dapkunas, J., Fernandez-Recio, J. & Moal, I. H.
> "SKEMPI 2.0: an updated benchmark of changes in protein-protein binding energy, kinetics and
> thermodynamics upon mutation." *Bioinformatics* 35(3):462-469, 2019.
> [10.1093/bioinformatics/bty635](https://doi.org/10.1093/bioinformatics/bty635)

## Known defects

Two defects travel with the pin and are **fixed upstream in 0.2.0**, which is a rebuilt store
rather than a patch. Both are documented in [`../README.md`](../README.md#known-defects-in-the-vendored-foldx-store):
two SKEMPI rows malformed upstream, and 31 records scored against a collapsed interface.
