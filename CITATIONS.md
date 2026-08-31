# Required citations

Dataset licences oblige us to cite the following in any publication or report.

## FaceForensics++ (primary training data)

> A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, M. Nießner.
> **FaceForensics++: Learning to Detect Manipulated Facial Images.** ICCV 2019.

```bibtex
@inproceedings{roessler2019faceforensicspp,
  author    = {Andreas R\"ossler and Davide Cozzolino and Luisa Verdoliva and
               Christian Riess and Justus Thies and Matthias Nie{\ss}ner},
  title     = {Face{F}orensics++: Learning to Detect Manipulated Facial Images},
  booktitle = {International Conference on Computer Vision (ICCV)},
  year      = {2019}
}
```

Cite additionally **only if used**:

- Original FaceForensics — Rössler et al., *FaceForensics: A Large-scale Video Dataset
  for Forgery Detection in Human Faces*, arXiv 2018.
- Google / JigSaw DeepFakes Detection Dataset (bundled with the FF++ download) —
  Dufour et al., 2019.

## Celeb-DF v2 (cross-dataset evaluation)

> Y. Li, X. Yang, P. Sun, H. Qi, S. Lyu.
> **Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics.** CVPR 2020.

```bibtex
@inproceedings{Celeb_DF_cvpr20,
  author    = {Yuezun Li and Xin Yang and Pu Sun and Honggang Qi and Siwei Lyu},
  title     = {Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020}
}
```

## Licence terms (summary — see the datasets' own agreements for the full text)

Both datasets: **non-commercial research/education only**, **no public redistribution
of the videos or any derived data (including extracted face crops)**, access may be
revoked by the providers at any time. Our extracted crops are kept in a **private**
store and are never committed to this repository (`.gitignore` blocks image files).
