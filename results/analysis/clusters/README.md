# FF++ source-target component structure

Closes **G4**: the 150-component figure was originally computed in an ad-hoc shell
pipeline and is now script-generated.

`component_stats.csv`: 300 sequences, 300 unique targets, 300 unique sources,
**150 components, all of size 2, zero singletons**, `video_groups_per_component =
2.0`. The subset is partner-closed — every source is also a target.

That last column matters: it is why a 30-target test split yields 28–29 components
rather than 15 (partners co-occur only by chance under video-level splitting), and
why component-disjoint splitting (L3) would be a substantive change rather than a
formality.

⚠️ These are **source-target components, not verified human identities**. FF++
sequence ids are not identity labels.

```
python -m src.cluster_stats_report --listing listing.txt --out-dir results/analysis/clusters
```
