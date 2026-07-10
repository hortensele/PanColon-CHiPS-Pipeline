# Demo slide

Drop a single public colon H&E whole-slide image here (e.g. a TCGA-COAD `.svs`)
to smoke-test the pipeline end to end. Then point the config at it:

```yaml
paths:
  wsi_dir: "examples/demo_slide"
  work_dir: "examples/demo_slide/work"
```

and run:

```bash
python pancolon_pipeline.py all --config config/pipeline.yaml
```

Expected outputs after a successful run (under `work_dir/`):

- `hpl/…_filtered.h5` with `img_z_latent` embeddings
- `datasets/<cohort>/HPL_PANCOLON_20x/pt_files/*.pt`
- `survclam/chips_scores.csv` — the CHiPS score for the demo slide
- `attention/chips_overlays.png` — the overlay figure

WSIs are gitignored; nothing in this folder except this README is tracked.
