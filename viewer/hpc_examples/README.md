Drop one representative tile image per HPC here, named to match the `id`
column in `../hpc_atlas.csv` (case-insensitive), e.g.:

    HPC0.jpg
    HPC1.png
    HPC17.jpeg

Supported extensions: jpg, jpeg, png, webp. A cluster with no matching file
here just won't show a thumbnail in the atlas — everything else still works.

`pancolon_pipeline.py export` picks these up automatically and copies the
matched ones into the results bundle.
