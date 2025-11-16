## VISTA

### Temporary Setup
Create a symbolic link from the `base_pipeline.py`, `splatfacto.py`,  and `trainer.py` 
files in your Nerfstudio Python Package to the `base_pipeline.py`, `splatfacto.py`,
and `trainer.py` files in `vista/utils/ns_utils`. For example:
```bash
cd <Python package directory/base_pipeline.py>
cp <base_pipeline.py> <base_pipeline_original.py>
ln -s <GitHub repo path/base_pipeline.py> <base_pipeline.py>
```