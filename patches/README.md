# Local Nerfstudio changes

`nerfstudio-local.patch` preserves the local Nerfstudio modifications present when
project data was offloaded on November 10, 2026. The dependency is an ignored,
independent checkout, so these edits are saved here with the project source.

Upstream: https://github.com/nerfstudio-project/nerfstudio.git

Base commit: `50e0e3c70c775e89333256213363badbf074f29d`

Apply to a clean checkout at that commit:

```bash
git -C nerfstudio checkout 50e0e3c70c775e89333256213363badbf074f29d
git -C nerfstudio apply ../patches/nerfstudio-local.patch
```

The existing dependency checkout already has this change; do not apply it twice.
