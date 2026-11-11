# Data storage and restore

Data archive preparation started on November 10, 2026 to recover space on the root filesystem. Source files remain in Git. Tracked data files are retained locally and excluded from the archives.

## Locations

- Google Drive destination: `stanford-gdrive:root-offload-20261110-three-projects/shadow_splat/`
- Temporary local archives and verification records: `/media/admin/data/.offload-staging/root-offload-20261110-three-projects`
- Selected data: `outputs`, `scripts/outputs`, `uncertainty_maps`.

**Status:** Google Drive confirmation is pending because its shared rclone API client is rate-limited. A local `*.prepared.json` record confirms an archive was compared against its source. A `*.completed.json` record is written only after Drive size/MD5 verification and removal of the corresponding local regular data files. Preparation alone does not mean a cloud backup has completed.

`data` already links to `/media/admin/data/StanfordMSL/shadow_splat/data`; that dataset is outside this root offload. Local Nerfstudio source edits are preserved in `patches/nerfstudio-local.patch`.

## Archive inventory

All selected archives have passed local content comparison. `data-archives-20261110.json` records their sizes and checksums; `data-archives-20261110.sha256` verifies downloaded archives independently of the staging drive. Cloud completion is still pending.

## Verify and restore

Restore only archives with verified/completed records. Check available disk space first. These archives contain repository-relative paths and preserve symlinks without copying their external targets. External link targets require their original media drive.

Download the archives and checksum records to a drive with sufficient space:

```bash
repo_root=$(git rev-parse --show-toplevel)
backup_dir=/media/admin/data/restore-shadow_splat-20261110
mkdir -p "$backup_dir"
rclone copy 'stanford-gdrive:root-offload-20261110-three-projects/shadow_splat' "$backup_dir" \
  --include '*.tar.zst' --include '*.verified.json' --include 'SHA256SUMS'
(cd "$backup_dir" && sha256sum --check "$repo_root/data-archives-20261110.sha256")
```

After checksums pass, run from a fresh checkout of this repository, or from the existing checkout after confirming the destination data directories are ordinary directories rather than staging symlinks:

```bash
for archive in "$backup_dir"/*.tar.zst; do
  tar --zstd --extract --skip-old-files --file "$archive"
done
```

`--skip-old-files` preserves files and links already present. To restore individual data files, first inspect archive paths with `tar --zstd --list --file ARCHIVE.tar.zst`, then pass the desired paths to the extraction command.
