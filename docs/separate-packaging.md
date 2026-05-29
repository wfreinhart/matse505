# Separate Packaging: CAMEL, Telemetry, Composable

This repo currently contains MATSE505 lecture content and CAMEL-related assets.  
Use `scripts/package_tracks.py` to stage CAMEL, telemetry, and composable courseware as separate directory trees you can move into another repo.

## What gets packaged

- `camel`: CAMEL spec/proposal files plus HSDemo authored/compiled/built artifacts.
- `telemetry`: `telemetry/` runtime code, telemetry activation module, and telemetry analysis notebooks/scripts.
- `composable`: composable courseware build scripts plus HSDemo modules/manifests and paired outputs.

All bundles explicitly exclude noisy/generated areas like `_staging/`, `telemetry_logs/`, and common environment/cache folders.

## Commands

List available bundle names:

```bash
python scripts/package_tracks.py --list
```

Preview what would be included:

```bash
python scripts/package_tracks.py --dry-run camel telemetry composable
```

Create all three bundle directories:

```bash
python scripts/package_tracks.py camel telemetry composable
```

Write bundle directories to a custom location:

```bash
python scripts/package_tracks.py --output-dir /tmp/matse505-exports camel telemetry composable
```

By default, directories are written to `dist/separated/`:

- `dist/separated/camel/`
- `dist/separated/telemetry/`
- `dist/separated/composable/`

Each directory includes a `PACKAGING_MANIFEST.json` file listing the staged files.

If you still need archives for handoff, add `--zip`:

```bash
python scripts/package_tracks.py --zip camel telemetry composable
```
