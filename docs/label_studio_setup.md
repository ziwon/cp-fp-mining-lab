# Label Studio Setup

## Minimal local setup

```bash
docker run -it -p 8080:8080 -v $(pwd)/label-studio-data:/label-studio/data heartexlabs/label-studio:latest
```

Open `http://localhost:8080` and create a project. Generate both files with:

```bash
uv run python scripts/03_export_for_label_studio.py
```

### Two separate steps (do not mix them up)

The XML and the JSON serve different roles and go to different places. Importing
the XML as data — or skipping the interface setup — is the most common mistake; if
the labeling interface is empty, clicking a task redirects you to "Go to setup".

1. **Labeling interface** — set the *form*, not data.
   `Settings → Labeling Interface → Code`, paste the contents of
   `data/processed/labelstudio_label_config.xml`, then `Save`.

2. **Tasks** — import the *data*.
   `Import` → upload `data/processed/labelstudio_tasks.json`.

In the Docker Compose stack you can set the interface for project 1 directly
instead of pasting (the API rejects legacy tokens, so use the ORM):

```bash
docker compose cp data/processed/labelstudio_label_config.xml \
  labelstudio:/tmp/label_config.xml
docker compose exec -T labelstudio python label_studio/manage.py shell -c \
  "from projects.models import Project; p=Project.objects.get(id=1); \
   p.label_config=open('/tmp/label_config.xml').read(); p.save()"
```

### Images don't render?

Task images are served by the `fileserver` container over plain HTTP (Label
Studio's `local-files` endpoint requires a registered Local Storage and otherwise
returns 401). Two things must hold:

- The browser must reach `http_base_url` (default `http://localhost:18090`); set it
  in `configs/pipeline.yaml` to match how you open the UI, then re-run script 03.
- The fileserver sends `Access-Control-Allow-Origin` (see `docker/fileserver.conf`).
  Label Studio loads images with `crossorigin="anonymous"`, so without CORS the
  browser discards the image as broken **even though the GET returns 200**.

## Review task design

The review UI asks:

- Is this a real event?
- If false positive, why?
- Was the predicted bbox useful?
- Reviewer comment

## Real deployment

For production, store images in object storage and pass signed URLs or internal accessible paths to Label Studio.

```text
S3/MinIO path
→ Label Studio task image URL
→ reviewer annotation
→ export JSON
→ dataset builder
→ W&B Artifact
```
