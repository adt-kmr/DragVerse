import React, { useState } from "react";

import * as api from "../../api.js";

/** uuid.uuid4().hex on the backend (orchestrator/db.py:new_id) — match the format. */
const newScanId = () => crypto.randomUUID().replace(/-/g, "");

/**
 * Scaniverse-import path only (Task 16): /capture/{scan_id}/import takes a server-local
 * export_path, not an uploaded blob (see capture/scaniverse.py's ingest_export, which
 * shutil.copyfile()s from that path). A browser file input can't hand us an absolute
 * filesystem path for security reasons, so the source of truth here is a text field;
 * the file picker is a convenience that pre-fills a filename to start from.
 *
 * On success, chains reconstruct -> segment -> generate-twin (mode="fast", no
 * per-substep UI — the v3 spec frames this as invisible post-capture processing) and
 * hands every id downstream steps need to onNext.
 */
export default function CaptureStep({ onNext }) {
  const [scanId] = useState(newScanId);
  const [exportPath, setExportPath] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const pickFile = (e) => {
    const file = e.target.files?.[0];
    if (file && !exportPath) setExportPath(file.name);
  };

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.captureImport(scanId, exportPath.trim());
      const mesh = await api.reconstruct(scanId, "fast");
      const seg = await api.segment(mesh.mesh_id);
      const twin = await api.generateTwin(mesh.mesh_id, seg.objects_id);
      onNext({
        scanId,
        meshId: mesh.mesh_id,
        objectsId: seg.objects_id,
        twinId: twin.twin_id,
      });
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="wizard__step">
      <h2>Scan the space</h2>
      <p className="prose">
        Export a point cloud from Scaniverse as a <code>.ply</code>, then point us at it.
        We'll reconstruct the mesh, label what's in it, and build the twin automatically.
      </p>

      <label>
        .ply file
        <input type="file" accept=".ply" onChange={pickFile} disabled={busy} />
      </label>
      <label>
        Export path
        <input
          value={exportPath}
          onChange={(e) => setExportPath(e.target.value)}
          placeholder="/path/to/export.ply"
          disabled={busy}
        />
      </label>
      <p className="hint">Path the orchestrator can read — same machine or a shared volume.</p>

      <button
        className="btn btn--solid"
        onClick={run}
        disabled={busy || exportPath.trim().length === 0}
      >
        {busy ? "Processing your scan…" : "Continue"}
      </button>

      {error && (
        <div className="error">
          <strong>{error.message}</strong>
          {typeof error.detail === "object" && error.detail !== null && (
            <dl>
              {Object.entries(error.detail)
                .filter(([key]) => key !== "error")
                .map(([key, value]) => (
                  <React.Fragment key={key}>
                    <dt>{key.replace(/_/g, " ")}</dt>
                    <dd>{String(value)}</dd>
                  </React.Fragment>
                ))}
            </dl>
          )}
        </div>
      )}
    </div>
  );
}
