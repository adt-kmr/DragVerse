package com.twinforge.capture

import android.Manifest
import android.content.pm.PackageManager
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Point the phone at a room; every captured ARCore depth frame is packed as an `.npz`
 * and chunk-uploaded to the orchestrator's `POST /capture`.
 *
 * ponytail: no camera preview is drawn — the GL surface exists only because ARCore
 * requires a texture bound before `session.update()`. Tracking state and frame count on
 * screen are the operator's feedback. Add a background quad shader if scanning by feel
 * turns out to be too hard in practice.
 */
class MainActivity : AppCompatActivity(), GLSurfaceView.Renderer {

    private lateinit var surface: GLSurfaceView
    private lateinit var stats: TextView
    private var session: Session? = null

    private val uploader = Uploader()
    // Serial: keeps chunk indices monotonic and stops a fast scan from queueing the
    // whole room in flight over hotel wifi.
    private val uploads = Executors.newSingleThreadExecutor()
    private val nextIndex = AtomicInteger(0)
    private val uploaded = AtomicInteger(0)

    @Volatile private var capturing = true
    private var lastCaptureNanos = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        surface = findViewById(R.id.surface)
        stats = findViewById(R.id.stats)

        surface.setEGLContextClientVersion(2)
        surface.setRenderer(this)
        surface.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY

        findViewById<Button>(R.id.finish).setOnClickListener { finishScan() }

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
        }
    }

    override fun onResume() {
        super.onResume()
        if (session == null) {
            session = Session(this).apply {
                configure(config.apply {
                    depthMode = Config.DepthMode.AUTOMATIC
                    focusMode = Config.FocusMode.AUTO
                    updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                })
            }
        }
        session?.resume()
        surface.onResume()
    }

    override fun onPause() {
        super.onPause()
        surface.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        super.onDestroy()
        uploads.shutdown()
        session?.close()
    }

    // ------------------------------------------------------------------ GL / capture

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textures[0])
        session?.setCameraTextureName(textures[0])
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        session?.setDisplayGeometry(display.rotation, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        val session = session ?: return
        val frame = try {
            session.update()
        } catch (e: Exception) {
            Log.w(TAG, "session update failed", e)
            return
        }

        val camera = frame.camera
        if (!capturing || camera.trackingState != TrackingState.TRACKING) {
            report(camera.trackingState.name)
            return
        }
        // ~3 fps: the depth maps overlap heavily above that and fusion just dedups them
        // back out, so the extra frames cost upload bandwidth and buy nothing.
        val now = System.nanoTime()
        if (now - lastCaptureNanos < CAPTURE_INTERVAL_NANOS) return
        lastCaptureNanos = now

        val image = try {
            frame.acquireDepthImage16Bits()
        } catch (e: NotYetAvailableException) {
            report("waiting for depth")
            return
        }

        val npz = image.use {
            NpzWriter.frame(
                depth = Frames.depthMetres(it),
                width = it.width,
                height = it.height,
                intrinsics = Frames.intrinsics(camera, it.width, it.height),
                pose = Frames.pose(camera),
            )
        }

        val index = nextIndex.getAndIncrement()
        uploads.execute {
            try {
                uploader.uploadFrame(index, npz)
                report("frames ${uploaded.incrementAndGet()} · ${uploader.scanId.orEmpty()}")
            } catch (e: Exception) {
                // Indices are stable, so a failed chunk can simply be re-sent later;
                // dropping it here keeps the scan going rather than aborting the room.
                Log.w(TAG, "chunk $index failed", e)
                report("frames ${uploaded.get()} · retrying")
            }
        }
    }

    private fun finishScan() {
        capturing = false
        uploads.execute {
            try {
                val count = uploader.complete()
                report("scan ${uploader.scanId} complete: $count frames")
            } catch (e: Exception) {
                Log.e(TAG, "complete failed", e)
                report("complete failed: ${e.message}")
            }
        }
    }

    private fun report(text: String) = runOnUiThread { stats.text = text }

    private companion object {
        const val TAG = "TwinForgeCapture"
        const val CAPTURE_INTERVAL_NANOS = 333_000_000L
    }
}
