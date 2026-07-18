package com.twinforge.capture

import android.media.Image
import com.google.ar.core.Camera
import com.google.ar.core.CameraIntrinsics

/**
 * ARCore frame -> the arrays `reconstruction/fast_path/fusion.py` back-projects.
 *
 * Two frame changes happen here, both once per frame and nowhere else downstream
 * (blueprint section 10):
 *
 *  1. Camera basis. ARCore's camera looks down -Z with +Y up; the pinhole model in
 *     `backproject()` looks down +Z with +Y down. That is a flip of Y and Z.
 *  2. World basis. ARCore's world is Y-up; the TwinForge map frame is Z-up,
 *     right-handed, metres. So (x, y, z)_arcore -> (x, -z, y)_map.
 */
object Frames {

    /** Map frame <- ARCore world, row-major. */
    private val MAP_FROM_AR = floatArrayOf(
        1f, 0f, 0f, 0f,
        0f, 0f, -1f, 0f,
        0f, 1f, 0f, 0f,
        0f, 0f, 0f, 1f,
    )

    /** ARCore camera <- OpenCV camera, row-major. */
    private val AR_FROM_CV = floatArrayOf(
        1f, 0f, 0f, 0f,
        0f, -1f, 0f, 0f,
        0f, 0f, -1f, 0f,
        0f, 0f, 0f, 1f,
    )

    /** DEPTH16 image -> metres, row-major, 0 where the sensor had no return. */
    fun depthMetres(image: Image): FloatArray {
        val plane = image.planes[0]
        val buffer = plane.buffer
        val strideShorts = plane.rowStride / 2
        val out = FloatArray(image.width * image.height)
        for (row in 0 until image.height) {
            for (col in 0 until image.width) {
                // DEPTH16 packs millimetres in the low 13 bits, confidence in the top 3.
                val raw = buffer.getShort((row * strideShorts + col) * 2).toInt() and 0x1FFF
                out[row * image.width + col] = raw / 1000f
            }
        }
        return out
    }

    /**
     * 3x3 pinhole intrinsics for the depth image.
     *
     * ARCore reports intrinsics against the CPU image, which is a different (larger)
     * resolution than the depth image — using them unscaled puts the principal point
     * off the depth map entirely.
     */
    fun intrinsics(camera: Camera, depthWidth: Int, depthHeight: Int): FloatArray {
        val k: CameraIntrinsics = camera.imageIntrinsics
        val focal = k.focalLength
        val principal = k.principalPoint
        val (imageWidth, imageHeight) = k.imageDimensions.let { it[0] to it[1] }
        val sx = depthWidth.toFloat() / imageWidth
        val sy = depthHeight.toFloat() / imageHeight
        return floatArrayOf(
            focal[0] * sx, 0f, principal[0] * sx,
            0f, focal[1] * sy, principal[1] * sy,
            0f, 0f, 1f,
        )
    }

    /** Camera -> world pose in the map frame, row-major 4x4. */
    fun pose(camera: Camera): FloatArray {
        val columnMajor = FloatArray(16)
        camera.pose.toMatrix(columnMajor, 0)
        val arPose = FloatArray(16)
        for (row in 0 until 4) {
            for (col in 0 until 4) {
                arPose[row * 4 + col] = columnMajor[col * 4 + row]
            }
        }
        return multiply(MAP_FROM_AR, multiply(arPose, AR_FROM_CV))
    }

    private fun multiply(a: FloatArray, b: FloatArray): FloatArray {
        val out = FloatArray(16)
        for (row in 0 until 4) {
            for (col in 0 until 4) {
                var sum = 0f
                for (k in 0 until 4) {
                    sum += a[row * 4 + k] * b[k * 4 + col]
                }
                out[row * 4 + col] = sum
            }
        }
        return out
    }
}
