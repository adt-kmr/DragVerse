package com.twinforge.capture

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Writes the `.npz` files `reconstruction/fast_path/fusion.py` reads back: a zip of
 * `.npy` members named `depth`, `intrinsics`, `pose` (and optionally `color`).
 *
 * `.npy` v1.0 = magic + version + uint16 little-endian header length + a Python dict
 * literal, the whole preamble padded with spaces to a 64-byte multiple and terminated
 * by a newline, then the raw little-endian array data in C order.
 */
object NpzWriter {

    private val MAGIC = byteArrayOf(0x93.toByte(), 'N'.code.toByte(), 'U'.code.toByte(),
                                    'M'.code.toByte(), 'P'.code.toByte(), 'Y'.code.toByte())

    fun npy(dtype: String, shape: IntArray, data: ByteArray): ByteArray {
        // numpy needs the trailing comma on a 1-tuple; `(3, 3)` and `(480,)` both parse.
        val dims = shape.joinToString(", ") { it.toString() } + if (shape.size == 1) "," else ""
        val dict = "{'descr': '$dtype', 'fortran_order': False, 'shape': ($dims), }"

        val preamble = MAGIC.size + 2 + 2 + dict.length + 1
        val padding = (64 - preamble % 64) % 64
        val header = dict + " ".repeat(padding) + "\n"

        val out = ByteArrayOutputStream()
        out.write(MAGIC)
        out.write(1)  // major version
        out.write(0)  // minor version
        out.write(ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN)
            .putShort(header.length.toShort()).array())
        out.write(header.toByteArray(Charsets.US_ASCII))
        out.write(data)
        return out.toByteArray()
    }

    fun floats(values: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        values.forEach { buffer.putFloat(it) }
        return buffer.array()
    }

    /** Zip the named arrays into one `.npz`. Stored, not deflated — depth frames are
     *  noisy and the phone's battery is worth more than the bytes. */
    fun npz(members: Map<String, ByteArray>): ByteArray {
        val out = ByteArrayOutputStream()
        ZipOutputStream(out).use { zip ->
            zip.setMethod(ZipOutputStream.STORED)
            members.forEach { (name, bytes) ->
                val entry = ZipEntry("$name.npy").apply {
                    method = ZipEntry.STORED
                    size = bytes.size.toLong()
                    compressedSize = bytes.size.toLong()
                    crc = java.util.zip.CRC32().apply { update(bytes) }.value
                }
                zip.putNextEntry(entry)
                zip.write(bytes)
                zip.closeEntry()
            }
        }
        return out.toByteArray()
    }

    /** One RGB-D frame in the shape the fusion stage expects. */
    fun frame(depth: FloatArray, width: Int, height: Int,
              intrinsics: FloatArray, pose: FloatArray): ByteArray = npz(
        mapOf(
            "depth" to npy("<f4", intArrayOf(height, width), floats(depth)),
            "intrinsics" to npy("<f4", intArrayOf(3, 3), floats(intrinsics)),
            "pose" to npy("<f4", intArrayOf(4, 4), floats(pose)),
        )
    )
}
